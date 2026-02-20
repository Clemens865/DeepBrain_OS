//! File indexer for DeepBrain
//!
//! Watches filesystem, chunks files, and indexes them with vector embeddings
//! for semantic file search.

pub mod chunker;
pub mod email;
pub mod parser;
pub mod watcher;

use std::path::{Path, PathBuf};
use std::sync::Arc;

use parking_lot::RwLock;
use rusqlite::{params, Connection};
use serde::{Deserialize, Serialize};

use crate::brain::embeddings::EmbeddingModel;
use crate::brain::utils::cosine_similarity;
use crate::deepbrain::vector_store::{DeepBrainVectorStore, VectorFilter, VectorMetadata};

/// File search result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FileResult {
    pub path: String,
    pub name: String,
    pub chunk: String,
    pub similarity: f64,
    pub file_type: String,
}

/// File index entry stored in SQLite
#[derive(Debug, Clone)]
struct FileEntry {
    path: String,
    name: String,
    ext: String,
    modified: i64,
    chunk_count: u32,
}

/// File chunk with embedding
#[derive(Debug, Clone)]
struct FileChunk {
    file_path: String,
    chunk_index: u32,
    content: String,
    vector: Vec<f32>,
}

/// The file indexer manages scanning, watching, and searching files
pub struct FileIndexer {
    db_path: PathBuf,
    watched_dirs: RwLock<Vec<PathBuf>>,
    embeddings: Arc<EmbeddingModel>,
    is_indexing: RwLock<bool>,
    /// RVF-backed vector store (None = brute-force fallback).
    vector_store: Option<Arc<DeepBrainVectorStore>>,
}

impl FileIndexer {
    /// Create a new file indexer (brute-force fallback, no RVF store).
    pub fn new(db_path: PathBuf, embeddings: Arc<EmbeddingModel>) -> Result<Self, String> {
        let indexer = Self {
            db_path,
            watched_dirs: RwLock::new(Vec::new()),
            embeddings,
            is_indexing: RwLock::new(false),
            vector_store: None,
        };
        indexer.initialize_db()?;
        Ok(indexer)
    }

    /// Create a new file indexer backed by a DeepBrain RVF vector store.
    pub fn with_vector_store(
        db_path: PathBuf,
        embeddings: Arc<EmbeddingModel>,
        store: Arc<DeepBrainVectorStore>,
    ) -> Result<Self, String> {
        let indexer = Self {
            db_path,
            watched_dirs: RwLock::new(Vec::new()),
            embeddings,
            is_indexing: RwLock::new(false),
            vector_store: Some(store),
        };
        indexer.initialize_db()?;
        Ok(indexer)
    }

    /// Build vector index from existing file_chunks in SQLite.
    ///
    /// When an RVF vector store is configured, this is a no-op because
    /// RVF boots automatically from its manifest. When no store is configured,
    /// this is also a no-op (brute-force scan doesn't need pre-building).
    pub fn build_hnsw_index(&self) -> Result<usize, String> {
        if self.vector_store.is_some() {
            // RVF store boots from manifest — no need to rebuild.
            tracing::info!("RVF vector store active — skipping HNSW rebuild");
            return Ok(0);
        }
        // No index to build in brute-force mode.
        Ok(0)
    }

    fn open_connection(&self) -> Result<Connection, String> {
        Connection::open(&self.db_path).map_err(|e| format!("DB open failed: {}", e))
    }

    fn initialize_db(&self) -> Result<(), String> {
        let conn = self.open_connection()?;
        conn.execute_batch(
            "
            PRAGMA journal_mode=WAL;

            CREATE TABLE IF NOT EXISTS file_index (
                path TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                ext TEXT NOT NULL,
                modified INTEGER NOT NULL,
                chunk_count INTEGER NOT NULL DEFAULT 0
            );

            CREATE TABLE IF NOT EXISTS file_chunks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                file_path TEXT NOT NULL,
                chunk_index INTEGER NOT NULL,
                content TEXT NOT NULL,
                vector BLOB NOT NULL,
                FOREIGN KEY (file_path) REFERENCES file_index(path) ON DELETE CASCADE
            );

            CREATE INDEX IF NOT EXISTS idx_chunks_path ON file_chunks(file_path);
            ",
        )
        .map_err(|e| format!("DB init failed: {}", e))?;
        Ok(())
    }

    /// Add directories to watch
    pub fn add_watch_dirs(&self, dirs: Vec<PathBuf>) {
        let mut watched = self.watched_dirs.write();
        for dir in dirs {
            if dir.exists() && !watched.contains(&dir) {
                watched.push(dir);
            }
        }
    }

    /// Index a single file
    pub async fn index_file(&self, path: &Path) -> Result<u32, String> {
        let ext = path
            .extension()
            .and_then(|e| e.to_str())
            .unwrap_or("")
            .to_lowercase();

        if !parser::is_supported(&ext) {
            return Ok(0);
        }

        let content = parser::parse_file(path)?;
        if content.trim().is_empty() {
            return Ok(0);
        }

        let chunks = chunker::chunk_text(&content, 512, 128);
        if chunks.is_empty() {
            return Ok(0);
        }

        let name = path
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("unknown")
            .to_string();

        let modified = path
            .metadata()
            .map(|m| {
                m.modified()
                    .ok()
                    .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
                    .map(|d| d.as_secs() as i64)
                    .unwrap_or(0)
            })
            .unwrap_or(0);

        let path_str = path.to_string_lossy().to_string();

        // Embed all chunks
        let mut file_chunks = Vec::with_capacity(chunks.len());
        for (i, chunk) in chunks.iter().enumerate() {
            let vector = self.embeddings.embed(chunk).await?;
            file_chunks.push(FileChunk {
                file_path: path_str.clone(),
                chunk_index: i as u32,
                content: chunk.clone(),
                vector,
            });
        }

        // Store in database
        let conn = self.open_connection()?;

        // Upsert file entry
        conn.execute(
            "INSERT OR REPLACE INTO file_index (path, name, ext, modified, chunk_count) VALUES (?1, ?2, ?3, ?4, ?5)",
            params![path_str, name, ext, modified, chunks.len() as u32],
        )
        .map_err(|e| format!("Store file failed: {}", e))?;

        // Delete old chunks
        conn.execute(
            "DELETE FROM file_chunks WHERE file_path = ?1",
            params![path_str],
        )
        .map_err(|e| format!("Delete chunks failed: {}", e))?;

        // Insert new chunks (SQLite + RVF vector store)
        for chunk in &file_chunks {
            let vector_bytes = vector_to_bytes(&chunk.vector);
            conn.execute(
                "INSERT INTO file_chunks (file_path, chunk_index, content, vector) VALUES (?1, ?2, ?3, ?4)",
                params![chunk.file_path, chunk.chunk_index, chunk.content, vector_bytes],
            )
            .map_err(|e| format!("Store chunk failed: {}", e))?;

            // Also insert into RVF vector store for indexed search.
            if let Some(ref store) = self.vector_store {
                let chunk_id = format!("file::{}::{}", chunk.file_path, chunk.chunk_index);
                let meta = VectorMetadata {
                    source: "file".to_string(),
                    path: chunk.file_path.clone(),
                    timestamp: modified,
                    ..Default::default()
                };
                if let Err(e) = store.store_vector(&chunk_id, &chunk.vector, meta) {
                    tracing::warn!("Failed to store file chunk in RVF: {}", e);
                }
            }
        }

        Ok(file_chunks.len() as u32)
    }

    /// Scan and index all files in watched directories (recursive)
    pub async fn scan_all(&self) -> Result<u32, String> {
        {
            let is_indexing = self.is_indexing.read();
            if *is_indexing {
                return Err("Indexing already in progress".to_string());
            }
        }
        *self.is_indexing.write() = true;

        let dirs: Vec<PathBuf> = self.watched_dirs.read().clone();
        let mut total = 0u32;

        // Collect all files recursively first
        let mut files = Vec::new();
        for dir in &dirs {
            collect_files_recursive(dir, &mut files, 10);
        }

        tracing::info!("Found {} files to index", files.len());

        for path in &files {
            match self.index_file(path).await {
                Ok(chunks) => total += chunks,
                Err(e) => tracing::debug!("Skipped {:?}: {}", path, e),
            }
        }

        *self.is_indexing.write() = false;
        tracing::info!("Indexed {} chunks from {} files", total, files.len());
        Ok(total)
    }

    /// Search indexed files by semantic similarity.
    /// Uses RVF vector store for O(log n) search when available, falls back to brute-force.
    pub async fn search(&self, query: &str, limit: u32) -> Result<Vec<FileResult>, String> {
        let query_vector = self.embeddings.embed(query).await?;

        // Try RVF vector store first (fast O(log n))
        if let Some(ref store) = self.vector_store {
            return self.search_rvf(store, &query_vector, limit);
        }

        // Fallback: brute-force scan of all chunks
        self.search_brute_force(&query_vector, limit)
    }

    /// RVF-accelerated file search.
    fn search_rvf(
        &self,
        store: &DeepBrainVectorStore,
        query_vector: &[f32],
        limit: u32,
    ) -> Result<Vec<FileResult>, String> {
        // Over-fetch to allow for type boosting to reorder results.
        let fetch_k = (limit as usize) * 3;
        let filter = Some(VectorFilter::Source("file".to_string()));

        let rvf_results = store
            .search(query_vector, fetch_k, filter)
            .map_err(|e| format!("RVF search failed: {}", e))?;

        let conn = self.open_connection()?;

        let mut results: Vec<FileResult> = Vec::with_capacity(rvf_results.len());
        for vr in &rvf_results {
            if vr.similarity < 0.25 {
                continue;
            }

            // Parse the RVF ID: "file::{file_path}::{chunk_index}"
            let id_str = &vr.id;
            let stripped = id_str.strip_prefix("file::").unwrap_or(id_str);
            let parts: Vec<&str> = stripped.rsplitn(2, "::").collect();
            if parts.len() != 2 {
                continue;
            }
            let file_path = parts[1];
            let chunk_index: u32 = match parts[0].parse() {
                Ok(i) => i,
                Err(_) => continue,
            };

            // Look up content and metadata from SQLite.
            let row_result = conn.query_row(
                "SELECT fc.content, fi.name, fi.ext
                 FROM file_chunks fc
                 JOIN file_index fi ON fc.file_path = fi.path
                 WHERE fc.file_path = ?1 AND fc.chunk_index = ?2",
                params![file_path, chunk_index],
                |row| {
                    Ok((
                        row.get::<_, String>(0)?,
                        row.get::<_, String>(1)?,
                        row.get::<_, String>(2)?,
                    ))
                },
            );

            if let Ok((content, name, ext)) = row_result {
                results.push(FileResult {
                    path: file_path.to_string(),
                    name,
                    chunk: content,
                    similarity: vr.similarity,
                    file_type: ext,
                });
            }
        }

        // Apply file-type boost.
        for result in &mut results {
            match result.file_type.as_str() {
                "md" | "txt" | "pdf" | "rtf" | "docx" | "doc" => {
                    result.similarity *= 1.15;
                }
                "rs" | "ts" | "tsx" | "js" | "jsx" | "py" | "go" | "java" | "c" | "cpp"
                | "h" | "swift" => {
                    result.similarity *= 0.95;
                }
                _ => {}
            }
        }

        results.sort_by(|a, b| {
            b.similarity
                .partial_cmp(&a.similarity)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        results.truncate(limit as usize);

        Ok(results)
    }

    /// Brute-force file search (fallback when HNSW not built)
    fn search_brute_force(
        &self,
        query_vector: &[f32],
        limit: u32,
    ) -> Result<Vec<FileResult>, String> {
        let conn = self.open_connection()?;
        let mut stmt = conn
            .prepare(
                "SELECT fc.file_path, fc.content, fc.vector, fi.name, fi.ext
                 FROM file_chunks fc
                 JOIN file_index fi ON fc.file_path = fi.path",
            )
            .map_err(|e| format!("Query failed: {}", e))?;

        let mut results: Vec<FileResult> = stmt
            .query_map([], |row| {
                let file_path: String = row.get(0)?;
                let content: String = row.get(1)?;
                let vector_bytes: Vec<u8> = row.get(2)?;
                let name: String = row.get(3)?;
                let ext: String = row.get(4)?;

                let vector = bytes_to_vector(&vector_bytes);
                let similarity = cosine_similarity(query_vector, &vector) as f64;

                Ok(FileResult {
                    path: file_path,
                    name,
                    chunk: content,
                    similarity,
                    file_type: ext,
                })
            })
            .map_err(|e| format!("Search failed: {}", e))?
            .filter_map(|r| r.ok())
            .filter(|r| r.similarity > 0.25)
            .collect();

        // Boost documentation files by 15%, slight penalty for code files
        for result in &mut results {
            match result.file_type.as_str() {
                "md" | "txt" | "pdf" | "rtf" | "docx" | "doc" => {
                    result.similarity *= 1.15;
                }
                "rs" | "ts" | "tsx" | "js" | "jsx" | "py" | "go" | "java" | "c" | "cpp"
                | "h" | "swift" => {
                    result.similarity *= 0.95;
                }
                _ => {}
            }
        }

        results.sort_by(|a, b| {
            b.similarity
                .partial_cmp(&a.similarity)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        results.truncate(limit as usize);

        Ok(results)
    }

    /// List indexed files for browsing
    pub fn list_files(&self, folder: Option<&str>, limit: u32, offset: u32) -> Result<Vec<FileListItem>, String> {
        let conn = self.open_connection()?;

        let (sql, params_vec): (String, Vec<Box<dyn rusqlite::types::ToSql>>) = if let Some(folder) = folder {
            let pattern = format!("{}%", folder);
            (
                "SELECT path, name, ext, modified, chunk_count FROM file_index WHERE path LIKE ?1 ORDER BY modified DESC LIMIT ?2 OFFSET ?3".to_string(),
                vec![Box::new(pattern) as Box<dyn rusqlite::types::ToSql>, Box::new(limit), Box::new(offset)],
            )
        } else {
            (
                "SELECT path, name, ext, modified, chunk_count FROM file_index ORDER BY modified DESC LIMIT ?1 OFFSET ?2".to_string(),
                vec![Box::new(limit) as Box<dyn rusqlite::types::ToSql>, Box::new(offset)],
            )
        };

        let mut stmt = conn.prepare(&sql).map_err(|e| format!("Query failed: {}", e))?;
        let params_refs: Vec<&dyn rusqlite::types::ToSql> = params_vec.iter().map(|p| p.as_ref()).collect();

        let items = stmt
            .query_map(params_refs.as_slice(), |row| {
                Ok(FileListItem {
                    path: row.get(0)?,
                    name: row.get(1)?,
                    ext: row.get(2)?,
                    modified: row.get(3)?,
                    chunk_count: row.get(4)?,
                })
            })
            .map_err(|e| format!("List files failed: {}", e))?
            .filter_map(|r| r.ok())
            .collect();

        Ok(items)
    }

    /// Get chunks for a specific file (for preview)
    pub fn get_file_chunks(&self, file_path: &str, limit: u32) -> Result<Vec<FileChunkItem>, String> {
        let conn = self.open_connection()?;
        let mut stmt = conn
            .prepare("SELECT chunk_index, content FROM file_chunks WHERE file_path = ?1 ORDER BY chunk_index ASC LIMIT ?2")
            .map_err(|e| format!("Query failed: {}", e))?;

        let items = stmt
            .query_map(params![file_path, limit], |row| {
                Ok(FileChunkItem {
                    chunk_index: row.get(0)?,
                    content: row.get(1)?,
                })
            })
            .map_err(|e| format!("Get chunks failed: {}", e))?
            .filter_map(|r| r.ok())
            .collect();

        Ok(items)
    }

    /// Sample random vectors from the file index for SONA bootstrapping.
    /// Returns (embedding, modified_timestamp) pairs.
    pub fn sample_vectors(&self, count: u32) -> Result<Vec<(Vec<f32>, i64)>, String> {
        let conn = self.open_connection()?;
        let mut stmt = conn
            .prepare(
                "SELECT fc.vector, fi.modified
                 FROM file_chunks fc
                 JOIN file_index fi ON fc.file_path = fi.path
                 ORDER BY RANDOM()
                 LIMIT ?1",
            )
            .map_err(|e| format!("sample_vectors query failed: {}", e))?;

        let rows = stmt
            .query_map(params![count], |row| {
                let vector_bytes: Vec<u8> = row.get(0)?;
                let modified: i64 = row.get(1)?;
                Ok((bytes_to_vector(&vector_bytes), modified))
            })
            .map_err(|e| format!("sample_vectors failed: {}", e))?
            .filter_map(|r| r.ok())
            .collect();

        Ok(rows)
    }

    /// Get index statistics
    pub fn stats(&self) -> Result<IndexStats, String> {
        let conn = self.open_connection()?;

        let file_count: u32 = conn
            .query_row("SELECT COUNT(*) FROM file_index", [], |row| row.get(0))
            .unwrap_or(0);

        let chunk_count: u32 = conn
            .query_row("SELECT COUNT(*) FROM file_chunks", [], |row| row.get(0))
            .unwrap_or(0);

        let is_indexing = *self.is_indexing.read();

        Ok(IndexStats {
            file_count,
            chunk_count,
            watched_dirs: self.watched_dirs.read().len() as u32,
            is_indexing,
        })
    }
}

/// File list item for browsing
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FileListItem {
    pub path: String,
    pub name: String,
    pub ext: String,
    pub modified: i64,
    pub chunk_count: u32,
}

/// File chunk content for preview
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FileChunkItem {
    pub chunk_index: u32,
    pub content: String,
}

/// Index statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IndexStats {
    pub file_count: u32,
    pub chunk_count: u32,
    pub watched_dirs: u32,
    pub is_indexing: bool,
}

fn vector_to_bytes(vector: &[f32]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(vector.len() * 4);
    for &val in vector {
        bytes.extend_from_slice(&val.to_le_bytes());
    }
    bytes
}

fn bytes_to_vector(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect()
}

/// Directories to skip during recursive scanning
const SKIP_DIRS: &[&str] = &[
    "node_modules",
    "target",
    ".git",
    ".svn",
    ".hg",
    "__pycache__",
    ".venv",
    "venv",
    ".cache",
    "build",
    "dist",
    ".Trash",
];

/// Recursively collect files, skipping hidden/undesirable directories
fn collect_files_recursive(dir: &Path, files: &mut Vec<PathBuf>, max_depth: u32) {
    if max_depth == 0 {
        return;
    }

    let entries = match std::fs::read_dir(dir) {
        Ok(e) => e,
        Err(_) => return,
    };

    for entry in entries.flatten() {
        let path = entry.path();
        let name = entry.file_name();
        let name_str = name.to_string_lossy();

        // Skip hidden files/dirs
        if name_str.starts_with('.') {
            continue;
        }

        if path.is_dir() {
            // Skip known undesirable directories
            if SKIP_DIRS.contains(&name_str.as_ref()) {
                continue;
            }
            collect_files_recursive(&path, files, max_depth - 1);
        } else if path.is_file() {
            files.push(path);
        }
    }
}
