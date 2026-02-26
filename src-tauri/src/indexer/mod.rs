//! File indexer for DeepBrain
//!
//! Watches filesystem, chunks files, and indexes them with vector embeddings
//! for semantic file search.

pub mod bootstrap;
pub mod browser;
pub mod chunker;
pub mod claude_listener;
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
    // Enhanced fields — only populated when include_preview=true
    #[serde(skip_serializing_if = "Option::is_none")]
    pub file_size_bytes: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub modified: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub project: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub line_count: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub preview_lines: Option<Vec<String>>,
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

    /// Index a single file: parse, embed, store in SQLite.
    ///
    /// Returns the file chunks and their RVF batch entries for deferred RVF write.
    /// The caller is responsible for flushing the RVF batch entries periodically
    /// to avoid O(n²) manifest growth from too many small epochs.
    async fn index_file_inner(
        &self,
        path: &Path,
    ) -> Result<(u32, Vec<(String, Vec<f32>, VectorMetadata)>), String> {
        let ext = path
            .extension()
            .and_then(|e| e.to_str())
            .unwrap_or("")
            .to_lowercase();

        if !parser::is_supported(&ext) {
            return Ok((0, vec![]));
        }

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

        // Skip re-indexing if the file hasn't changed since last index.
        // The modification time is already stored in SQLite — just compare it.
        {
            let conn = self.open_connection()?;
            let stored_mtime: Option<i64> = conn
                .query_row(
                    "SELECT modified FROM file_index WHERE path = ?1",
                    params![path_str],
                    |row| row.get(0),
                )
                .ok();

            if stored_mtime == Some(modified) {
                return Ok((0, vec![]));
            }
        }

        let content = parser::parse_file(path)?;
        if content.trim().is_empty() {
            return Ok((0, vec![]));
        }

        // Prepend file path to content so the location becomes part of the
        // semantic knowledge. This lets searches like "files in Context-Capture"
        // or "documents on Desktop" return relevant results.
        let content_with_path = format!("[{}]\n{}", path_str, content);

        let chunks = chunker::chunk_text(&content_with_path, 512, 128);
        if chunks.is_empty() {
            return Ok((0, vec![]));
        }

        let name = path
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("unknown")
            .to_string();

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

        // Store in SQLite database
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

        // Insert new chunks into SQLite
        for chunk in &file_chunks {
            let vector_bytes = vector_to_bytes(&chunk.vector);
            conn.execute(
                "INSERT INTO file_chunks (file_path, chunk_index, content, vector) VALUES (?1, ?2, ?3, ?4)",
                params![chunk.file_path, chunk.chunk_index, chunk.content, vector_bytes],
            )
            .map_err(|e| format!("Store chunk failed: {}", e))?;
        }

        // Build RVF batch entries (deferred — caller flushes them)
        let rvf_batch: Vec<(String, Vec<f32>, VectorMetadata)> = file_chunks
            .iter()
            .map(|chunk| {
                let chunk_id = format!("file::{}::{}", chunk.file_path, chunk.chunk_index);
                let meta = VectorMetadata {
                    source: "file".to_string(),
                    path: chunk.file_path.clone(),
                    timestamp: modified,
                    ..Default::default()
                };
                (chunk_id, chunk.vector.clone(), meta)
            })
            .collect();

        Ok((file_chunks.len() as u32, rvf_batch))
    }

    /// Index a single file (public API — for individual file updates via watcher).
    ///
    /// Immediately flushes to the vector store. For bulk scans, use `scan_all`
    /// which accumulates vectors across files for efficient batch writes.
    pub async fn index_file(&self, path: &Path) -> Result<u32, String> {
        // Apply the same filtering as the watcher (for callers that bypass it)
        if !watcher::should_index(path) {
            return Ok(0);
        }

        let (count, rvf_batch) = self.index_file_inner(path).await?;
        if !rvf_batch.is_empty() {
            if let Some(ref store) = self.vector_store {
                if let Err(e) = store.store_batch(&rvf_batch) {
                    tracing::warn!("Failed to store file chunks in RVF: {}", e);
                }
            }
        }
        Ok(count)
    }

    /// Scan specific directories (not all watched dirs).
    /// Useful for startup scan that should only cover fast local dirs.
    pub async fn scan_dirs(&self, dirs: &[PathBuf]) -> Result<u32, String> {
        self.scan_impl(dirs).await
    }

    /// Scan and index all files in watched directories (recursive).
    ///
    /// Accumulates RVF vectors across files and flushes every `RVF_FLUSH_INTERVAL`
    /// files to minimise the number of epochs (each epoch writes a full manifest,
    /// so fewer epochs = dramatically smaller file).
    pub async fn scan_all(&self) -> Result<u32, String> {
        let dirs: Vec<PathBuf> = self.watched_dirs.read().clone();
        self.scan_impl(&dirs).await
    }

    /// Internal scan implementation shared by scan_all and scan_dirs.
    async fn scan_impl(&self, dirs: &[PathBuf]) -> Result<u32, String> {
        /// Flush RVF batch every N files to keep epoch count low.
        const RVF_FLUSH_INTERVAL: usize = 500;

        {
            let is_indexing = self.is_indexing.read();
            if *is_indexing {
                return Err("Indexing already in progress".to_string());
            }
        }
        *self.is_indexing.write() = true;

        let mut total = 0u32;

        // Collect all files recursively first
        let mut files = Vec::new();
        for dir in dirs {
            collect_files_recursive(dir, &mut files, 10);
        }

        // Apply the same filtering as the watcher
        let before_filter = files.len();
        files.retain(|p| watcher::should_index(p));
        tracing::info!(
            "Found {} files to index ({} filtered out)",
            files.len(),
            before_filter - files.len()
        );

        // Accumulate batch entries across files
        let mut rvf_pending: Vec<(String, Vec<f32>, VectorMetadata)> = Vec::new();
        let mut files_since_flush = 0usize;

        for (i, path) in files.iter().enumerate() {
            match self.index_file_inner(path).await {
                Ok((chunks, rvf_batch)) => {
                    total += chunks;
                    rvf_pending.extend(rvf_batch);
                    files_since_flush += 1;
                }
                Err(e) => tracing::debug!("Skipped {:?}: {}", path, e),
            }

            // Flush accumulated vectors to RVF every N files
            if files_since_flush >= RVF_FLUSH_INTERVAL && !rvf_pending.is_empty() {
                if let Some(ref store) = self.vector_store {
                    let batch_len = rvf_pending.len();
                    if let Err(e) = store.store_batch(&rvf_pending) {
                        tracing::warn!("Failed to flush RVF batch: {}", e);
                    } else {
                        tracing::info!(
                            "RVF flush: {} vectors from {} files (progress: {}/{})",
                            batch_len, files_since_flush, i + 1, files.len()
                        );
                    }
                }
                rvf_pending.clear();
                files_since_flush = 0;
            }

            // Yield to the async runtime every 10 files
            if i % 10 == 9 {
                tokio::task::yield_now().await;
            }
        }

        // Final flush of remaining vectors
        if !rvf_pending.is_empty() {
            if let Some(ref store) = self.vector_store {
                let batch_len = rvf_pending.len();
                if let Err(e) = store.store_batch(&rvf_pending) {
                    tracing::warn!("Failed to flush final RVF batch: {}", e);
                } else {
                    tracing::info!("RVF final flush: {} vectors from {} files", batch_len, files_since_flush);
                }
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
            let results = self.search_rvf(store, &query_vector, limit)?;
            if !results.is_empty() {
                return Ok(results);
            }
            // RVF returned nothing — fall through to brute-force.
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

        // Try metadata-filtered search first; fall back to unfiltered + ID-prefix
        // filtering if metadata filters aren't supported (degraded/flat mode).
        let rvf_results = {
            let filtered = store
                .search(query_vector, fetch_k, Some(VectorFilter::Source("file".to_string())))
                .map_err(|e| format!("RVF search failed: {}", e))?;
            if filtered.is_empty() {
                store
                    .search(query_vector, fetch_k * 10, None)
                    .map_err(|e| format!("RVF search (unfiltered) failed: {}", e))?
                    .into_iter()
                    .filter(|r| r.id.starts_with("file::"))
                    .collect()
            } else {
                filtered
            }
        };

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
                    file_size_bytes: None,
                    modified: None,
                    project: None,
                    line_count: None,
                    preview_lines: None,
                });
            }
        }

        // Apply file-type boost.
        for result in &mut results {
            match result.file_type.as_str() {
                "md" | "txt" | "pdf" | "rtf" | "docx" | "doc" | "xlsx" | "xls" | "csv" | "ods" => {
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
                    file_size_bytes: None,
                    modified: None,
                    project: None,
                    line_count: None,
                    preview_lines: None,
                })
            })
            .map_err(|e| format!("Search failed: {}", e))?
            .filter_map(|r| r.ok())
            .filter(|r| r.similarity > 0.25)
            .collect();

        // Boost documentation files by 15%, slight penalty for code files
        for result in &mut results {
            match result.file_type.as_str() {
                "md" | "txt" | "pdf" | "rtf" | "docx" | "doc" | "xlsx" | "xls" | "csv" | "ods" => {
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

    /// Detect projects from indexed file paths.
    ///
    /// Groups all files in `file_index` by their detected project root and
    /// returns aggregate counts. Computes at query time (no schema change).
    pub fn projects(&self) -> Result<Vec<crate::activity::ProjectInfo>, String> {
        let conn = self.open_connection()?;
        let mut stmt = conn
            .prepare("SELECT path, chunk_count FROM file_index WHERE path NOT LIKE 'mail://%'")
            .map_err(|e| format!("projects query failed: {}", e))?;

        let rows: Vec<(String, u32)> = stmt
            .query_map([], |row| {
                Ok((row.get::<_, String>(0)?, row.get::<_, u32>(1)?))
            })
            .map_err(|e| format!("projects scan failed: {}", e))?
            .filter_map(|r| r.ok())
            .collect();

        // Group by detected project
        let mut project_map: std::collections::HashMap<String, (Option<String>, u32, u32)> =
            std::collections::HashMap::new();

        for (path, chunk_count) in &rows {
            if let Some(project_name) = crate::activity::detect_project_from_path(path) {
                let entry = project_map
                    .entry(project_name)
                    .or_insert((None, 0, 0));
                // Try to extract the project root path (walk up to marker)
                if entry.0.is_none() {
                    entry.0 = detect_project_root(path);
                }
                entry.1 += 1; // file count
                entry.2 += chunk_count; // chunk count
            }
        }

        let results = project_map
            .into_iter()
            .map(|(name, (path, files, chunks))| crate::activity::ProjectInfo {
                name,
                path,
                last_active: None,
                indexed_files: files,
                indexed_chunks: chunks,
            })
            .collect();

        Ok(results)
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

/// Enrich a `FileResult` with filesystem metadata (size, modified, preview, project).
///
/// Called by the API handler when `include_preview=true`. Keeps the core
/// search path fast by only doing I/O when explicitly requested.
pub fn enrich_result(result: &mut FileResult) {
    let path = std::path::Path::new(&result.path);

    // file_size_bytes + modified from filesystem metadata
    if let Ok(meta) = std::fs::metadata(path) {
        result.file_size_bytes = Some(meta.len());
        if let Ok(modified) = meta.modified() {
            let dt: chrono::DateTime<chrono::Utc> = modified.into();
            result.modified = Some(dt.to_rfc3339());
        }
    }

    // project detection
    result.project = crate::activity::detect_project_from_path(&result.path);

    // preview_lines + line_count (only for files < 1 MB)
    let size = result.file_size_bytes.unwrap_or(0);
    if size > 0 && size < 1_048_576 {
        if let Ok(content) = std::fs::read_to_string(path) {
            let lines: Vec<&str> = content.lines().collect();
            result.line_count = Some(lines.len() as u32);
            result.preview_lines = Some(
                lines
                    .iter()
                    .take(5)
                    .map(|l| l.to_string())
                    .collect(),
            );
        }
    }
}

/// Walk up from a file path to find the project root directory path.
fn detect_project_root(path_str: &str) -> Option<String> {
    let expanded = if path_str.starts_with('~') {
        dirs::home_dir()
            .map(|h| path_str.replacen('~', &h.to_string_lossy(), 1))
            .unwrap_or_else(|| path_str.to_string())
    } else {
        path_str.to_string()
    };

    let path = std::path::Path::new(&expanded);
    let markers = [
        ".git",
        "Cargo.toml",
        "package.json",
        "pyproject.toml",
        ".xcodeproj",
        "go.mod",
    ];

    let mut current = if path.is_file() {
        path.parent().map(|p| p.to_path_buf())
    } else {
        Some(path.to_path_buf())
    };

    while let Some(dir) = current {
        for marker in &markers {
            if dir.join(marker).exists() {
                return Some(dir.to_string_lossy().to_string());
            }
        }
        current = dir.parent().map(|p| p.to_path_buf());
        if let Some(home) = dirs::home_dir() {
            if current.as_deref() == Some(home.as_path()) {
                break;
            }
        }
    }
    None
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

/// Directories to skip during recursive scanning.
/// Keep in sync with watcher::SKIP_DIRS.
const SKIP_DIRS: &[&str] = &[
    // VCS
    ".git", ".svn", ".hg",
    // Build / dependency
    "node_modules", "target", "build", "dist", "__pycache__",
    ".venv", "venv", ".cache", "Pods", ".next", ".nuxt",
    "vendor", "bower_components",
    // Package managers
    ".npm", ".cargo", ".rustup",
    // System (macOS)
    ".Trash", "Library",
    // Media (not useful for knowledge layer)
    "Movies", "Music", "Pictures", "Photos Library.photoslibrary",
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
