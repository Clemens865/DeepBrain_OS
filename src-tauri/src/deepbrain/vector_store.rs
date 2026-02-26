//! Thread-safe vector store for DeepBrain using ruvector-core VectorDB.
//!
//! Provides a safe concurrent API over VectorDB (redb + HNSW).
//! All public methods take `&self` — VectorDB handles internal locking.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::time::Instant;

use parking_lot::RwLock;
use ruvector_core::{
    DistanceMetric, SearchQuery, VectorDB, VectorEntry,
};
use ruvector_core::types::{DbOptions, HnswConfig};

use super::error::DeepBrainError;
use super::metrics::StorageMetrics;

/// Metadata associated with a vector in DeepBrain.
#[derive(Clone, Debug, Default)]
pub struct VectorMetadata {
    /// Source type: "memory", "file", or "email".
    pub source: String,
    /// Memory type (e.g., "Semantic", "Episodic"). Empty for file/email.
    pub memory_type: String,
    /// Importance score (0.0-1.0). 0.0 for file/email.
    pub importance: f64,
    /// Timestamp in milliseconds since epoch.
    pub timestamp: i64,
    /// File path or mail:// URI. Empty for memories.
    pub path: String,
}

impl VectorMetadata {
    /// Convert to a HashMap for VectorEntry metadata.
    fn to_hashmap(&self) -> Option<HashMap<String, serde_json::Value>> {
        let mut map = HashMap::new();
        map.insert("source".to_string(), serde_json::Value::String(self.source.clone()));
        map.insert("timestamp".to_string(), serde_json::json!(self.timestamp));

        if !self.memory_type.is_empty() {
            map.insert("memory_type".to_string(), serde_json::Value::String(self.memory_type.clone()));
        }
        if self.importance > 0.0 {
            map.insert("importance".to_string(), serde_json::json!(self.importance));
        }
        if !self.path.is_empty() {
            map.insert("path".to_string(), serde_json::Value::String(self.path.clone()));
        }

        Some(map)
    }
}

/// Filter for vector search queries.
#[derive(Clone, Debug)]
pub enum VectorFilter {
    /// Filter by source type ("memory", "file", "email").
    Source(String),
    /// Filter by memory type ("Semantic", "Episodic", etc.).
    MemoryType(String),
    /// Combined source + memory type filter.
    SourceAndType(String, String),
}

impl VectorFilter {
    /// Convert to a HashMap for SearchQuery filter.
    fn to_hashmap(&self) -> HashMap<String, serde_json::Value> {
        let mut map = HashMap::new();
        match self {
            Self::Source(s) => {
                map.insert("source".to_string(), serde_json::Value::String(s.clone()));
            }
            Self::MemoryType(t) => {
                map.insert("memory_type".to_string(), serde_json::Value::String(t.clone()));
            }
            Self::SourceAndType(s, t) => {
                map.insert("source".to_string(), serde_json::Value::String(s.clone()));
                map.insert("memory_type".to_string(), serde_json::Value::String(t.clone()));
            }
        }
        map
    }
}

/// A single search result from DeepBrain.
#[derive(Clone, Debug)]
pub struct VectorResult {
    /// The string ID of the vector.
    pub id: String,
    /// Similarity score (1.0 = identical, 0.0 = orthogonal, -1.0 = opposite).
    pub similarity: f64,
}

/// Minimal store status (replaces RVF's StoreStatus).
pub struct StoreStatus {
    pub total_vectors: u64,
    pub file_size: u64,
}

/// Thread-safe wrapper around ruvector-core VectorDB.
///
/// VectorDB uses redb for crash-safe storage and HNSW for fast search.
/// No separate reader/writer — VectorDB handles concurrency internally.
pub struct DeepBrainVectorStore {
    /// The ruvector-core vector database.
    db: VectorDB,
    /// Vector dimensionality.
    dimension: u16,
    /// Path to the .redb store file.
    data_path: PathBuf,
    /// Query latency tracking for metrics.
    query_latencies_us: RwLock<Vec<u64>>,
}

impl DeepBrainVectorStore {
    /// Default vector dimension (all-MiniLM-L6-v2).
    pub const DEFAULT_DIMENSION: u16 = 384;

    /// Create a new DeepBrain vector store at the given directory.
    ///
    /// Creates `knowledge.redb` in `data_dir`.
    pub fn create(data_dir: &Path) -> Result<Self, DeepBrainError> {
        std::fs::create_dir_all(data_dir)?;

        let redb_path = data_dir.join("knowledge.redb");

        let options = DbOptions {
            dimensions: Self::DEFAULT_DIMENSION as usize,
            distance_metric: DistanceMetric::Cosine,
            storage_path: redb_path.to_string_lossy().to_string(),
            hnsw_config: Some(HnswConfig {
                m: 32,
                ef_construction: 200,
                ef_search: 100,
                max_elements: 10_000_000,
            }),
            quantization: None,
        };

        let db = VectorDB::new(options)?;

        Ok(Self {
            db,
            dimension: Self::DEFAULT_DIMENSION,
            data_path: redb_path,
            query_latencies_us: RwLock::new(Vec::new()),
        })
    }

    /// Open an existing DeepBrain vector store.
    pub fn open(data_dir: &Path) -> Result<Self, DeepBrainError> {
        let redb_path = data_dir.join("knowledge.redb");

        if !redb_path.exists() {
            return Err(DeepBrainError::StoreNotFound(
                redb_path.display().to_string(),
            ));
        }

        let options = DbOptions {
            dimensions: Self::DEFAULT_DIMENSION as usize,
            distance_metric: DistanceMetric::Cosine,
            storage_path: redb_path.to_string_lossy().to_string(),
            hnsw_config: Some(HnswConfig {
                m: 32,
                ef_construction: 200,
                ef_search: 100,
                max_elements: 10_000_000,
            }),
            quantization: None,
        };

        let db = VectorDB::new(options)?;
        let dimension = Self::DEFAULT_DIMENSION;

        Ok(Self {
            db,
            dimension,
            data_path: redb_path,
            query_latencies_us: RwLock::new(Vec::new()),
        })
    }

    /// Open an existing store or create a new one if it doesn't exist.
    ///
    /// If the existing store is corrupt, it is deleted and a fresh store is created.
    pub fn open_or_create(data_dir: &Path) -> Result<Self, DeepBrainError> {
        let redb_path = data_dir.join("knowledge.redb");
        if redb_path.exists() {
            match Self::open(data_dir) {
                Ok(store) => Ok(store),
                Err(e) => {
                    if let Ok(meta) = std::fs::metadata(&redb_path) {
                        let size_mb = meta.len() / (1024 * 1024);
                        tracing::warn!(
                            "Deleting corrupt vector store ({} MB): {}",
                            size_mb, e
                        );
                    } else {
                        tracing::warn!(
                            "Failed to open existing vector store, creating fresh: {}",
                            e
                        );
                    }
                    let _ = std::fs::remove_file(&redb_path);
                    Self::create(data_dir)
                }
            }
        } else {
            Self::create(data_dir)
        }
    }

    /// Store a single vector with metadata.
    pub fn store_vector(
        &self,
        id: &str,
        vector: &[f32],
        metadata: VectorMetadata,
    ) -> Result<(), DeepBrainError> {
        if vector.len() != self.dimension as usize {
            return Err(DeepBrainError::DimensionMismatch {
                expected: self.dimension,
                got: vector.len(),
            });
        }

        let entry = VectorEntry {
            id: Some(id.to_string()),
            vector: vector.to_vec(),
            metadata: metadata.to_hashmap(),
        };

        self.db.insert(entry)?;
        Ok(())
    }

    /// Store a batch of vectors with metadata.
    ///
    /// Returns the number of vectors accepted.
    pub fn store_batch(
        &self,
        entries: &[(String, Vec<f32>, VectorMetadata)],
    ) -> Result<u64, DeepBrainError> {
        if entries.is_empty() {
            return Ok(0);
        }

        let mut vector_entries = Vec::with_capacity(entries.len());

        for (str_id, vector, metadata) in entries {
            if vector.len() != self.dimension as usize {
                return Err(DeepBrainError::DimensionMismatch {
                    expected: self.dimension,
                    got: vector.len(),
                });
            }

            vector_entries.push(VectorEntry {
                id: Some(str_id.clone()),
                vector: vector.clone(),
                metadata: metadata.to_hashmap(),
            });
        }

        let ids = self.db.insert_batch(vector_entries)?;
        Ok(ids.len() as u64)
    }

    /// Search for the k nearest vectors to the query.
    pub fn search(
        &self,
        query: &[f32],
        k: usize,
        filter: Option<VectorFilter>,
    ) -> Result<Vec<VectorResult>, DeepBrainError> {
        if query.len() != self.dimension as usize {
            return Err(DeepBrainError::DimensionMismatch {
                expected: self.dimension,
                got: query.len(),
            });
        }

        let start = Instant::now();

        let search_query = SearchQuery {
            vector: query.to_vec(),
            k,
            filter: filter.map(|f| f.to_hashmap()),
            ef_search: Some(100),
        };

        let results = self.db.search(search_query)?;

        let elapsed_us = start.elapsed().as_micros() as u64;
        {
            let mut latencies = self.query_latencies_us.write();
            latencies.push(elapsed_us);
            if latencies.len() > 1000 {
                let drain_count = latencies.len() - 1000;
                latencies.drain(..drain_count);
            }
        }

        let mut vector_results = Vec::with_capacity(results.len());
        for sr in results {
            vector_results.push(VectorResult {
                id: sr.id,
                similarity: distance_to_similarity(sr.score),
            });
        }

        Ok(vector_results)
    }

    /// Delete a vector by its string ID.
    pub fn delete(&self, id: &str) -> Result<bool, DeepBrainError> {
        Ok(self.db.delete(id)?)
    }

    /// Delete vectors matching a source filter.
    ///
    /// Iterates all keys, checks metadata, and deletes matches.
    /// O(n) but acceptable for <100k vectors.
    pub fn delete_by_source(&self, source: &str) -> Result<u64, DeepBrainError> {
        let keys = self.db.keys()?;
        let mut deleted = 0u64;

        for key in &keys {
            if let Ok(Some(entry)) = self.db.get(key) {
                if let Some(ref meta) = entry.metadata {
                    if let Some(serde_json::Value::String(s)) = meta.get("source") {
                        if s == source {
                            if self.db.delete(key)? {
                                deleted += 1;
                            }
                        }
                    }
                }
            }
        }

        Ok(deleted)
    }

    /// Run compaction — no-op for redb (handles compaction internally).
    pub fn compact(&self) -> Result<u64, DeepBrainError> {
        Ok(0)
    }

    /// Get current store status.
    pub fn status(&self) -> StoreStatus {
        let total_vectors = self.db.len().unwrap_or(0) as u64;
        let file_size = std::fs::metadata(&self.data_path)
            .map(|m| m.len())
            .unwrap_or(0);

        StoreStatus {
            total_vectors,
            file_size,
        }
    }

    /// Get storage metrics for the health dashboard.
    pub fn metrics(&self) -> StorageMetrics {
        let status = self.status();
        let latencies = self.query_latencies_us.read();

        let (p50, p99) = if latencies.is_empty() {
            (0, 0)
        } else {
            let mut sorted = latencies.clone();
            sorted.sort_unstable();
            let p50_idx = sorted.len() / 2;
            let p99_idx = (sorted.len() as f64 * 0.99) as usize;
            (
                sorted[p50_idx],
                sorted[p99_idx.min(sorted.len() - 1)],
            )
        };

        StorageMetrics {
            total_vectors: status.total_vectors,
            file_size_bytes: status.file_size,
            query_latency_p50_us: p50,
            query_latency_p99_us: p99,
        }
    }

    /// Get the vector dimension.
    pub fn dimension(&self) -> u16 {
        self.dimension
    }
}

/// Convert ruvector-core cosine distance to similarity score.
///
/// ruvector-core cosine distance: 0.0 = identical, 2.0 = opposite.
/// Similarity: 1.0 = identical, -1.0 = opposite.
#[inline]
pub fn distance_to_similarity(distance: f32) -> f64 {
    1.0 - distance as f64
}
