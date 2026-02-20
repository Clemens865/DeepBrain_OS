//! Thread-safe wrapper around RvfStore for DeepBrain.
//!
//! Provides a safe concurrent API over the single-writer/multi-reader RvfStore.
//! All public methods take `&self` with internal locking.

use std::path::{Path, PathBuf};
use std::sync::Mutex;
use std::time::Instant;

use parking_lot::RwLock;
use rvf_runtime::options::{
    DistanceMetric, MetadataEntry, MetadataValue, QueryOptions, RvfOptions, WitnessConfig,
};
use rvf_runtime::filter::{FilterExpr, FilterValue};
use rvf_runtime::store::RvfStore;
use rvf_runtime::StoreStatus;
use rvf_types::DomainProfile;
use rvf_types::quality::QualityPreference;

use super::error::DeepBrainError;
use super::id_map::IdMap;
use super::metrics::StorageMetrics;

// Metadata field IDs — stable constants for the DeepBrain schema.
pub const FIELD_SOURCE: u16 = 0; // "memory", "file", "email"
pub const FIELD_MEMORY_TYPE: u16 = 1; // "Semantic", "Episodic", etc.
pub const FIELD_IMPORTANCE: u16 = 2; // f64 importance score
pub const FIELD_TIMESTAMP: u16 = 3; // i64 unix timestamp ms
pub const FIELD_PATH: u16 = 4; // file path or "mail://<id>"

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
    fn to_rvf_metadata(&self) -> Vec<MetadataEntry> {
        let mut entries = vec![
            MetadataEntry {
                field_id: FIELD_SOURCE,
                value: MetadataValue::String(self.source.clone()),
            },
            MetadataEntry {
                field_id: FIELD_TIMESTAMP,
                value: MetadataValue::I64(self.timestamp),
            },
        ];

        if !self.memory_type.is_empty() {
            entries.push(MetadataEntry {
                field_id: FIELD_MEMORY_TYPE,
                value: MetadataValue::String(self.memory_type.clone()),
            });
        }

        if self.importance > 0.0 {
            entries.push(MetadataEntry {
                field_id: FIELD_IMPORTANCE,
                value: MetadataValue::F64(self.importance),
            });
        }

        if !self.path.is_empty() {
            entries.push(MetadataEntry {
                field_id: FIELD_PATH,
                value: MetadataValue::String(self.path.clone()),
            });
        }

        entries
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
    fn to_filter_expr(&self) -> FilterExpr {
        match self {
            Self::Source(s) => FilterExpr::Eq(FIELD_SOURCE, FilterValue::String(s.clone())),
            Self::MemoryType(t) => {
                FilterExpr::Eq(FIELD_MEMORY_TYPE, FilterValue::String(t.clone()))
            }
            Self::SourceAndType(s, t) => FilterExpr::And(vec![
                FilterExpr::Eq(FIELD_SOURCE, FilterValue::String(s.clone())),
                FilterExpr::Eq(FIELD_MEMORY_TYPE, FilterValue::String(t.clone())),
            ]),
        }
    }
}

/// A single search result from DeepBrain.
#[derive(Clone, Debug)]
pub struct VectorResult {
    /// The string ID of the vector.
    pub id: String,
    /// Similarity score (1.0 = identical, 0.0 = orthogonal, -1.0 = opposite).
    pub similarity: f64,
    /// The numeric ID from RvfStore.
    pub num_id: u64,
}

/// Thread-safe wrapper around RvfStore.
///
/// Uses Mutex for write operations and RwLock for concurrent reads.
/// The writer holds an exclusive advisory lock on the .rvtext file.
pub struct DeepBrainVectorStore {
    /// Writer handle — exclusive access for ingest/delete/compact.
    writer: Mutex<RvfStore>,
    /// Reader handle — concurrent access for queries.
    reader: RwLock<RvfStore>,
    /// Bidirectional String ↔ u64 ID mapping.
    id_map: IdMap,
    /// Vector dimensionality.
    dimension: u16,
    /// Path to the .rvtext store file.
    data_path: PathBuf,
    /// Query latency tracking for metrics.
    query_latencies_us: RwLock<Vec<u64>>,
}

impl DeepBrainVectorStore {
    /// Default vector dimension (all-MiniLM-L6-v2).
    pub const DEFAULT_DIMENSION: u16 = 384;

    /// Create a new DeepBrain vector store at the given directory.
    ///
    /// Creates `knowledge.rvtext` and `deepbrain_ids.db` in `data_dir`.
    pub fn create(data_dir: &Path) -> Result<Self, DeepBrainError> {
        std::fs::create_dir_all(data_dir)?;

        let rvtext_path = data_dir.join("knowledge.rvtext");
        let idmap_path = data_dir.join("deepbrain_ids.db");

        let options = RvfOptions {
            dimension: Self::DEFAULT_DIMENSION,
            metric: DistanceMetric::Cosine,
            profile: 0,
            domain_profile: DomainProfile::RvText,
            m: 16,
            ef_construction: 200,
            witness: WitnessConfig {
                witness_ingest: true,
                witness_delete: true,
                witness_compact: true,
                audit_queries: false,
            },
            ..Default::default()
        };

        let writer = RvfStore::create(&rvtext_path, options)?;
        let reader = RvfStore::open_readonly(&rvtext_path)?;
        let id_map = IdMap::open(&idmap_path)?;

        Ok(Self {
            writer: Mutex::new(writer),
            reader: RwLock::new(reader),
            id_map,
            dimension: Self::DEFAULT_DIMENSION,
            data_path: rvtext_path,
            query_latencies_us: RwLock::new(Vec::new()),
        })
    }

    /// Open an existing DeepBrain vector store.
    pub fn open(data_dir: &Path) -> Result<Self, DeepBrainError> {
        let rvtext_path = data_dir.join("knowledge.rvtext");
        let idmap_path = data_dir.join("deepbrain_ids.db");

        if !rvtext_path.exists() {
            return Err(DeepBrainError::StoreNotFound(
                rvtext_path.display().to_string(),
            ));
        }

        let writer = RvfStore::open(&rvtext_path)?;
        let dimension = writer.dimension();
        let reader = RvfStore::open_readonly(&rvtext_path)?;
        let id_map = IdMap::open(&idmap_path)?;

        Ok(Self {
            writer: Mutex::new(writer),
            reader: RwLock::new(reader),
            id_map,
            dimension,
            data_path: rvtext_path,
            query_latencies_us: RwLock::new(Vec::new()),
        })
    }

    /// Open an existing store or create a new one if it doesn't exist.
    pub fn open_or_create(data_dir: &Path) -> Result<Self, DeepBrainError> {
        let rvtext_path = data_dir.join("knowledge.rvtext");
        if rvtext_path.exists() {
            Self::open(data_dir)
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

        let (num_id, _) = self.id_map.get_or_insert(id)?;
        let meta_entries = metadata.to_rvf_metadata();

        {
            let mut writer = self.writer.lock().map_err(|_| {
                DeepBrainError::Io(std::io::Error::new(
                    std::io::ErrorKind::Other,
                    "Writer mutex poisoned",
                ))
            })?;

            writer.ingest_batch(
                &[vector],
                &[num_id],
                Some(&meta_entries),
            )?;
        }

        // Refresh the reader to see the new data.
        self.refresh_reader()?;

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

        // Allocate all numeric IDs first.
        let mut num_ids = Vec::with_capacity(entries.len());
        let mut all_metadata = Vec::new();

        for (str_id, vector, metadata) in entries {
            if vector.len() != self.dimension as usize {
                return Err(DeepBrainError::DimensionMismatch {
                    expected: self.dimension,
                    got: vector.len(),
                });
            }
            let (num_id, _) = self.id_map.get_or_insert(str_id)?;
            num_ids.push(num_id);
            all_metadata.extend(metadata.to_rvf_metadata());
        }

        let vectors: Vec<&[f32]> = entries.iter().map(|(_, v, _)| v.as_slice()).collect();

        let result = {
            let mut writer = self.writer.lock().map_err(|_| {
                DeepBrainError::Io(std::io::Error::new(
                    std::io::ErrorKind::Other,
                    "Writer mutex poisoned",
                ))
            })?;

            writer.ingest_batch(&vectors, &num_ids, Some(&all_metadata))?
        };

        self.refresh_reader()?;

        Ok(result.accepted)
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

        let options = QueryOptions {
            ef_search: 100,
            filter: filter.map(|f| f.to_filter_expr()),
            timeout_ms: 0,
            quality_preference: QualityPreference::Auto,
            ..Default::default()
        };

        let envelope = {
            let reader = self.reader.read();
            reader.query_with_envelope(query, k, &options)?
        };

        let elapsed_us = start.elapsed().as_micros() as u64;
        {
            let mut latencies = self.query_latencies_us.write();
            latencies.push(elapsed_us);
            // Keep only the last 1000 latencies for metrics.
            if latencies.len() > 1000 {
                let drain_count = latencies.len() - 1000;
                latencies.drain(..drain_count);
            }
        }

        // Convert RVF results to DeepBrain results, mapping u64 → String IDs.
        let mut results = Vec::with_capacity(envelope.results.len());
        for sr in &envelope.results {
            if let Ok(Some(str_id)) = self.id_map.get_str_id(sr.id) {
                results.push(VectorResult {
                    id: str_id,
                    similarity: distance_to_similarity(sr.distance),
                    num_id: sr.id,
                });
            }
        }

        Ok(results)
    }

    /// Delete a vector by its string ID.
    pub fn delete(&self, id: &str) -> Result<bool, DeepBrainError> {
        let num_id = match self.id_map.get_num_id(id)? {
            Some(nid) => nid,
            None => return Ok(false),
        };

        {
            let mut writer = self.writer.lock().map_err(|_| {
                DeepBrainError::Io(std::io::Error::new(
                    std::io::ErrorKind::Other,
                    "Writer mutex poisoned",
                ))
            })?;

            writer.delete(&[num_id])?;
        }

        self.id_map.remove(id)?;
        self.refresh_reader()?;

        Ok(true)
    }

    /// Delete vectors matching a filter.
    pub fn delete_by_source(&self, source: &str) -> Result<u64, DeepBrainError> {
        let filter = FilterExpr::Eq(FIELD_SOURCE, FilterValue::String(source.to_string()));

        let result = {
            let mut writer = self.writer.lock().map_err(|_| {
                DeepBrainError::Io(std::io::Error::new(
                    std::io::ErrorKind::Other,
                    "Writer mutex poisoned",
                ))
            })?;

            writer.delete_by_filter(&filter)?
        };

        if result.deleted > 0 {
            self.refresh_reader()?;
        }

        Ok(result.deleted)
    }

    /// Run compaction to reclaim dead space.
    pub fn compact(&self) -> Result<u64, DeepBrainError> {
        let result = {
            let mut writer = self.writer.lock().map_err(|_| {
                DeepBrainError::Io(std::io::Error::new(
                    std::io::ErrorKind::Other,
                    "Writer mutex poisoned",
                ))
            })?;

            writer.compact()?
        };

        self.refresh_reader()?;
        Ok(result.bytes_reclaimed)
    }

    /// Get current store status.
    pub fn status(&self) -> StoreStatus {
        let reader = self.reader.read();
        reader.status()
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
            current_epoch: status.current_epoch,
            dead_space_ratio: status.dead_space_ratio,
            query_latency_p50_us: p50,
            query_latency_p99_us: p99,
            id_map_count: self.id_map.count().unwrap_or(0),
        }
    }

    /// Get the vector dimension.
    pub fn dimension(&self) -> u16 {
        self.dimension
    }

    /// Refresh the reader handle after a write operation.
    fn refresh_reader(&self) -> Result<(), DeepBrainError> {
        let new_reader = RvfStore::open_readonly(&self.data_path)?;
        let mut reader = self.reader.write();
        *reader = new_reader;
        Ok(())
    }
}

/// Convert RVF cosine distance to similarity score.
///
/// RVF cosine distance: 0.0 = identical, 2.0 = opposite.
/// Similarity: 1.0 = identical, -1.0 = opposite.
#[inline]
pub fn distance_to_similarity(distance: f32) -> f64 {
    1.0 - distance as f64
}
