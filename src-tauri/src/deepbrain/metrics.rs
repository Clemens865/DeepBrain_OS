//! Storage health and observability metrics.

use serde::Serialize;

/// Snapshot of DeepBrain storage health metrics.
#[derive(Clone, Debug, Serialize)]
pub struct StorageMetrics {
    /// Total number of live (non-deleted) vectors in the RVF store.
    pub total_vectors: u64,
    /// Total .rvtext file size in bytes.
    pub file_size_bytes: u64,
    /// Current manifest epoch (increments on each write).
    pub current_epoch: u32,
    /// Ratio of dead space to total file size (0.0-1.0).
    pub dead_space_ratio: f64,
    /// Median query latency in microseconds (last 1000 queries).
    pub query_latency_p50_us: u64,
    /// 99th percentile query latency in microseconds.
    pub query_latency_p99_us: u64,
    /// Total entries in the string↔u64 id_map.
    pub id_map_count: u64,
}
