//! Storage health and observability metrics.

use serde::Serialize;

/// Snapshot of DeepBrain storage health metrics.
#[derive(Clone, Debug, Serialize)]
pub struct StorageMetrics {
    /// Total number of live vectors in the store.
    pub total_vectors: u64,
    /// Total store file size in bytes.
    pub file_size_bytes: u64,
    /// Median query latency in microseconds (last 1000 queries).
    pub query_latency_p50_us: u64,
    /// 99th percentile query latency in microseconds.
    pub query_latency_p99_us: u64,
}
