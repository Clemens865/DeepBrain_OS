//! Tensor compression bridge — wraps `ruvector_gnn::TensorCompress`.

use ruvector_gnn::{CompressedTensor, TensorCompress};
use serde::{Deserialize, Serialize};

/// A memory entry prepared for compression scanning.
pub struct MemoryForCompression {
    pub id: String,
    pub access_count: u32,
    pub vector_len: usize,
}

/// Statistics from a compression scan.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompressionStats {
    pub total_memories: usize,
    pub hot_count: usize,      // freq > 0.8 → full precision
    pub warm_count: usize,     // freq > 0.4 → half precision
    pub cold_count: usize,     // freq <= 0.4 → PQ8/PQ4/binary
    pub estimated_savings_pct: f64,
}

/// Bridge between DeepBrain and ruvector-gnn tensor compression.
pub struct CompressBridge {
    compressor: TensorCompress,
}

impl CompressBridge {
    /// Create a new compression bridge with default settings.
    pub fn new() -> Self {
        Self {
            compressor: TensorCompress::new(),
        }
    }

    /// Compress an embedding vector based on access frequency.
    ///
    /// Returns the compressed tensor as a JSON string.
    pub fn compress(&self, embedding: &[f32], access_freq: f32) -> Result<String, String> {
        let compressed = self
            .compressor
            .compress(embedding, access_freq)
            .map_err(|e| format!("Compression error: {}", e))?;
        serde_json::to_string(&compressed).map_err(|e| format!("Serialization error: {}", e))
    }

    /// Decompress a tensor from its JSON representation.
    pub fn decompress(&self, json: &str) -> Result<Vec<f32>, String> {
        let compressed: CompressedTensor =
            serde_json::from_str(json).map_err(|e| format!("Deserialization error: {}", e))?;
        self.compressor
            .decompress(&compressed)
            .map_err(|e| format!("Decompression error: {}", e))
    }

    /// Scan all memories and compute compression statistics.
    ///
    /// `total_accesses` is the sum of all access counts across all memories,
    /// used to normalize per-memory access frequency.
    pub fn scan(
        &self,
        memories: &[MemoryForCompression],
        total_accesses: u64,
    ) -> CompressionStats {
        let total = memories.len();
        let denominator = if total_accesses > 0 {
            total_accesses as f32
        } else {
            1.0
        };

        let mut hot = 0usize;
        let mut warm = 0usize;
        let mut cold = 0usize;
        let mut savings_bytes: f64 = 0.0;

        for mem in memories {
            let freq = mem.access_count as f32 / denominator;

            if freq > 0.8 {
                hot += 1;
                // Full precision: no savings
            } else if freq > 0.4 {
                warm += 1;
                // Half precision: ~50% savings
                savings_bytes += (mem.vector_len * 2) as f64; // 4 bytes → 2 bytes per element
            } else {
                cold += 1;
                // PQ8 or lower: ~75% savings
                savings_bytes += (mem.vector_len * 3) as f64; // 4 bytes → ~1 byte per element
            }
        }

        let total_original_bytes = memories
            .iter()
            .map(|m| m.vector_len * 4) // f32 = 4 bytes
            .sum::<usize>() as f64;

        let pct = if total_original_bytes > 0.0 {
            (savings_bytes / total_original_bytes) * 100.0
        } else {
            0.0
        };

        CompressionStats {
            total_memories: total,
            hot_count: hot,
            warm_count: warm,
            cold_count: cold,
            estimated_savings_pct: (pct * 10.0).round() / 10.0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_compress_decompress_roundtrip() {
        let bridge = CompressBridge::new();

        let embedding: Vec<f32> = (0..384).map(|i| (i as f32 / 384.0).sin()).collect();
        let json = bridge.compress(&embedding, 0.5).unwrap();
        assert!(!json.is_empty());

        let decompressed = bridge.decompress(&json).unwrap();
        assert_eq!(decompressed.len(), 384);
    }

    #[test]
    fn test_scan_stats() {
        let bridge = CompressBridge::new();

        let memories = vec![
            MemoryForCompression {
                id: "a".to_string(),
                access_count: 90,
                vector_len: 384,
            },
            MemoryForCompression {
                id: "b".to_string(),
                access_count: 50,
                vector_len: 384,
            },
            MemoryForCompression {
                id: "c".to_string(),
                access_count: 5,
                vector_len: 384,
            },
        ];

        let stats = bridge.scan(&memories, 100);
        assert_eq!(stats.total_memories, 3);
        assert_eq!(stats.hot_count, 1);
        assert_eq!(stats.warm_count, 1);
        assert_eq!(stats.cold_count, 1);
        assert!(stats.estimated_savings_pct > 0.0);
    }
}
