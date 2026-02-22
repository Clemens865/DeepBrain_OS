//! GNN attention-based re-ranking bridge — wraps `ruvector_gnn::RuvectorLayer`.

use parking_lot::RwLock;
use ruvector_gnn::RuvectorLayer;
use serde::{Deserialize, Serialize};

/// Information about a neighbor for GNN message passing.
#[derive(Debug, Clone)]
pub struct NeighborInfo {
    pub id: String,
    pub embedding: Vec<f32>,
    pub edge_weight: f32,
}

/// A candidate for re-ranking.
#[derive(Debug, Clone)]
pub struct RerankCandidate {
    pub id: String,
    pub content: String,
    pub embedding: Vec<f32>,
    pub vector_score: f64,
}

/// Result of GNN re-ranking.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RankedResult {
    pub id: String,
    pub content: String,
    pub vector_score: f64,
    pub gnn_score: f64,
    pub blended_score: f64,
}

/// Bridge between DeepBrain and the ruvector-gnn attention layer.
pub struct GnnBridge {
    layer: RwLock<RuvectorLayer>,
}

impl GnnBridge {
    /// Create a new GNN bridge with a fresh RuvectorLayer.
    ///
    /// Dimensions: 384 (matching nomic-embed-text / all-MiniLM-L6-v2).
    /// 4 attention heads, 0.1 dropout.
    pub fn new() -> Self {
        Self {
            layer: RwLock::new(RuvectorLayer::new(384, 384, 4, 0.1)),
        }
    }

    /// Restore a GNN bridge from persisted JSON weights.
    pub fn from_persisted(json: &str) -> Result<Self, String> {
        let layer: RuvectorLayer =
            serde_json::from_str(json).map_err(|e| format!("GNN deserialization error: {}", e))?;
        Ok(Self {
            layer: RwLock::new(layer),
        })
    }

    /// Serialize the current layer weights to JSON.
    pub fn to_json(&self) -> Result<String, String> {
        let layer = self.layer.read();
        serde_json::to_string(&*layer).map_err(|e| format!("GNN serialization error: {}", e))
    }

    /// Re-rank candidates using GNN attention-based message passing.
    ///
    /// For each candidate, the GNN layer performs a forward pass using the query
    /// embedding and the candidate's graph neighbors. The final score blends
    /// `0.7 * vector_score + 0.3 * gnn_score` (matching V2).
    ///
    /// `get_neighbors` is a callback that returns neighbor info for a given node ID.
    pub fn rerank(
        &self,
        query: &[f32],
        candidates: &[RerankCandidate],
        get_neighbors: &dyn Fn(&str) -> Vec<NeighborInfo>,
    ) -> Vec<RankedResult> {
        let layer = self.layer.read();

        let mut results: Vec<RankedResult> = candidates
            .iter()
            .map(|candidate| {
                let neighbors = get_neighbors(&candidate.id);

                let gnn_score = if neighbors.is_empty() {
                    // No neighbors: GNN score is just cosine similarity with query
                    cosine_sim(query, &candidate.embedding) as f64
                } else {
                    let neighbor_embeddings: Vec<Vec<f32>> =
                        neighbors.iter().map(|n| n.embedding.clone()).collect();
                    let edge_weights: Vec<f32> =
                        neighbors.iter().map(|n| n.edge_weight).collect();

                    // GNN forward pass: attention over neighbors
                    let output = layer.forward(
                        &candidate.embedding,
                        &neighbor_embeddings,
                        &edge_weights,
                    );

                    // Score: cosine similarity between GNN-enhanced embedding and query
                    cosine_sim(query, &output) as f64
                };

                let blended = 0.7 * candidate.vector_score + 0.3 * gnn_score;

                RankedResult {
                    id: candidate.id.clone(),
                    content: candidate.content.clone(),
                    vector_score: candidate.vector_score,
                    gnn_score,
                    blended_score: blended,
                }
            })
            .collect();

        // Sort by blended score descending
        results.sort_by(|a, b| {
            b.blended_score
                .partial_cmp(&a.blended_score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        results
    }
}

/// Cosine similarity between two vectors.
fn cosine_sim(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }
    let mut dot = 0.0f32;
    let mut norm_a = 0.0f32;
    let mut norm_b = 0.0f32;
    for (x, y) in a.iter().zip(b.iter()) {
        dot += x * y;
        norm_a += x * x;
        norm_b += y * y;
    }
    let denom = norm_a.sqrt() * norm_b.sqrt();
    if denom < 1e-10 {
        0.0
    } else {
        dot / denom
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rerank_empty_neighbors() {
        let bridge = GnnBridge::new();

        let query: Vec<f32> = (0..384).map(|i| (i as f32 / 384.0).sin()).collect();
        let embedding: Vec<f32> = (0..384).map(|i| (i as f32 / 384.0).sin()).collect();

        let candidates = vec![
            RerankCandidate {
                id: "a".to_string(),
                content: "Test A".to_string(),
                embedding: embedding.clone(),
                vector_score: 0.9,
            },
            RerankCandidate {
                id: "b".to_string(),
                content: "Test B".to_string(),
                embedding: (0..384).map(|i| (i as f32 / 384.0).cos()).collect(),
                vector_score: 0.7,
            },
        ];

        let results = bridge.rerank(&query, &candidates, &|_| vec![]);

        assert_eq!(results.len(), 2);
        // First result should have higher blended score
        assert!(results[0].blended_score >= results[1].blended_score);
    }

    #[test]
    fn test_serialization_roundtrip() {
        let bridge = GnnBridge::new();
        let json = bridge.to_json().unwrap();
        assert!(!json.is_empty());

        let restored = GnnBridge::from_persisted(&json).unwrap();
        let json2 = restored.to_json().unwrap();
        assert_eq!(json, json2);
    }
}
