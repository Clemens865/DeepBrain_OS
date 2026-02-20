//! GNN-based search re-ranker for DeepBrain
//!
//! A lightweight 2-layer neural re-ranker that learns from query engagement logs.
//! Sits after HNSW retrieval to refine result ordering based on graph features.

use ndarray::{Array1, Array2};
use parking_lot::RwLock;
use rand::Rng;
use serde::{Deserialize, Serialize};

/// Query log entry for training the re-ranker
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QueryLogEntry {
    pub query_hash: i64,
    pub result_ids: Vec<String>,
    pub selected_id: Option<String>,
    pub source: String,
    pub timestamp: i64,
}

/// Features for a single search candidate
#[derive(Debug, Clone)]
pub struct CandidateFeatures {
    pub id: String,
    pub vector: Vec<f32>,
    pub hnsw_score: f64,
    pub access_count: u32,
    pub timestamp: i64,
    pub type_match: bool,
}

/// 2-layer neural re-ranker for search results
pub struct SearchReranker {
    /// Layer 1: (input_dim → 128) where input_dim = 2 * vec_dim
    w1: RwLock<Array2<f32>>,
    b1: RwLock<Array1<f32>>,
    /// Layer 2: (132 → 1) — 128 from layer1 + 4 graph features
    w2: RwLock<Array2<f32>>,
    b2: RwLock<Array1<f32>>,
    /// Vector dimensions (384 for MiniLM-L6-v2)
    vec_dim: usize,
    /// Hidden layer size
    hidden_dim: usize,
    /// Learning rate for SGD
    learning_rate: f32,
    /// Whether the model has been trained at least once
    trained: RwLock<bool>,
    /// Number of training steps completed
    train_steps: RwLock<u64>,
}

impl SearchReranker {
    /// Create a new re-ranker with Xavier-initialized weights
    pub fn new(vec_dim: usize) -> Self {
        let hidden_dim = 128;
        let input_dim = vec_dim * 2; // concat(query, candidate)
        let feature_dim = hidden_dim + 4; // hidden output + 4 graph features

        let mut rng = rand::thread_rng();

        // Xavier initialization: scale = sqrt(2 / (fan_in + fan_out))
        let scale1 = (2.0 / (input_dim + hidden_dim) as f32).sqrt();
        let w1 = Array2::from_shape_fn((input_dim, hidden_dim), |_| {
            rng.gen_range(-scale1..scale1)
        });
        let b1 = Array1::zeros(hidden_dim);

        let scale2 = (2.0 / (feature_dim + 1) as f32).sqrt();
        let w2 = Array2::from_shape_fn((feature_dim, 1), |_| {
            rng.gen_range(-scale2..scale2)
        });
        let b2 = Array1::zeros(1);

        Self {
            w1: RwLock::new(w1),
            b1: RwLock::new(b1),
            w2: RwLock::new(w2),
            b2: RwLock::new(b2),
            vec_dim,
            hidden_dim,
            learning_rate: 0.001,
            trained: RwLock::new(false),
            train_steps: RwLock::new(0),
        }
    }

    /// Check if the model has been trained
    pub fn is_trained(&self) -> bool {
        *self.trained.read()
    }

    /// Get the number of training steps
    pub fn train_step_count(&self) -> u64 {
        *self.train_steps.read()
    }

    /// Forward pass for a single candidate. Returns a scalar score.
    fn forward(&self, query: &[f32], candidate: &CandidateFeatures) -> f32 {
        let w1 = self.w1.read();
        let b1 = self.b1.read();
        let w2 = self.w2.read();
        let b2 = self.b2.read();

        // Concatenate query and candidate vectors
        let mut input = Vec::with_capacity(self.vec_dim * 2);
        input.extend_from_slice(query);
        if candidate.vector.len() >= self.vec_dim {
            input.extend_from_slice(&candidate.vector[..self.vec_dim]);
        } else {
            input.extend_from_slice(&candidate.vector);
            input.resize(self.vec_dim * 2, 0.0);
        }
        let input = Array1::from_vec(input);

        // Layer 1: ReLU(W1 @ input + b1)
        let hidden = w1.t().dot(&input) + &*b1;
        let hidden = hidden.mapv(|x| x.max(0.0)); // ReLU

        // Append 4 graph features
        let now_ms = crate::brain::utils::now_millis();
        let access_norm = (1.0 + candidate.access_count as f32).ln() / 10.0;
        let recency = (-(now_ms - candidate.timestamp) as f32 / 86_400_000.0).exp();
        let type_match_f = if candidate.type_match { 1.0 } else { 0.0 };
        let hnsw_score_f = candidate.hnsw_score as f32;

        let mut features = Vec::with_capacity(self.hidden_dim + 4);
        features.extend(hidden.iter());
        features.push(hnsw_score_f);
        features.push(access_norm);
        features.push(recency);
        features.push(type_match_f);
        let features = Array1::from_vec(features);

        // Layer 2: W2 @ features + b2 → scalar
        let output = w2.t().dot(&features) + &*b2;
        output[0]
    }

    /// Re-rank candidates using the trained model.
    /// Returns candidates sorted by blended score (higher = better).
    pub fn rerank(
        &self,
        query: &[f32],
        candidates: &[CandidateFeatures],
    ) -> Vec<(String, f64)> {
        let blend_hnsw = 0.7;
        let blend_gnn = 0.3;

        let mut scored: Vec<(String, f64)> = candidates
            .iter()
            .map(|c| {
                let gnn_score = self.forward(query, c) as f64;
                // Sigmoid to normalize GNN output to [0, 1]
                let gnn_norm = 1.0 / (1.0 + (-gnn_score).exp());
                let final_score = blend_hnsw * c.hnsw_score + blend_gnn * gnn_norm;
                (c.id.clone(), final_score)
            })
            .collect();

        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        scored
    }

    /// Train on a batch of query log entries using margin ranking loss.
    /// Returns the average loss for this batch.
    pub fn train_step(
        &self,
        logs: &[QueryLogEntry],
        get_vector: &dyn Fn(&str) -> Option<Vec<f32>>,
        get_features: &dyn Fn(&str) -> (u32, i64, bool), // (access_count, timestamp, type_match)
    ) -> f32 {
        if logs.is_empty() {
            return 0.0;
        }

        let margin = 0.1f32;
        let mut total_loss = 0.0f32;
        let mut num_pairs = 0u32;

        // Accumulate gradients
        let w1 = self.w1.read().clone();
        let b1 = self.b1.read().clone();
        let w2 = self.w2.read().clone();
        let b2 = self.b2.read().clone();

        let mut dw1 = Array2::<f32>::zeros(w1.raw_dim());
        let mut db1 = Array1::<f32>::zeros(b1.raw_dim());
        let mut dw2 = Array2::<f32>::zeros(w2.raw_dim());
        let mut db2 = Array1::<f32>::zeros(b2.raw_dim());

        for log in logs {
            let selected = match &log.selected_id {
                Some(id) => id,
                None => continue,
            };

            // Get query vector (use hash to find a representative vector)
            // We use the selected result's vector as a proxy for the query
            let query_vec = match get_vector(selected) {
                Some(v) => v,
                None => continue,
            };

            let (sel_access, sel_ts, sel_type) = get_features(selected);
            let pos_features = CandidateFeatures {
                id: selected.clone(),
                vector: query_vec.clone(),
                hnsw_score: 1.0, // positive example
                access_count: sel_access,
                timestamp: sel_ts,
                type_match: sel_type,
            };
            let pos_score = self.forward(&query_vec, &pos_features);

            // Negative examples: other results that weren't selected
            for neg_id in &log.result_ids {
                if neg_id == selected {
                    continue;
                }

                let neg_vec = match get_vector(neg_id) {
                    Some(v) => v,
                    None => continue,
                };

                let (neg_access, neg_ts, neg_type) = get_features(neg_id);
                let neg_features = CandidateFeatures {
                    id: neg_id.clone(),
                    vector: neg_vec,
                    hnsw_score: 0.5,
                    access_count: neg_access,
                    timestamp: neg_ts,
                    type_match: neg_type,
                };
                let neg_score = self.forward(&query_vec, &neg_features);

                // Margin ranking loss: max(0, margin - pos_score + neg_score)
                let loss = (margin - pos_score + neg_score).max(0.0);
                total_loss += loss;
                num_pairs += 1;

                if loss > 0.0 {
                    // Compute approximate gradients using finite differences
                    // (Simpler than full backprop for this small network)
                    let eps = 0.001f32;
                    self.accumulate_gradients_fd(
                        &query_vec,
                        &pos_features,
                        &neg_features,
                        eps,
                        &w1,
                        &b1,
                        &w2,
                        &b2,
                        &mut dw1,
                        &mut db1,
                        &mut dw2,
                        &mut db2,
                    );
                }
            }
        }

        if num_pairs == 0 {
            return 0.0;
        }

        // Normalize gradients and apply SGD update
        let scale = self.learning_rate / num_pairs as f32;
        {
            let mut w1_lock = self.w1.write();
            let mut b1_lock = self.b1.write();
            let mut w2_lock = self.w2.write();
            let mut b2_lock = self.b2.write();

            *w1_lock = &*w1_lock - &(scale * &dw1);
            *b1_lock = &*b1_lock - &(scale * &db1);
            *w2_lock = &*w2_lock - &(scale * &dw2);
            *b2_lock = &*b2_lock - &(scale * &db2);
        }

        *self.trained.write() = true;
        *self.train_steps.write() += 1;

        total_loss / num_pairs as f32
    }

    /// Accumulate finite-difference gradients for the margin ranking loss.
    /// We perturb each weight slightly and measure the effect on (pos_score - neg_score).
    /// This is simpler than full backprop for a 2-layer network.
    #[allow(clippy::too_many_arguments)]
    fn accumulate_gradients_fd(
        &self,
        query: &[f32],
        pos: &CandidateFeatures,
        neg: &CandidateFeatures,
        eps: f32,
        w1: &Array2<f32>,
        b1: &Array1<f32>,
        w2: &Array2<f32>,
        b2: &Array1<f32>,
        dw1: &mut Array2<f32>,
        db1: &mut Array1<f32>,
        dw2: &mut Array2<f32>,
        db2: &mut Array1<f32>,
    ) {
        // For efficiency, only perturb a random subset of weights
        let mut rng = rand::thread_rng();
        let sample_rate = 0.05; // Perturb 5% of weights

        // Gradient for w2 (smaller, perturb all)
        let (rows2, cols2) = w2.dim();
        for i in 0..rows2 {
            for j in 0..cols2 {
                {
                    let mut w2_mut = self.w2.write();
                    w2_mut[[i, j]] += eps;
                }
                let pos_plus = self.forward(query, pos);
                let neg_plus = self.forward(query, neg);
                {
                    let mut w2_mut = self.w2.write();
                    w2_mut[[i, j]] = w2[[i, j]]; // restore
                }
                // gradient of loss = gradient of (neg_score - pos_score)
                let grad = ((neg_plus - pos_plus) - (self.forward(query, neg) - self.forward(query, pos))) / eps;
                dw2[[i, j]] += grad;
            }
        }

        // Gradient for b2
        for i in 0..b2.len() {
            {
                let mut b2_mut = self.b2.write();
                b2_mut[i] += eps;
            }
            let pos_plus = self.forward(query, pos);
            let neg_plus = self.forward(query, neg);
            {
                let mut b2_mut = self.b2.write();
                b2_mut[i] = b2[i];
            }
            let grad = ((neg_plus - pos_plus) - (self.forward(query, neg) - self.forward(query, pos))) / eps;
            db2[i] += grad;
        }

        // Gradient for w1 (large matrix — sample randomly)
        let (rows1, cols1) = w1.dim();
        for i in 0..rows1 {
            for j in 0..cols1 {
                if rng.gen::<f32>() > sample_rate {
                    continue;
                }
                {
                    let mut w1_mut = self.w1.write();
                    w1_mut[[i, j]] += eps;
                }
                let pos_plus = self.forward(query, pos);
                let neg_plus = self.forward(query, neg);
                {
                    let mut w1_mut = self.w1.write();
                    w1_mut[[i, j]] = w1[[i, j]];
                }
                let grad = ((neg_plus - pos_plus) - (self.forward(query, neg) - self.forward(query, pos))) / eps;
                dw1[[i, j]] += grad / sample_rate;
            }
        }

        // Gradient for b1 (small — perturb all)
        for i in 0..b1.len() {
            {
                let mut b1_mut = self.b1.write();
                b1_mut[i] += eps;
            }
            let pos_plus = self.forward(query, pos);
            let neg_plus = self.forward(query, neg);
            {
                let mut b1_mut = self.b1.write();
                b1_mut[i] = b1[i];
            }
            let grad = ((neg_plus - pos_plus) - (self.forward(query, neg) - self.forward(query, pos))) / eps;
            db1[i] += grad;
        }
    }

    /// Serialize the model weights to bytes
    pub fn serialize(&self) -> Result<Vec<u8>, String> {
        let data = SerializedReranker {
            vec_dim: self.vec_dim,
            hidden_dim: self.hidden_dim,
            w1: self.w1.read().as_slice().unwrap_or(&[]).to_vec(),
            b1: self.b1.read().as_slice().unwrap_or(&[]).to_vec(),
            w2: self.w2.read().as_slice().unwrap_or(&[]).to_vec(),
            b2: self.b2.read().as_slice().unwrap_or(&[]).to_vec(),
            trained: *self.trained.read(),
            train_steps: *self.train_steps.read(),
        };
        bincode::serialize(&data).map_err(|e| format!("Serialize error: {}", e))
    }

    /// Deserialize model weights from bytes
    pub fn deserialize(bytes: &[u8]) -> Result<Self, String> {
        let data: SerializedReranker =
            bincode::deserialize(bytes).map_err(|e| format!("Deserialize error: {}", e))?;

        let input_dim = data.vec_dim * 2;
        let feature_dim = data.hidden_dim + 4;

        let w1 = Array2::from_shape_vec((input_dim, data.hidden_dim), data.w1)
            .map_err(|e| format!("W1 shape error: {}", e))?;
        let b1 = Array1::from_vec(data.b1);
        let w2 = Array2::from_shape_vec((feature_dim, 1), data.w2)
            .map_err(|e| format!("W2 shape error: {}", e))?;
        let b2 = Array1::from_vec(data.b2);

        Ok(Self {
            w1: RwLock::new(w1),
            b1: RwLock::new(b1),
            w2: RwLock::new(w2),
            b2: RwLock::new(b2),
            vec_dim: data.vec_dim,
            hidden_dim: data.hidden_dim,
            learning_rate: 0.001,
            trained: RwLock::new(data.trained),
            train_steps: RwLock::new(data.train_steps),
        })
    }
}

#[derive(Serialize, Deserialize)]
struct SerializedReranker {
    vec_dim: usize,
    hidden_dim: usize,
    w1: Vec<f32>,
    b1: Vec<f32>,
    w2: Vec<f32>,
    b2: Vec<f32>,
    trained: bool,
    train_steps: u64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_reranker_creation() {
        let reranker = SearchReranker::new(384);
        assert!(!reranker.is_trained());
        assert_eq!(reranker.train_step_count(), 0);
    }

    #[test]
    fn test_reranker_forward() {
        let reranker = SearchReranker::new(4);
        let query = vec![1.0, 0.0, 0.0, 0.0];
        let features = CandidateFeatures {
            id: "test".to_string(),
            vector: vec![0.9, 0.1, 0.0, 0.0],
            hnsw_score: 0.95,
            access_count: 5,
            timestamp: crate::brain::utils::now_millis(),
            type_match: true,
        };
        // Should produce a finite score
        let score = reranker.forward(&query, &features);
        assert!(score.is_finite());
    }

    #[test]
    fn test_reranker_serialize_roundtrip() {
        let reranker = SearchReranker::new(4);
        let bytes = reranker.serialize().unwrap();
        let restored = SearchReranker::deserialize(&bytes).unwrap();
        assert_eq!(restored.vec_dim, 4);
        assert!(!restored.is_trained());
    }
}
