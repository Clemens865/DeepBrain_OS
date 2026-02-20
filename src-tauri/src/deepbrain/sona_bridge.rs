//! SONA engine bridge for DeepBrain.
//!
//! Wraps SONA's `SonaEngine` with a thread-safe API matching DeepBrain's
//! usage patterns. Replaces both the Q-learning `NativeLearner` and the
//! GNN `SearchReranker` with SONA's 3-tier learning system.
//!
//! ## Architecture
//!
//! - **Instant loop** (<1ms): MicroLoRA adapts per-inference via trajectory recording
//! - **Background loop** (hourly): K-means++ pattern extraction from accumulated trajectories
//! - **EWC++**: Prevents catastrophic forgetting across task boundaries
//!
//! ## Key differences from Q-learning/GNN
//!
//! - Continuous embedding space (no discrete state hashing)
//! - Automatic pattern extraction (no explicit neural network training)
//! - Forgetting prevention built-in (EWC++ with Fisher information)

use parking_lot::RwLock;
use ruvector_sona::engine::SonaEngine;
use ruvector_sona::types::SonaConfig;
use serde::{Deserialize, Serialize};

/// Thread-safe bridge to SONA's adaptive learning engine.
///
/// Replaces both `NativeLearner` (Q-learning) and `SearchReranker` (GNN).
pub struct SonaBridge {
    engine: RwLock<SonaEngine>,
}

impl SonaBridge {
    /// Create a new SONA bridge with default configuration for 384-dim embeddings.
    pub fn new() -> Self {
        let config = SonaConfig {
            hidden_dim: 384,
            embedding_dim: 384,
            micro_lora_rank: 2,
            base_lora_rank: 8,
            background_interval_ms: 3_600_000, // 1 hour
            quality_threshold: 0.3,
            trajectory_capacity: 10_000,
            pattern_clusters: 100,
            ..Default::default()
        };

        Self {
            engine: RwLock::new(SonaEngine::with_config(config)),
        }
    }

    /// Create with custom hidden dimension.
    pub fn with_dim(hidden_dim: usize) -> Self {
        let config = SonaConfig {
            hidden_dim,
            embedding_dim: hidden_dim,
            micro_lora_rank: 2,
            base_lora_rank: 8,
            background_interval_ms: 3_600_000,
            quality_threshold: 0.3,
            trajectory_capacity: 10_000,
            pattern_clusters: 100,
            ..Default::default()
        };

        Self {
            engine: RwLock::new(SonaEngine::with_config(config)),
        }
    }

    // ---- Trajectory recording (replaces NativeLearner::learn) ----

    /// Record a query trajectory with a quality signal.
    ///
    /// This replaces `NativeLearner::learn(experience)`. Instead of discrete
    /// state/action/reward, SONA learns from continuous embedding trajectories.
    ///
    /// # Arguments
    /// * `query_embedding` - The query vector (f32, 384-dim)
    /// * `hidden_states` - Intermediate hidden states from the inference pipeline
    /// * `quality` - Final quality score (0.0-1.0), e.g., user satisfaction or confidence
    pub fn record_trajectory(
        &self,
        query_embedding: &[f32],
        hidden_states: &[f32],
        quality: f32,
    ) {
        let engine = self.engine.read();
        let mut builder = engine.begin_trajectory(query_embedding.to_vec());
        builder.add_step(hidden_states.to_vec(), vec![], quality);
        engine.end_trajectory(builder, quality);
    }

    /// Record a simple query with quality (no intermediate hidden states).
    ///
    /// Convenience method for when you only have the query embedding and
    /// a quality signal (e.g., after think_impl completes).
    pub fn record_query(&self, query_embedding: &[f32], quality: f32) {
        self.record_trajectory(query_embedding, query_embedding, quality);
    }

    // ---- Pattern search (replaces NativeLearner::select_action + SearchReranker::rerank) ----

    /// Find the k most relevant learned patterns for a query.
    ///
    /// This replaces both `NativeLearner::select_action()` (for routing decisions)
    /// and `SearchReranker::rerank()` (for result scoring). Patterns encode learned
    /// centroids with quality scores, replacing both Q-values and GNN weights.
    pub fn find_patterns(&self, query_embedding: &[f32], k: usize) -> Vec<PatternMatch> {
        let engine = self.engine.read();
        let patterns = engine.find_patterns(query_embedding, k);

        patterns
            .into_iter()
            .map(|p| PatternMatch {
                similarity: p.similarity(query_embedding),
                avg_quality: p.avg_quality,
                cluster_size: p.cluster_size,
            })
            .collect()
    }

    /// Compute a reranking boost score for a query based on learned patterns.
    ///
    /// Returns a score in [0.0, 1.0] that can be blended with HNSW scores.
    /// This replaces `SearchReranker::rerank()`.
    pub fn rerank_score(&self, query_embedding: &[f32]) -> f64 {
        let patterns = self.find_patterns(query_embedding, 5);
        if patterns.is_empty() {
            return 0.5; // Neutral when no patterns learned yet
        }

        // Weighted average quality, weighted by similarity
        let total_weight: f32 = patterns.iter().map(|p| p.similarity.max(0.0)).sum();
        if total_weight <= 0.0 {
            return 0.5;
        }

        let weighted_quality: f32 = patterns
            .iter()
            .map(|p| p.similarity.max(0.0) * p.avg_quality)
            .sum();

        (weighted_quality / total_weight) as f64
    }

    // ---- Background learning (replaces GNN training loop + cognitive cycle) ----

    /// Tick the SONA engine — runs background learning cycle if interval elapsed.
    ///
    /// Call this from the background cognitive cycle (e.g., every 60s).
    /// SONA internally tracks the interval (default: 1 hour) and only
    /// runs pattern extraction when enough time has passed.
    ///
    /// Returns a description of what happened (if anything).
    pub fn tick(&self) -> Option<String> {
        self.engine.read().tick()
    }

    /// Force a background learning cycle immediately.
    pub fn force_learn(&self) -> String {
        self.engine.read().force_learn()
    }

    /// Flush instant loop updates (apply accumulated MicroLoRA gradients).
    pub fn flush(&self) {
        self.engine.read().flush();
    }

    // ---- Statistics ----

    /// Get SONA engine statistics.
    pub fn stats(&self) -> SonaStats {
        let stats = self.engine.read().stats();
        SonaStats {
            trajectories_buffered: stats.trajectories_buffered,
            trajectories_dropped: stats.trajectories_dropped,
            patterns_stored: stats.patterns_stored,
            ewc_tasks: stats.ewc_tasks,
            instant_enabled: stats.instant_enabled,
            background_enabled: stats.background_enabled,
        }
    }

    // ---- Persistence ----

    /// Serialize all learned patterns for persistence.
    ///
    /// Returns JSON-serializable patterns that can be stored in SQLite.
    /// On startup, replay these as trajectories to rebuild the reasoning bank.
    pub fn export_patterns(&self) -> Vec<SerializedPattern> {
        let engine = self.engine.read();
        let patterns = engine.get_all_patterns();

        patterns
            .into_iter()
            .map(|p| SerializedPattern {
                centroid: p.centroid.clone(),
                avg_quality: p.avg_quality,
                cluster_size: p.cluster_size,
            })
            .collect()
    }

    /// Import patterns by replaying them as trajectories.
    ///
    /// This rebuilds the SONA reasoning bank from persisted patterns.
    /// Call this at startup to restore learned patterns.
    pub fn import_patterns(&self, patterns: &[SerializedPattern]) {
        let engine = self.engine.read();
        for pattern in patterns {
            let mut builder = engine.begin_trajectory(pattern.centroid.clone());
            builder.add_step(pattern.centroid.clone(), vec![], pattern.avg_quality);
            engine.end_trajectory(builder, pattern.avg_quality);
        }

        // Force a background cycle to rebuild the reasoning bank from imported trajectories
        if !patterns.is_empty() {
            engine.force_learn();
        }
    }
}

/// A pattern match result from SONA's reasoning bank.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PatternMatch {
    /// Cosine similarity to the query (0.0-1.0).
    pub similarity: f32,
    /// Average quality of trajectories in this cluster.
    pub avg_quality: f32,
    /// Number of trajectories that formed this cluster.
    pub cluster_size: usize,
}

/// SONA engine statistics.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SonaStats {
    /// Trajectories waiting in the instant loop buffer.
    pub trajectories_buffered: usize,
    /// Trajectories dropped (buffer full).
    pub trajectories_dropped: u64,
    /// Total patterns in the reasoning bank.
    pub patterns_stored: usize,
    /// Number of EWC++ task boundaries detected.
    pub ewc_tasks: usize,
    /// Whether instant loop is enabled.
    pub instant_enabled: bool,
    /// Whether background loop is enabled.
    pub background_enabled: bool,
}

/// Serializable pattern for persistence.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SerializedPattern {
    /// Pattern centroid (embedding vector).
    pub centroid: Vec<f32>,
    /// Average quality score.
    pub avg_quality: f32,
    /// Number of trajectories in the cluster.
    pub cluster_size: usize,
}
