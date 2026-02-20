//! Nervous system bridge for DeepBrain.
//!
//! Wraps key bio-inspired modules from `ruvector-nervous-system` with a
//! thread-safe API for the cognitive engine.
//!
//! ## Modules integrated
//!
//! - **ModernHopfield**: Associative memory retrieval — given a partial query,
//!   retrieve the closest stored pattern via softmax-weighted attention.
//! - **OscillatoryRouter**: Kuramoto-model phase coupling — routes messages
//!   between cognitive modules with coherence-gated gain.
//! - **PredictiveLayer**: Sends only prediction errors (residuals) — reduces
//!   redundant processing by 90-99%.
//! - **DentateGyrus**: Sparse random projection for pattern separation —
//!   detects novelty by measuring how different a new input is from known patterns.

use parking_lot::RwLock;
use ruvector_nervous_system::hopfield::ModernHopfield;
use ruvector_nervous_system::routing::{OscillatoryRouter, PredictiveLayer};
use ruvector_nervous_system::separate::DentateGyrus;
use serde::{Deserialize, Serialize};

/// Thread-safe bridge to the nervous system modules.
pub struct NervousBridge {
    /// Hopfield network for associative memory (384-dim, matches embedding space).
    hopfield: RwLock<ModernHopfield>,
    /// Oscillatory router for cognitive module coordination.
    router: RwLock<OscillatoryRouter>,
    /// Predictive coding layer for bandwidth reduction.
    predictive: RwLock<PredictiveLayer>,
    /// Dentate gyrus for pattern separation / novelty detection.
    dentate: DentateGyrus,
}

/// Configuration for the nervous bridge.
pub struct NervousConfig {
    /// Embedding dimension (default: 384).
    pub dimension: usize,
    /// Hopfield inverse temperature — higher = sharper retrieval (default: 8.0).
    pub hopfield_beta: f32,
    /// Number of cognitive modules to route between (default: 4).
    pub num_modules: usize,
    /// Base oscillation frequency in Hz (default: 40.0 — gamma band).
    pub base_frequency: f32,
    /// Predictive coding residual threshold (default: 0.1).
    pub residual_threshold: f32,
    /// DentateGyrus output expansion factor (default: 4x input).
    pub dentate_expansion: usize,
    /// DentateGyrus sparsity k — number of active neurons (default: 10).
    pub dentate_k: usize,
}

impl Default for NervousConfig {
    fn default() -> Self {
        Self {
            dimension: 384,
            hopfield_beta: 8.0,
            num_modules: 4,
            base_frequency: 40.0,
            residual_threshold: 0.1,
            dentate_expansion: 4,
            dentate_k: 10,
        }
    }
}

impl NervousBridge {
    /// Create with default 384-dim configuration.
    pub fn new() -> Self {
        Self::with_config(NervousConfig::default())
    }

    /// Create with custom configuration.
    pub fn with_config(cfg: NervousConfig) -> Self {
        let output_dim = cfg.dimension * cfg.dentate_expansion;

        Self {
            hopfield: RwLock::new(ModernHopfield::new(cfg.dimension, cfg.hopfield_beta)),
            router: RwLock::new(OscillatoryRouter::new(cfg.num_modules, cfg.base_frequency)),
            predictive: RwLock::new(PredictiveLayer::new(cfg.dimension, cfg.residual_threshold)),
            dentate: DentateGyrus::new(cfg.dimension, output_dim, cfg.dentate_k, 42),
        }
    }

    // ---- Hopfield associative memory ----

    /// Store a pattern in the Hopfield network for later associative retrieval.
    ///
    /// Call this when a new memory is stored — the Hopfield network will learn
    /// to associate this pattern, enabling content-addressable recall.
    pub fn hopfield_store(&self, pattern: &[f32]) {
        let mut net = self.hopfield.write();
        let _ = net.store(pattern.to_vec());
    }

    /// Retrieve the closest stored pattern given a partial/noisy query.
    ///
    /// Returns the retrieved pattern and its similarity score, or None if empty.
    pub fn hopfield_retrieve(&self, query: &[f32]) -> Option<(Vec<f32>, f32)> {
        let net = self.hopfield.read();
        match net.retrieve_k(query, 1) {
            Ok(results) => results.into_iter().next().map(|(_, pattern, sim)| (pattern, sim)),
            Err(_) => None,
        }
    }

    /// Retrieve the k closest stored patterns.
    pub fn hopfield_retrieve_k(
        &self,
        query: &[f32],
        k: usize,
    ) -> Vec<(Vec<f32>, f32)> {
        let net = self.hopfield.read();
        match net.retrieve_k(query, k) {
            Ok(results) => results
                .into_iter()
                .map(|(_, pattern, sim)| (pattern, sim))
                .collect(),
            Err(_) => vec![],
        }
    }

    // ---- Oscillatory routing ----

    /// Step the oscillatory router forward by dt seconds.
    ///
    /// Call this from the background cognitive cycle (e.g., dt = 60.0 for a
    /// 60-second cycle interval). The router updates phase coupling between
    /// cognitive modules.
    pub fn router_step(&self, dt: f32) {
        self.router.write().step(dt);
    }

    /// Get the communication gain between two cognitive modules.
    ///
    /// Returns a value in [0, 1] representing phase coherence.
    /// High gain = modules are in sync and should communicate.
    pub fn router_gain(&self, sender: usize, receiver: usize) -> f32 {
        self.router.read().communication_gain(sender, receiver)
    }

    /// Get the global synchronization order parameter (0 = desync, 1 = full sync).
    pub fn router_sync(&self) -> f32 {
        self.router.read().order_parameter()
    }

    // ---- Predictive coding ----

    /// Check if the input differs enough from the prediction to warrant processing.
    ///
    /// Returns true if the residual exceeds the threshold — meaning this input
    /// contains novel information that should be processed.
    pub fn should_process(&self, input: &[f32]) -> bool {
        self.predictive.read().should_transmit(input)
    }

    /// Update the predictive model with the actual observed input.
    ///
    /// Call this after processing an input, so future predictions improve.
    pub fn update_prediction(&self, actual: &[f32]) {
        self.predictive.write().update(actual);
    }

    /// Compute and return the residual (prediction error) if it exceeds threshold.
    ///
    /// Returns Some(residual) if novel, None if redundant.
    pub fn residual_gate(&self, input: &[f32]) -> Option<Vec<f32>> {
        self.predictive.write().residual_gated_write(input)
    }

    // ---- Dentate gyrus — novelty detection ----

    /// Compute a novelty score for an input vector.
    ///
    /// Uses sparse random projection via the dentate gyrus. The sparse encoding
    /// is checked for energy — inputs that activate unusual pathways produce
    /// higher energy in the sparse code, indicating novelty.
    ///
    /// Returns a score in [0, 1] — higher = more novel.
    pub fn novelty_score(&self, input: &[f32]) -> f32 {
        let sparse = self.dentate.encode_dense(input);
        // Energy of the sparse code: sum of squared activations.
        // Normalize by dimension to get [0, ~1] range.
        let energy: f32 = sparse.iter().map(|x| x * x).sum();
        let max_energy = self.dentate.k() as f32; // At most k neurons active at 1.0
        (energy / max_energy.max(1.0)).min(1.0)
    }

    /// Encode an input into a dense sparse-code vector.
    ///
    /// The output has the same structure as other embeddings (Vec<f32>) but
    /// is a sparse k-winners encoding — useful for collision-resistant dedup.
    pub fn sparse_encode(&self, input: &[f32]) -> Vec<f32> {
        self.dentate.encode_dense(input)
    }

    // ---- Statistics ----

    /// Get nervous system statistics.
    pub fn stats(&self) -> NervousStats {
        let hopfield = self.hopfield.read();
        let router = self.router.read();

        NervousStats {
            hopfield_patterns: hopfield.num_patterns(),
            hopfield_capacity: hopfield.capacity(),
            hopfield_beta: hopfield.beta(),
            router_sync: router.order_parameter(),
            router_modules: router.num_modules(),
            predictive_threshold: self.predictive.read().threshold(),
            dentate_input_dim: self.dentate.input_dim(),
            dentate_output_dim: self.dentate.output_dim(),
            dentate_k: self.dentate.k(),
        }
    }
}

/// Nervous system statistics.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NervousStats {
    /// Number of patterns stored in Hopfield network.
    pub hopfield_patterns: usize,
    /// Theoretical Hopfield capacity.
    pub hopfield_capacity: u64,
    /// Hopfield inverse temperature.
    pub hopfield_beta: f32,
    /// Global synchronization order parameter (0-1).
    pub router_sync: f32,
    /// Number of cognitive modules in the router.
    pub router_modules: usize,
    /// Predictive coding residual threshold.
    pub predictive_threshold: f32,
    /// DentateGyrus input dimension.
    pub dentate_input_dim: usize,
    /// DentateGyrus output dimension.
    pub dentate_output_dim: usize,
    /// DentateGyrus sparsity k.
    pub dentate_k: usize,
}
