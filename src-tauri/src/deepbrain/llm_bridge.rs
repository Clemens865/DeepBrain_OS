//! Local LLM bridge for DeepBrain.
//!
//! Wraps `ruvllm::CandleBackend` to provide local LLM inference on Apple
//! Silicon (Metal GPU) without requiring an external Ollama server.
//!
//! Implements the `AiProvider` trait so it can be used as a drop-in
//! replacement for OllamaProvider or ClaudeProvider.

use parking_lot::RwLock;
use std::path::PathBuf;
use std::sync::Arc;

use ruvllm::{CandleBackend, GenerateParams, LlmBackend, ModelConfig};

use crate::ai::{format_memory_context, AiResponse, AiProvider};
use crate::brain::cognitive::RecallResult;

/// Status of the local LLM backend.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct LlmStatus {
    /// Whether a model is currently loaded.
    pub model_loaded: bool,
    /// Model identifier (HuggingFace ID or local path).
    pub model_id: Option<String>,
    /// Model info string from backend.
    pub model_info: Option<String>,
}

/// Configuration for the local LLM bridge.
#[derive(Debug, Clone)]
pub struct LlmBridgeConfig {
    /// HuggingFace model ID or local path to GGUF file.
    pub model_id: String,
    /// Maximum tokens to generate per response.
    pub max_tokens: usize,
    /// Sampling temperature (0.0 = deterministic, 0.7 = balanced).
    pub temperature: f32,
    /// Top-p nucleus sampling threshold.
    pub top_p: f32,
    /// Repetition penalty (1.0 = no penalty).
    pub repetition_penalty: f32,
}

impl Default for LlmBridgeConfig {
    fn default() -> Self {
        Self {
            model_id: String::new(),
            max_tokens: 512,
            temperature: 0.7,
            top_p: 0.9,
            repetition_penalty: 1.1,
        }
    }
}

/// Thread-safe bridge to the local LLM via ruvllm's CandleBackend.
pub struct LlmBridge {
    /// The inference backend (mutable for load/unload, so RwLock).
    backend: RwLock<CandleBackend>,
    /// Current configuration.
    config: RwLock<LlmBridgeConfig>,
    /// Model cache directory.
    cache_dir: PathBuf,
    /// Currently loaded model ID (for status reporting).
    loaded_model: RwLock<Option<String>>,
}

impl LlmBridge {
    /// Create a new LLM bridge with default cache directory.
    pub fn new() -> Self {
        let cache_dir = dirs::cache_dir()
            .unwrap_or_else(|| PathBuf::from("/tmp"))
            .join("DeepBrain")
            .join("models");

        Self {
            backend: RwLock::new(CandleBackend::default()),
            config: RwLock::new(LlmBridgeConfig::default()),
            cache_dir,
            loaded_model: RwLock::new(None),
        }
    }

    /// Create with a specific configuration.
    pub fn with_config(config: LlmBridgeConfig) -> Self {
        let bridge = Self::new();
        *bridge.config.write() = config;
        bridge
    }

    /// Load a model into the backend.
    ///
    /// `model_id` can be a HuggingFace model ID (e.g., "Qwen/Qwen2.5-1.5B-Instruct")
    /// or a local path to a GGUF file.
    pub fn load_model(&self, model_id: &str) -> Result<(), String> {
        let model_config = ModelConfig::default();

        let mut backend = self.backend.write();
        backend
            .load_model(model_id, model_config)
            .map_err(|e| format!("Failed to load model '{}': {}", model_id, e))?;

        *self.loaded_model.write() = Some(model_id.to_string());
        self.config.write().model_id = model_id.to_string();

        tracing::info!("Local LLM loaded: {}", model_id);
        Ok(())
    }

    /// Unload the current model (frees GPU/memory resources).
    pub fn unload_model(&self) {
        self.backend.write().unload_model();
        *self.loaded_model.write() = None;
        tracing::info!("Local LLM unloaded");
    }

    /// Check if a model is loaded and ready.
    pub fn is_loaded(&self) -> bool {
        self.backend.read().is_model_loaded()
    }

    /// Generate text using the loaded model.
    pub fn generate_text(
        &self,
        prompt: &str,
        max_tokens: Option<usize>,
        temperature: Option<f32>,
    ) -> Result<String, String> {
        let cfg = self.config.read();
        let params = GenerateParams {
            max_tokens: max_tokens.unwrap_or(cfg.max_tokens),
            temperature: temperature.unwrap_or(cfg.temperature),
            top_p: cfg.top_p,
            repetition_penalty: cfg.repetition_penalty,
            ..GenerateParams::default()
        };

        let backend = self.backend.read();
        backend
            .generate(prompt, params)
            .map_err(|e| format!("Generation failed: {}", e))
    }

    /// Get embeddings for a text string via the model's embedding layer.
    pub fn embed(&self, text: &str) -> Result<Vec<f32>, String> {
        let backend = self.backend.read();
        backend
            .get_embeddings(text)
            .map_err(|e| format!("Embedding failed: {}", e))
    }

    /// Get the current LLM status.
    pub fn status(&self) -> LlmStatus {
        let backend = self.backend.read();
        let model_info = backend.model_info().map(|info| {
            format!(
                "{:?} ({}M params)",
                info.architecture,
                info.num_parameters / 1_000_000,
            )
        });

        LlmStatus {
            model_loaded: backend.is_model_loaded(),
            model_id: self.loaded_model.read().clone(),
            model_info,
        }
    }

    /// Get the model cache directory.
    pub fn cache_dir(&self) -> &PathBuf {
        &self.cache_dir
    }
}

/// Implement the AiProvider trait so LlmBridge can be used as a drop-in
/// replacement for Ollama/Claude in the think pipeline.
#[async_trait::async_trait]
impl AiProvider for Arc<LlmBridge> {
    async fn generate(
        &self,
        prompt: &str,
        context_memories: &[RecallResult],
        extra_context: &str,
    ) -> Result<AiResponse, String> {
        if !self.is_loaded() {
            return Err("No local LLM model loaded".to_string());
        }

        let memory_context = format_memory_context(context_memories);

        let full_prompt = format!(
            "You are DeepBrain, an intelligent cognitive assistant running on macOS with \
             local LLM inference via RuVector. You have access to the user's memories, \
             indexed files, and system search results. Use the provided context to give \
             helpful, accurate responses.\n\
             {memory_context}\
             {extra_context}\
             User: {prompt}\n\
             Assistant:",
        );

        let response = self.generate_text(&full_prompt, None, None)?;

        Ok(AiResponse {
            content: response.trim().to_string(),
            model: self
                .loaded_model
                .read()
                .clone()
                .unwrap_or_else(|| "local".to_string()),
            tokens_used: None,
        })
    }

    async fn is_available(&self) -> bool {
        self.is_loaded()
    }

    fn name(&self) -> &str {
        "ruvllm"
    }
}
