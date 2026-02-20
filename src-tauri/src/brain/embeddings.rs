//! Embedding model for DeepBrain
//!
//! Supports:
//! - ONNX all-MiniLM-L6-v2 (384-dim, local, fast)
//! - Ollama embeddings API (fallback)
//! - Simple hash-based embeddings (ultimate fallback)

use std::path::PathBuf;

use parking_lot::RwLock;
use serde::{Deserialize, Serialize};

use crate::brain::utils::normalize_vector;

const EMBEDDING_DIM: usize = 384;
const MODEL_REPO: &str = "sentence-transformers/all-MiniLM-L6-v2";
const MODEL_URL: &str = "https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/main/onnx/model.onnx";
const TOKENIZER_URL: &str = "https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/main/tokenizer.json";

/// Embedding provider type
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum EmbeddingProvider {
    Onnx,
    Ollama,
    Hash,
}

/// Holds a loaded ONNX session + tokenizer
struct OnnxSession {
    session: ort::session::Session,
    tokenizer: tokenizers::Tokenizer,
}

/// Embedding model manager
pub struct EmbeddingModel {
    provider: RwLock<EmbeddingProvider>,
    onnx_session: parking_lot::Mutex<Option<OnnxSession>>,
    ollama_url: String,
    ollama_model: String,
    model_dir: PathBuf,
}

impl EmbeddingModel {
    /// Create a new embedding model (starts with hash fallback, can be upgraded)
    pub fn new() -> Self {
        let model_dir = dirs::data_dir()
            .unwrap_or_else(|| PathBuf::from("."))
            .join("DeepBrain")
            .join("models");

        Self {
            provider: RwLock::new(EmbeddingProvider::Hash),
            onnx_session: parking_lot::Mutex::new(None),
            ollama_url: "http://127.0.0.1:11434".to_string(),
            ollama_model: "nomic-embed-text".to_string(),
            model_dir,
        }
    }

    /// Try to initialize the best available embedding provider
    /// Priority: ONNX (local, fast) > Ollama > Hash (fallback)
    pub async fn try_init_ollama(&self) -> bool {
        // First, try ONNX (best option: fast, offline, semantic)
        if self.try_init_onnx().await {
            return true;
        }

        // Then try Ollama
        let client = reqwest::Client::new();
        let url = format!("{}/api/tags", self.ollama_url);

        match client.get(&url).timeout(std::time::Duration::from_secs(2)).send().await {
            Ok(resp) if resp.status().is_success() => {
                *self.provider.write() = EmbeddingProvider::Ollama;
                tracing::info!("Ollama embedding provider initialized");
                true
            }
            _ => {
                tracing::warn!("Ollama not available, using hash embeddings");
                false
            }
        }
    }

    /// Try to initialize ONNX model (download if needed)
    async fn try_init_onnx(&self) -> bool {
        let model_path = self.model_dir.join("model.onnx");
        let tokenizer_path = self.model_dir.join("tokenizer.json");

        // Download model files if not present
        if !model_path.exists() || !tokenizer_path.exists() {
            tracing::info!("ONNX model not found, downloading {}...", MODEL_REPO);
            if let Err(e) = self.download_model_files().await {
                tracing::warn!("Failed to download ONNX model: {}", e);
                return false;
            }
        }

        // Load ONNX session
        match self.load_onnx_session(&model_path, &tokenizer_path) {
            Ok(session) => {
                *self.onnx_session.lock() = Some(session);
                *self.provider.write() = EmbeddingProvider::Onnx;
                tracing::info!("ONNX embedding model loaded (all-MiniLM-L6-v2)");
                true
            }
            Err(e) => {
                tracing::warn!("Failed to load ONNX model: {}", e);
                false
            }
        }
    }

    /// Download model and tokenizer files from HuggingFace
    async fn download_model_files(&self) -> Result<(), String> {
        std::fs::create_dir_all(&self.model_dir)
            .map_err(|e| format!("Failed to create model dir: {}", e))?;

        let client = reqwest::Client::new();

        // Download model.onnx (~23MB)
        let model_path = self.model_dir.join("model.onnx");
        if !model_path.exists() {
            tracing::info!("Downloading model.onnx...");
            let resp = client.get(MODEL_URL)
                .timeout(std::time::Duration::from_secs(300))
                .send().await
                .map_err(|e| format!("Download failed: {}", e))?;

            if !resp.status().is_success() {
                return Err(format!("Model download returned status: {}", resp.status()));
            }

            let bytes = resp.bytes().await
                .map_err(|e| format!("Failed to read model bytes: {}", e))?;

            std::fs::write(&model_path, &bytes)
                .map_err(|e| format!("Failed to write model file: {}", e))?;

            tracing::info!("Downloaded model.onnx ({:.1}MB)", bytes.len() as f64 / 1_048_576.0);
        }

        // Download tokenizer.json (~700KB)
        let tokenizer_path = self.model_dir.join("tokenizer.json");
        if !tokenizer_path.exists() {
            tracing::info!("Downloading tokenizer.json...");
            let resp = client.get(TOKENIZER_URL)
                .timeout(std::time::Duration::from_secs(30))
                .send().await
                .map_err(|e| format!("Download failed: {}", e))?;

            if !resp.status().is_success() {
                return Err(format!("Tokenizer download returned status: {}", resp.status()));
            }

            let bytes = resp.bytes().await
                .map_err(|e| format!("Failed to read tokenizer bytes: {}", e))?;

            std::fs::write(&tokenizer_path, &bytes)
                .map_err(|e| format!("Failed to write tokenizer file: {}", e))?;

            tracing::info!("Downloaded tokenizer.json");
        }

        Ok(())
    }

    /// Load ONNX session and tokenizer from disk
    fn load_onnx_session(
        &self,
        model_path: &std::path::Path,
        tokenizer_path: &std::path::Path,
    ) -> Result<OnnxSession, String> {
        let session = ort::session::Session::builder()
            .map_err(|e| format!("Failed to create session builder: {}", e))?
            .with_intra_threads(2)
            .map_err(|e| format!("Failed to set threads: {}", e))?
            .commit_from_file(model_path)
            .map_err(|e| format!("Failed to load ONNX model: {}", e))?;

        let tokenizer = tokenizers::Tokenizer::from_file(tokenizer_path)
            .map_err(|e| format!("Failed to load tokenizer: {}", e))?;

        Ok(OnnxSession { session, tokenizer })
    }

    /// Get current provider type
    pub fn provider(&self) -> EmbeddingProvider {
        self.provider.read().clone()
    }

    /// Get embedding dimension
    pub fn dimensions(&self) -> usize {
        EMBEDDING_DIM
    }

    /// Embed a single text
    pub async fn embed(&self, text: &str) -> Result<Vec<f32>, String> {
        // Clone the provider to avoid holding lock across await
        let provider = self.provider.read().clone();
        match provider {
            EmbeddingProvider::Ollama => self.embed_ollama(text).await,
            EmbeddingProvider::Onnx => self.embed_onnx(text),
            EmbeddingProvider::Hash => Ok(self.embed_hash(text)),
        }
    }

    /// Embed multiple texts
    pub async fn embed_batch(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>, String> {
        let mut results = Vec::with_capacity(texts.len());
        for text in texts {
            results.push(self.embed(text).await?);
        }
        Ok(results)
    }

    /// Ollama embedding via REST API
    async fn embed_ollama(&self, text: &str) -> Result<Vec<f32>, String> {
        let client = reqwest::Client::new();
        let url = format!("{}/api/embed", self.ollama_url);

        #[derive(Serialize)]
        struct EmbedRequest<'a> {
            model: &'a str,
            input: &'a str,
        }

        #[derive(Deserialize)]
        struct EmbedResponse {
            embeddings: Vec<Vec<f64>>,
        }

        let resp = client
            .post(&url)
            .json(&EmbedRequest {
                model: &self.ollama_model,
                input: text,
            })
            .timeout(std::time::Duration::from_secs(30))
            .send()
            .await
            .map_err(|e| format!("Ollama request failed: {}", e))?;

        if !resp.status().is_success() {
            return Err(format!("Ollama returned status: {}", resp.status()));
        }

        let body: EmbedResponse = resp
            .json()
            .await
            .map_err(|e| format!("Failed to parse Ollama response: {}", e))?;

        if body.embeddings.is_empty() {
            return Err("No embeddings returned from Ollama".to_string());
        }

        let embedding: Vec<f32> = body.embeddings[0].iter().map(|&x| x as f32).collect();

        // Pad or truncate to EMBEDDING_DIM
        let mut result = if embedding.len() >= EMBEDDING_DIM {
            embedding[..EMBEDDING_DIM].to_vec()
        } else {
            let mut padded = embedding;
            padded.resize(EMBEDDING_DIM, 0.0);
            padded
        };

        normalize_vector(&mut result);
        Ok(result)
    }

    /// ONNX embedding using all-MiniLM-L6-v2
    fn embed_onnx(&self, text: &str) -> Result<Vec<f32>, String> {
        let mut session_guard = self.onnx_session.lock();
        let session = session_guard.as_mut()
            .ok_or("ONNX session not loaded")?;

        // Tokenize
        let encoding = session.tokenizer.encode(text, true)
            .map_err(|e| format!("Tokenization failed: {}", e))?;

        let input_ids: Vec<i64> = encoding.get_ids().iter().map(|&id| id as i64).collect();
        let attention_mask: Vec<i64> = encoding.get_attention_mask().iter().map(|&m| m as i64).collect();
        let token_type_ids: Vec<i64> = encoding.get_type_ids().iter().map(|&t| t as i64).collect();
        let seq_len = input_ids.len();

        // Create input tensors [1, seq_len]
        let shape = [1_usize, seq_len];
        let input_ids_tensor = ort::value::Tensor::from_array(
            ndarray::Array2::from_shape_vec(shape, input_ids)
                .map_err(|e| format!("input_ids shape error: {}", e))?
        ).map_err(|e| format!("input_ids tensor error: {}", e))?;

        let attention_mask_tensor = ort::value::Tensor::from_array(
            ndarray::Array2::from_shape_vec(shape, attention_mask.clone())
                .map_err(|e| format!("attention_mask shape error: {}", e))?
        ).map_err(|e| format!("attention_mask tensor error: {}", e))?;

        let token_type_ids_tensor = ort::value::Tensor::from_array(
            ndarray::Array2::from_shape_vec(shape, token_type_ids)
                .map_err(|e| format!("token_type_ids shape error: {}", e))?
        ).map_err(|e| format!("token_type_ids tensor error: {}", e))?;

        // Run inference
        let outputs = session.session.run(
            ort::inputs![
                "input_ids" => input_ids_tensor,
                "attention_mask" => attention_mask_tensor,
                "token_type_ids" => token_type_ids_tensor,
            ]
        ).map_err(|e| format!("ONNX inference failed: {}", e))?;

        // Extract last_hidden_state [1, seq_len, 384]
        let output = &outputs[0];
        let (output_shape, output_data) = output.try_extract_tensor::<f32>()
            .map_err(|e| format!("Failed to extract output tensor: {}", e))?;

        // output_shape should be [1, seq_len, 384]
        let hidden_dim = *output_shape.last().unwrap_or(&(EMBEDDING_DIM as i64)) as usize;

        // Mean pooling with attention mask
        let mut embedding = vec![0.0f32; EMBEDDING_DIM];
        let mut mask_sum = 0.0f32;

        for token_idx in 0..seq_len {
            let mask_val = attention_mask[token_idx] as f32;
            mask_sum += mask_val;
            let offset = token_idx * hidden_dim;
            for dim in 0..EMBEDDING_DIM.min(hidden_dim) {
                embedding[dim] += output_data[offset + dim] * mask_val;
            }
        }

        if mask_sum > 0.0 {
            for dim in 0..EMBEDDING_DIM {
                embedding[dim] /= mask_sum;
            }
        }

        normalize_vector(&mut embedding);
        Ok(embedding)
    }

    /// Hash-based embedding (deterministic, fast, but not semantic)
    /// Uses character n-gram hashing to produce a fixed-size vector.
    /// This provides basic word-level similarity but not true semantic understanding.
    fn embed_hash(&self, text: &str) -> Vec<f32> {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let mut embedding = vec![0.0f32; EMBEDDING_DIM];
        let text_lower = text.to_lowercase();
        let words: Vec<&str> = text_lower.split_whitespace().collect();

        // Character trigram features
        let chars: Vec<char> = text_lower.chars().collect();
        for window in chars.windows(3) {
            let mut hasher = DefaultHasher::new();
            window.hash(&mut hasher);
            let idx = (hasher.finish() % EMBEDDING_DIM as u64) as usize;
            embedding[idx] += 1.0;
        }

        // Word-level features
        for word in &words {
            let mut hasher = DefaultHasher::new();
            word.hash(&mut hasher);
            let hash = hasher.finish();
            let idx1 = (hash % EMBEDDING_DIM as u64) as usize;
            let idx2 = ((hash >> 16) % EMBEDDING_DIM as u64) as usize;
            embedding[idx1] += 2.0;
            embedding[idx2] += 1.0;
        }

        // Word bigram features
        for pair in words.windows(2) {
            let mut hasher = DefaultHasher::new();
            pair[0].hash(&mut hasher);
            pair[1].hash(&mut hasher);
            let idx = (hasher.finish() % EMBEDDING_DIM as u64) as usize;
            embedding[idx] += 1.5;
        }

        normalize_vector(&mut embedding);
        embedding
    }
}

impl Default for EmbeddingModel {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::brain::utils::cosine_similarity;

    #[tokio::test]
    async fn test_hash_embedding_similarity() {
        let model = EmbeddingModel::new();

        let cat = model.embed("cat").await.unwrap();
        let _kitten = model.embed("kitten").await.unwrap();
        let car = model.embed("automobile racing championship").await.unwrap();

        // Hash embeddings won't be truly semantic, but identical strings should match
        let cat2 = model.embed("cat").await.unwrap();
        let self_sim = cosine_similarity(&cat, &cat2);
        assert!(
            (self_sim - 1.0).abs() < 1e-6,
            "Same text should have similarity 1.0"
        );

        // Different texts should have some difference
        let cat_car_sim = cosine_similarity(&cat, &car);
        assert!(
            cat_car_sim < 0.95,
            "Very different texts should have lower similarity"
        );
    }

    #[tokio::test]
    async fn test_embedding_dimensions() {
        let model = EmbeddingModel::new();
        let embedding = model.embed("test text").await.unwrap();
        assert_eq!(embedding.len(), EMBEDDING_DIM);
    }
}
