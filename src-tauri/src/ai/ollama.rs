//! Ollama local LLM provider

use serde::{Deserialize, Serialize};

use crate::ai::{format_memory_context, AiResponse};
use crate::brain::cognitive::RecallResult;

/// Ollama provider configuration
pub struct OllamaProvider {
    base_url: String,
    model: String,
    client: reqwest::Client,
}

impl OllamaProvider {
    pub fn new(model: &str) -> Self {
        Self {
            base_url: "http://127.0.0.1:11434".to_string(),
            model: model.to_string(),
            client: reqwest::Client::new(),
        }
    }

    pub fn with_url(base_url: &str, model: &str) -> Self {
        Self {
            base_url: base_url.to_string(),
            model: model.to_string(),
            client: reqwest::Client::new(),
        }
    }
}

#[derive(Serialize)]
struct GenerateRequest {
    model: String,
    prompt: String,
    stream: bool,
    options: GenerateOptions,
}

#[derive(Serialize)]
struct GenerateOptions {
    num_ctx: u32,
}

#[derive(Deserialize)]
struct GenerateResponse {
    response: String,
}

#[derive(Deserialize)]
struct TagsResponse {
    models: Vec<ModelInfo>,
}

#[derive(Deserialize)]
struct ModelInfo {
    name: String,
}

#[async_trait::async_trait]
impl super::AiProvider for OllamaProvider {
    async fn generate(
        &self,
        prompt: &str,
        context_memories: &[RecallResult],
        extra_context: &str,
    ) -> Result<AiResponse, String> {
        let memory_context = format_memory_context(context_memories);

        let full_prompt = format!(
            "You are DeepBrain, an intelligent cognitive assistant running on macOS. \
             You have access to the user's memories, indexed files, and system search results. \
             Use ALL of the following context to provide an informed, helpful response.\n\
             {memory_context}\
             {extra_context}\
             User: {prompt}\n\
             Assistant:"
        );

        let url = format!("{}/api/generate", self.base_url);

        let resp = self
            .client
            .post(&url)
            .json(&GenerateRequest {
                model: self.model.clone(),
                prompt: full_prompt,
                stream: false,
                options: GenerateOptions { num_ctx: 8192 },
            })
            .timeout(std::time::Duration::from_secs(30))
            .send()
            .await
            .map_err(|e| format!("Ollama request failed: {}", e))?;

        if !resp.status().is_success() {
            return Err(format!("Ollama returned status: {}", resp.status()));
        }

        let body: GenerateResponse = resp
            .json()
            .await
            .map_err(|e| format!("Failed to parse response: {}", e))?;

        Ok(AiResponse {
            content: body.response.trim().to_string(),
            model: self.model.clone(),
            tokens_used: None,
        })
    }

    async fn is_available(&self) -> bool {
        let url = format!("{}/api/tags", self.base_url);
        self.client
            .get(&url)
            .timeout(std::time::Duration::from_secs(2))
            .send()
            .await
            .map(|r| r.status().is_success())
            .unwrap_or(false)
    }

    fn name(&self) -> &str {
        "ollama"
    }
}

/// List available Ollama models
pub async fn list_models(base_url: &str) -> Result<Vec<String>, String> {
    let client = reqwest::Client::new();
    let url = format!("{}/api/tags", base_url);

    let resp = client
        .get(&url)
        .timeout(std::time::Duration::from_secs(5))
        .send()
        .await
        .map_err(|e| format!("Failed to connect to Ollama: {}", e))?;

    let tags: TagsResponse = resp
        .json()
        .await
        .map_err(|e| format!("Failed to parse models: {}", e))?;

    Ok(tags.models.into_iter().map(|m| m.name).collect())
}
