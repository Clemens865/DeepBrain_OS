//! Claude (Anthropic) cloud AI provider

use serde::{Deserialize, Serialize};

use crate::ai::{format_memory_context, AiResponse};
use crate::brain::cognitive::RecallResult;

/// Claude provider configuration
pub struct ClaudeProvider {
    api_key: String,
    model: String,
    client: reqwest::Client,
}

impl ClaudeProvider {
    pub fn new(api_key: &str) -> Self {
        Self {
            api_key: api_key.to_string(),
            model: "claude-sonnet-4-5-20250929".to_string(),
            client: reqwest::Client::new(),
        }
    }

    pub fn with_model(api_key: &str, model: &str) -> Self {
        Self {
            api_key: api_key.to_string(),
            model: model.to_string(),
            client: reqwest::Client::new(),
        }
    }
}

#[derive(Serialize)]
struct MessagesRequest {
    model: String,
    max_tokens: u32,
    messages: Vec<Message>,
    system: String,
}

#[derive(Serialize)]
struct Message {
    role: String,
    content: String,
}

#[derive(Deserialize)]
struct MessagesResponse {
    content: Vec<ContentBlock>,
    usage: Usage,
}

#[derive(Deserialize)]
struct ContentBlock {
    text: String,
}

#[derive(Deserialize)]
struct Usage {
    output_tokens: u32,
}

#[async_trait::async_trait]
impl super::AiProvider for ClaudeProvider {
    async fn generate(
        &self,
        prompt: &str,
        context_memories: &[RecallResult],
        extra_context: &str,
    ) -> Result<AiResponse, String> {
        let memory_context = format_memory_context(context_memories);

        let system_prompt = format!(
            "You are DeepBrain, an intelligent cognitive assistant running as a macOS app. \
             You have access to the user's memories, indexed files, and system search results. \
             Use ALL of the following context to provide an informed, helpful response. \
             Be concise and helpful.\n\
             {memory_context}\
             {extra_context}"
        );

        let url = "https://api.anthropic.com/v1/messages";

        let resp = self
            .client
            .post(url)
            .header("x-api-key", &self.api_key)
            .header("anthropic-version", "2023-06-01")
            .header("content-type", "application/json")
            .json(&MessagesRequest {
                model: self.model.clone(),
                max_tokens: 1024,
                system: system_prompt,
                messages: vec![Message {
                    role: "user".to_string(),
                    content: prompt.to_string(),
                }],
            })
            .timeout(std::time::Duration::from_secs(60))
            .send()
            .await
            .map_err(|e| format!("Claude API request failed: {}", e))?;

        if !resp.status().is_success() {
            let status = resp.status();
            let body = resp.text().await.unwrap_or_default();
            return Err(format!("Claude API error ({}): {}", status, body));
        }

        let body: MessagesResponse = resp
            .json()
            .await
            .map_err(|e| format!("Failed to parse Claude response: {}", e))?;

        let content = body
            .content
            .first()
            .map(|c| c.text.clone())
            .unwrap_or_default();

        Ok(AiResponse {
            content,
            model: self.model.clone(),
            tokens_used: Some(body.usage.output_tokens),
        })
    }

    async fn is_available(&self) -> bool {
        !self.api_key.is_empty()
    }

    fn name(&self) -> &str {
        "claude"
    }
}
