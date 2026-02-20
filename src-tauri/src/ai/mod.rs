//! AI Provider layer for DeepBrain
//!
//! Supports local (Ollama) and cloud (Claude) LLM providers.

pub mod claude;
pub mod ollama;

use serde::{Deserialize, Serialize};

use crate::brain::cognitive::RecallResult;

/// Response from an AI provider
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AiResponse {
    pub content: String,
    pub model: String,
    pub tokens_used: Option<u32>,
}

/// AI provider trait
#[async_trait::async_trait]
pub trait AiProvider: Send + Sync {
    async fn generate(
        &self,
        prompt: &str,
        context_memories: &[RecallResult],
        extra_context: &str,
    ) -> Result<AiResponse, String>;

    async fn is_available(&self) -> bool;

    fn name(&self) -> &str;
}

/// Format memory context for LLM prompts
pub fn format_memory_context(memories: &[RecallResult]) -> String {
    if memories.is_empty() {
        return String::new();
    }

    let mut context = String::from("\n--- Relevant Memories ---\n");
    for (i, mem) in memories.iter().enumerate() {
        context.push_str(&format!(
            "{}. [{}] (similarity: {:.2}): {}\n",
            i + 1,
            mem.memory_type,
            mem.similarity,
            mem.content
        ));
    }
    context.push_str("--- End Memories ---\n\n");
    context
}
