//! Workflow automation for DeepBrain
//!
//! Built-in actions that combine multiple cognitive operations.

use std::sync::Arc;

use serde::{Deserialize, Serialize};

use crate::brain::cognitive::CognitiveEngine;
use crate::brain::embeddings::EmbeddingModel;
use crate::context::ContextManager;

/// Available workflow actions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum WorkflowAction {
    RememberClipboard,
    SummarizeRecent,
    LearningDigest,
    SearchAndRemember { query: String },
}

/// Workflow execution result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkflowResult {
    pub action: String,
    pub success: bool,
    pub message: String,
    pub data: Option<serde_json::Value>,
}

/// Execute a workflow action
pub async fn execute_workflow(
    action: WorkflowAction,
    engine: &Arc<CognitiveEngine>,
    embeddings: &Arc<EmbeddingModel>,
    context: &ContextManager,
) -> Result<WorkflowResult, String> {
    match action {
        WorkflowAction::RememberClipboard => {
            remember_clipboard(engine, embeddings, context).await
        }
        WorkflowAction::SummarizeRecent => {
            summarize_recent(engine).await
        }
        WorkflowAction::LearningDigest => {
            learning_digest(engine).await
        }
        WorkflowAction::SearchAndRemember { query } => {
            search_and_remember(&query, engine, embeddings).await
        }
    }
}

/// Remember the most recent clipboard content
async fn remember_clipboard(
    engine: &Arc<CognitiveEngine>,
    embeddings: &Arc<EmbeddingModel>,
    context: &ContextManager,
) -> Result<WorkflowResult, String> {
    let content = context
        .last_clipboard()
        .ok_or("No clipboard content available")?;

    let vector = embeddings.embed(&content).await?;
    let id = engine.remember_with_embedding(
        content.clone(),
        vector,
        "working".to_string(),
        Some(0.6),
    )?;

    Ok(WorkflowResult {
        action: "RememberClipboard".to_string(),
        success: true,
        message: format!("Stored clipboard content as memory {}", &id[..8]),
        data: Some(serde_json::json!({ "id": id, "content_preview": &content[..content.len().min(100)] })),
    })
}

/// Summarize recent thoughts and memories
async fn summarize_recent(engine: &Arc<CognitiveEngine>) -> Result<WorkflowResult, String> {
    let thoughts = engine.get_thoughts(Some(10));
    let stats = engine.stats();
    let introspection = engine.introspect();

    let summary = format!(
        "Brain Status: {}\n\
         Memories: {}, Thoughts: {}, Experiences: {}\n\
         Learning Trend: {}\n\
         Recent Thoughts:\n{}",
        introspection.status,
        stats.total_memories,
        stats.total_thoughts,
        stats.total_experiences,
        introspection.learning_trend,
        thoughts
            .iter()
            .take(5)
            .map(|t| format!("  - [{}] {}", t.thought_type, &t.content[..t.content.len().min(80)]))
            .collect::<Vec<_>>()
            .join("\n")
    );

    Ok(WorkflowResult {
        action: "SummarizeRecent".to_string(),
        success: true,
        message: summary,
        data: Some(serde_json::json!({
            "memories": stats.total_memories,
            "thoughts": stats.total_thoughts,
            "trend": introspection.learning_trend,
        })),
    })
}

/// Generate a learning digest
async fn learning_digest(engine: &Arc<CognitiveEngine>) -> Result<WorkflowResult, String> {
    let stats = engine.stats();
    let evolution = engine.evolve();

    let digest = format!(
        "Learning Digest:\n\
         - Total memories stored: {}\n\
         - Average reward: {:.3}\n\
         - Learning trend: {:.3}\n\
         - Adaptations: {}\n\
         - Improvements: {}",
        stats.total_memories,
        stats.avg_reward,
        stats.learning_trend,
        if evolution.adaptations.is_empty() {
            "None needed".to_string()
        } else {
            evolution.adaptations.join(", ")
        },
        if evolution.improvements.is_empty() {
            "System performing normally".to_string()
        } else {
            evolution.improvements.join(", ")
        },
    );

    Ok(WorkflowResult {
        action: "LearningDigest".to_string(),
        success: true,
        message: digest,
        data: None,
    })
}

/// Search for information and store results as memories
async fn search_and_remember(
    query: &str,
    engine: &Arc<CognitiveEngine>,
    embeddings: &Arc<EmbeddingModel>,
) -> Result<WorkflowResult, String> {
    let vector = embeddings.embed(query).await?;
    let results = engine.recall_f32(&vector, Some(5), None)?;

    if results.is_empty() {
        return Ok(WorkflowResult {
            action: "SearchAndRemember".to_string(),
            success: true,
            message: format!("No results found for '{}'", query),
            data: None,
        });
    }

    Ok(WorkflowResult {
        action: "SearchAndRemember".to_string(),
        success: true,
        message: format!(
            "Found {} relevant memories for '{}'",
            results.len(),
            query
        ),
        data: Some(serde_json::json!({
            "count": results.len(),
            "top_result": results.first().map(|r| &r.content),
        })),
    })
}
