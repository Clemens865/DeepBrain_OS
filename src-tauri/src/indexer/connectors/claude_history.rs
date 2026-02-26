//! Claude History connector — imports session prompts/commands from ~/.claude/history.jsonl.

use std::path::PathBuf;

use crate::indexer::bootstrap::{
    deterministic_hash, emit_progress, store_as_memory, BootstrapProgress, SourceResult,
};
use crate::state::AppState;

use super::{Connector, ConnectorConfig, DetectionResult};

pub struct ClaudeHistoryConnector;

impl ClaudeHistoryConnector {
    fn default_path() -> Option<PathBuf> {
        dirs::home_dir().map(|h| h.join(".claude").join("history.jsonl"))
    }
}

#[async_trait::async_trait]
impl Connector for ClaudeHistoryConnector {
    fn id(&self) -> &str {
        "claude_history"
    }
    fn name(&self) -> &str {
        "Claude History"
    }
    fn description(&self) -> &str {
        "Session prompts"
    }
    fn icon(&self) -> &str {
        "history"
    }
    fn default_enabled(&self) -> bool {
        true
    }
    fn memory_type(&self) -> &str {
        "episodic"
    }
    fn default_importance(&self) -> f64 {
        0.5
    }

    fn detect(&self) -> DetectionResult {
        let path = Self::default_path();
        let available = path.as_ref().map(|p| p.exists()).unwrap_or(false);
        DetectionResult {
            available,
            path: path.map(|p| p.to_string_lossy().to_string()),
            details: None,
        }
    }

    async fn import(
        &self,
        state: &AppState,
        app_handle: Option<&tauri::AppHandle>,
        config: &ConnectorConfig,
    ) -> SourceResult {
        let mut result = SourceResult {
            source: "claude_history".to_string(),
            ..Default::default()
        };

        let history_file = config
            .path_override
            .as_ref()
            .map(PathBuf::from)
            .or_else(Self::default_path);

        let history_file = match history_file {
            Some(f) if f.exists() => f,
            _ => {
                tracing::info!("Bootstrap: claude history file not found, skipping");
                return result;
            }
        };

        let content = match std::fs::read_to_string(&history_file) {
            Ok(c) => c,
            Err(e) => {
                tracing::info!("Bootstrap claude_history: read error: {}", e);
                return result;
            }
        };

        let mut entries: Vec<String> = Vec::new();
        for line in content.lines() {
            if line.trim().is_empty() {
                continue;
            }
            if let Ok(json) = serde_json::from_str::<serde_json::Value>(line) {
                let mut entry_text = String::new();

                if let Some(prompt) = json.get("prompt").and_then(|p| p.as_str()) {
                    entry_text.push_str(prompt);
                }
                if let Some(command) = json.get("command").and_then(|c| c.as_str()) {
                    if !command.is_empty() && entry_text.is_empty() {
                        entry_text.push_str(command);
                    }
                }
                if let Some(cwd) = json.get("cwd").and_then(|c| c.as_str()) {
                    if !entry_text.is_empty() {
                        entry_text = format!("[Project: {}] {}", cwd, entry_text);
                    }
                }

                if entry_text.len() > 20 {
                    entries.push(entry_text);
                }
            }
        }

        let total = entries.len() as u32;
        tracing::info!("Bootstrap claude_history: found {} history entries", total);
        let importance = config.importance_override.unwrap_or(self.default_importance());

        let batch_size = 10;
        let batches: Vec<String> = entries
            .chunks(batch_size)
            .map(|batch| batch.join("\n---\n"))
            .collect();

        for (i, batch) in batches.iter().enumerate() {
            let batch_hash = deterministic_hash(batch);
            let memory_id = format!("bootstrap::claude_history::{}", batch_hash);

            match store_as_memory(state, &memory_id, batch, "episodic", importance).await {
                Ok(true) => result.memories_created += 1,
                Ok(false) => result.skipped_existing += 1,
                Err(e) => {
                    tracing::debug!("Bootstrap claude_history error: {}", e);
                    result.errors += 1;
                }
            }
            result.items_scanned += batch_size.min(entries.len() - i * batch_size) as u32;

            if i % 10 == 0 {
                emit_progress(
                    app_handle,
                    &BootstrapProgress {
                        source: "claude_history".to_string(),
                        phase: "storing".to_string(),
                        current: i as u32 + 1,
                        total: batches.len() as u32,
                        memories_created: result.memories_created,
                    },
                );
                tokio::task::yield_now().await;
            }
        }

        tracing::info!(
            "Bootstrap claude_history complete: {} created, {} skipped",
            result.memories_created,
            result.skipped_existing
        );
        result
    }
}
