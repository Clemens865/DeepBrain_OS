//! Claude Desktop connector — imports agent-mode sessions from Claude Desktop app.

use std::path::PathBuf;

use crate::indexer::bootstrap::{
    collect_files_with_ext, deterministic_hash, emit_progress, parse_claude_jsonl,
    store_as_memory, BootstrapProgress, SourceResult,
};
use crate::indexer::chunker;
use crate::state::AppState;

use super::{Connector, ConnectorConfig, DetectionResult};

pub struct ClaudeDesktopConnector;

impl ClaudeDesktopConnector {
    fn default_path() -> Option<PathBuf> {
        dirs::home_dir().map(|h| {
            h.join("Library")
                .join("Application Support")
                .join("Claude")
                .join("local-agent-mode-sessions")
        })
    }
}

#[async_trait::async_trait]
impl Connector for ClaudeDesktopConnector {
    fn id(&self) -> &str {
        "claude_desktop"
    }
    fn name(&self) -> &str {
        "Claude Desktop"
    }
    fn description(&self) -> &str {
        "Agent sessions"
    }
    fn icon(&self) -> &str {
        "desktop"
    }
    fn default_enabled(&self) -> bool {
        true
    }
    fn memory_type(&self) -> &str {
        "episodic"
    }
    fn default_importance(&self) -> f64 {
        0.65
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
            source: "claude_desktop".to_string(),
            ..Default::default()
        };

        let sessions_dir = config
            .path_override
            .as_ref()
            .map(PathBuf::from)
            .or_else(Self::default_path);

        let sessions_dir = match sessions_dir {
            Some(d) if d.exists() => d,
            _ => {
                tracing::info!("Bootstrap: Claude Desktop sessions not found, skipping");
                return result;
            }
        };

        let mut session_files = Vec::new();
        collect_files_with_ext(&sessions_dir, "jsonl", &mut session_files, 3);

        let total = session_files.len() as u32;
        tracing::info!("Bootstrap claude_desktop: found {} session files", total);
        let importance = config.importance_override.unwrap_or(self.default_importance());

        for (i, file_path) in session_files.iter().enumerate() {
            result.items_scanned += 1;

            let content = match std::fs::read_to_string(file_path) {
                Ok(c) => c,
                Err(_) => continue,
            };

            let file_hash = deterministic_hash(&file_path.to_string_lossy());
            let conversation_text = parse_claude_jsonl(&content);

            if conversation_text.trim().len() < 50 {
                continue;
            }

            let segments = chunker::chunk_text(&conversation_text, 400, 50);

            for (si, segment) in segments.iter().enumerate().take(15) {
                let seg_id = format!("bootstrap::claude_desktop::{}::{}", file_hash, si);

                match store_as_memory(state, &seg_id, segment, "episodic", importance).await {
                    Ok(true) => result.memories_created += 1,
                    Ok(false) => result.skipped_existing += 1,
                    Err(e) => {
                        tracing::debug!("Bootstrap claude_desktop error: {}", e);
                        result.errors += 1;
                    }
                }
            }

            emit_progress(
                app_handle,
                &BootstrapProgress {
                    source: "claude_desktop".to_string(),
                    phase: "storing".to_string(),
                    current: i as u32 + 1,
                    total,
                    memories_created: result.memories_created,
                },
            );

            if i % 5 == 4 {
                tokio::task::yield_now().await;
            }
        }

        tracing::info!(
            "Bootstrap claude_desktop complete: {} created, {} skipped",
            result.memories_created,
            result.skipped_existing
        );
        result
    }
}
