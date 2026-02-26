//! Claude Plans connector — imports implementation plans from ~/.claude/plans/*.md.

use std::path::PathBuf;

use crate::indexer::bootstrap::{
    collect_files_with_ext, deterministic_hash, emit_progress, store_as_memory,
    BootstrapProgress, SourceResult,
};
use crate::indexer::chunker;
use crate::state::AppState;

use super::{Connector, ConnectorConfig, DetectionResult};

pub struct ClaudePlansConnector;

impl ClaudePlansConnector {
    fn default_path() -> Option<PathBuf> {
        dirs::home_dir().map(|h| h.join(".claude").join("plans"))
    }
}

#[async_trait::async_trait]
impl Connector for ClaudePlansConnector {
    fn id(&self) -> &str {
        "claude_plans"
    }
    fn name(&self) -> &str {
        "Claude Plans"
    }
    fn description(&self) -> &str {
        "Implementation plans"
    }
    fn icon(&self) -> &str {
        "plan"
    }
    fn default_enabled(&self) -> bool {
        true
    }
    fn memory_type(&self) -> &str {
        "goal"
    }
    fn default_importance(&self) -> f64 {
        0.75
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
            source: "claude_plans".to_string(),
            ..Default::default()
        };

        let plans_dir = config
            .path_override
            .as_ref()
            .map(PathBuf::from)
            .or_else(Self::default_path);

        let plans_dir = match plans_dir {
            Some(d) if d.exists() => d,
            _ => {
                tracing::info!("Bootstrap: claude_plans path not found, skipping");
                return result;
            }
        };

        let mut plan_files = Vec::new();
        collect_files_with_ext(&plans_dir, "md", &mut plan_files, 2);

        let total = plan_files.len() as u32;
        tracing::info!("Bootstrap claude_plans: found {} plan files", total);
        let importance = config.importance_override.unwrap_or(self.default_importance());

        for (i, file_path) in plan_files.iter().enumerate() {
            result.items_scanned += 1;

            let content = match std::fs::read_to_string(file_path) {
                Ok(c) => c,
                Err(_) => continue,
            };

            if content.trim().len() < 50 {
                continue;
            }

            let file_hash = deterministic_hash(&file_path.to_string_lossy());
            let file_name = file_path
                .file_name()
                .unwrap_or_default()
                .to_string_lossy();

            let enriched = format!("[Implementation Plan: {}]\n{}", file_name, content);
            let chunks = chunker::chunk_text(&enriched, 512, 128);

            for (ci, chunk) in chunks.iter().enumerate() {
                let chunk_id = format!("bootstrap::claude_plan::{}::{}", file_hash, ci);
                match store_as_memory(state, &chunk_id, chunk, "goal", importance).await {
                    Ok(true) => result.memories_created += 1,
                    Ok(false) => result.skipped_existing += 1,
                    Err(e) => {
                        tracing::debug!("Bootstrap claude_plans error: {}", e);
                        result.errors += 1;
                    }
                }
            }

            emit_progress(
                app_handle,
                &BootstrapProgress {
                    source: "claude_plans".to_string(),
                    phase: "storing".to_string(),
                    current: i as u32 + 1,
                    total,
                    memories_created: result.memories_created,
                },
            );
        }

        tracing::info!(
            "Bootstrap claude_plans complete: {} created, {} skipped",
            result.memories_created,
            result.skipped_existing
        );
        result
    }
}
