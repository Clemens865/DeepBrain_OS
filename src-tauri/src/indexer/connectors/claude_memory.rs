//! Claude Memory connector — imports MEMORY.md files from ~/.claude/projects/**/memory/.

use std::path::PathBuf;

use crate::indexer::bootstrap::{
    collect_memory_md_files, deterministic_hash, emit_progress,
    store_as_memory, BootstrapProgress, SourceResult,
};
use crate::indexer::chunker;
use crate::state::AppState;

use super::{Connector, ConnectorConfig, DetectionResult};

pub struct ClaudeMemoryConnector;

impl ClaudeMemoryConnector {
    fn default_path() -> Option<PathBuf> {
        dirs::home_dir().map(|h| h.join(".claude").join("projects"))
    }
}

#[async_trait::async_trait]
impl Connector for ClaudeMemoryConnector {
    fn id(&self) -> &str {
        "claude_memory"
    }
    fn name(&self) -> &str {
        "Claude Memory"
    }
    fn description(&self) -> &str {
        "MEMORY.md files"
    }
    fn icon(&self) -> &str {
        "brain"
    }
    fn default_enabled(&self) -> bool {
        true
    }
    fn memory_type(&self) -> &str {
        "semantic"
    }
    fn default_importance(&self) -> f64 {
        0.8
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
            source: "claude_memory".to_string(),
            ..Default::default()
        };

        let home = match dirs::home_dir() {
            Some(h) => h,
            None => return result,
        };

        let claude_dir = config
            .path_override
            .as_ref()
            .map(PathBuf::from)
            .or_else(Self::default_path);

        let claude_dir = match claude_dir {
            Some(d) if d.exists() => d,
            _ => return result,
        };

        let mut memory_files = Vec::new();
        collect_memory_md_files(&claude_dir, &mut memory_files, 4);

        // Also check top-level MEMORY.md
        let top_memory = home.join(".claude").join("MEMORY.md");
        if top_memory.exists() {
            memory_files.push(top_memory);
        }

        let total = memory_files.len() as u32;
        tracing::info!("Bootstrap claude_memory: found {} memory files", total);
        let importance = config.importance_override.unwrap_or(self.default_importance());

        for (i, file_path) in memory_files.iter().enumerate() {
            result.items_scanned += 1;

            let content = match std::fs::read_to_string(file_path) {
                Ok(c) => c,
                Err(_) => continue,
            };

            if content.trim().len() < 20 {
                continue;
            }

            let file_hash = deterministic_hash(&file_path.to_string_lossy());
            let memory_id = format!("bootstrap::claude_memory::{}", file_hash);

            let enriched = format!(
                "[Claude Memory: {}]\n{}",
                file_path
                    .file_name()
                    .unwrap_or_default()
                    .to_string_lossy(),
                content
            );

            let chunks = chunker::chunk_text(&enriched, 512, 128);
            for (ci, chunk) in chunks.iter().enumerate() {
                let chunk_id = format!("{}::{}", memory_id, ci);
                match store_as_memory(state, &chunk_id, chunk, "semantic", importance).await {
                    Ok(true) => result.memories_created += 1,
                    Ok(false) => result.skipped_existing += 1,
                    Err(e) => {
                        tracing::debug!("Bootstrap claude_memory error: {}", e);
                        result.errors += 1;
                    }
                }
            }

            emit_progress(
                app_handle,
                &BootstrapProgress {
                    source: "claude_memory".to_string(),
                    phase: "storing".to_string(),
                    current: i as u32 + 1,
                    total,
                    memories_created: result.memories_created,
                },
            );
        }

        tracing::info!(
            "Bootstrap claude_memory complete: {} created, {} skipped",
            result.memories_created,
            result.skipped_existing
        );
        result
    }
}
