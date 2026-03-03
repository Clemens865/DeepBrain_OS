//! Obsidian connector — imports markdown notes from an Obsidian vault.
//!
//! Recursively scans `.md` files, strips YAML frontmatter, extracts tags,
//! and chunks long notes. Requires `path_override` to be set to the vault
//! directory (or auto-detects common locations).

use std::path::{Path, PathBuf};

use crate::indexer::bootstrap::{
    deterministic_hash, emit_progress, store_as_memory_with_summary, BootstrapProgress,
    SourceResult,
};
use crate::indexer::chunker;
use crate::state::AppState;

use super::{Connector, ConnectorConfig, DetectionResult};

pub struct ObsidianConnector;

/// Common locations where Obsidian vaults are found on macOS.
const COMMON_VAULT_LOCATIONS: &[&str] = &[
    "Documents/Obsidian",
    "Documents/Obsidian Vault",
    "Obsidian",
    "vaults",
];

/// Find an Obsidian vault directory (contains `.obsidian/` subdirectory).
fn find_vault(path_override: Option<&str>) -> Option<PathBuf> {
    // Check explicit override first
    if let Some(p) = path_override {
        let path = PathBuf::from(p);
        if path.join(".obsidian").is_dir() {
            return Some(path);
        }
    }

    // Scan common locations
    let home = dirs::home_dir()?;
    for loc in COMMON_VAULT_LOCATIONS {
        let candidate = home.join(loc);
        if candidate.join(".obsidian").is_dir() {
            return Some(candidate);
        }
        // Also check subdirectories (user may have multiple vaults in a parent folder)
        if candidate.is_dir() {
            if let Ok(entries) = std::fs::read_dir(&candidate) {
                for entry in entries.flatten() {
                    let sub = entry.path();
                    if sub.is_dir() && sub.join(".obsidian").is_dir() {
                        return Some(sub);
                    }
                }
            }
        }
    }
    None
}

/// Recursively collect `.md` files from the vault, skipping internal directories.
fn collect_md_files(dir: &Path, files: &mut Vec<PathBuf>, max_depth: u32) {
    if max_depth == 0 {
        return;
    }

    let entries = match std::fs::read_dir(dir) {
        Ok(e) => e,
        Err(_) => return,
    };

    for entry in entries.flatten() {
        let path = entry.path();
        let name = entry.file_name();
        let name_str = name.to_string_lossy();

        // Skip Obsidian internal dirs, hidden dirs, and trash
        if path.is_dir() {
            if name_str.starts_with('.')
                || name_str == "templates"
                || name_str == ".trash"
                || name_str == "node_modules"
            {
                continue;
            }
            collect_md_files(&path, files, max_depth - 1);
        } else if name_str.ends_with(".md") {
            files.push(path);
        }
    }
}

/// Strip YAML frontmatter from markdown content and extract tags.
/// Returns (body_without_frontmatter, tags_string).
fn strip_frontmatter(content: &str) -> (String, String) {
    let mut tags = Vec::new();

    let body = if content.starts_with("---") {
        // Find closing ---
        if let Some(end) = content[3..].find("\n---") {
            let frontmatter = &content[3..3 + end];

            // Extract tags from frontmatter
            for line in frontmatter.lines() {
                let trimmed = line.trim();
                if let Some(rest) = trimmed.strip_prefix("tags:") {
                    // Inline YAML list: tags: [a, b, c] or tags: a, b, c
                    let rest = rest.trim().trim_start_matches('[').trim_end_matches(']');
                    for tag in rest.split(',') {
                        let t = tag.trim().trim_matches('"').trim_matches('\'').trim();
                        if !t.is_empty() {
                            tags.push(t.to_string());
                        }
                    }
                } else if trimmed.starts_with("- ") && !tags.is_empty() {
                    // YAML list continuation
                    let t = trimmed
                        .strip_prefix("- ")
                        .unwrap_or("")
                        .trim()
                        .trim_matches('"')
                        .trim_matches('\'');
                    if !t.is_empty() {
                        tags.push(t.to_string());
                    }
                }
            }

            content[3 + end + 4..].to_string() // skip past closing ---\n
        } else {
            content.to_string()
        }
    } else {
        content.to_string()
    };

    let tags_str = if tags.is_empty() {
        String::new()
    } else {
        tags.join(", ")
    };

    (body, tags_str)
}

#[async_trait::async_trait]
impl Connector for ObsidianConnector {
    fn id(&self) -> &str {
        "obsidian"
    }
    fn name(&self) -> &str {
        "Obsidian"
    }
    fn description(&self) -> &str {
        "Markdown notes from Obsidian vault"
    }
    fn icon(&self) -> &str {
        "gem"
    }
    fn default_enabled(&self) -> bool {
        false
    }
    fn memory_type(&self) -> &str {
        "semantic"
    }
    fn default_importance(&self) -> f64 {
        0.6
    }

    fn detect(&self) -> DetectionResult {
        match find_vault(None) {
            Some(path) => DetectionResult {
                available: true,
                path: Some(path.to_string_lossy().to_string()),
                details: Some(format!("Vault: {}", path.display())),
            },
            None => DetectionResult {
                available: false,
                path: None,
                details: Some("Set vault path in connector settings".to_string()),
            },
        }
    }

    async fn import(
        &self,
        state: &AppState,
        app_handle: Option<&tauri::AppHandle>,
        config: &ConnectorConfig,
    ) -> SourceResult {
        let mut result = SourceResult {
            source: "obsidian".to_string(),
            ..Default::default()
        };

        let vault_path = match find_vault(config.path_override.as_deref()) {
            Some(p) => p,
            None => {
                tracing::info!("Bootstrap obsidian: no vault found");
                return result;
            }
        };

        let mut md_files = Vec::new();
        collect_md_files(&vault_path, &mut md_files, 10);

        let total = md_files.len() as u32;
        tracing::info!(
            "Bootstrap obsidian: found {} markdown files in {}",
            total,
            vault_path.display()
        );

        let base_importance = config.importance_override.unwrap_or(self.default_importance());

        for (i, file_path) in md_files.iter().enumerate() {
            result.items_scanned += 1;

            let content = match std::fs::read_to_string(file_path) {
                Ok(c) => c,
                Err(e) => {
                    tracing::debug!("Bootstrap obsidian: can't read {}: {}", file_path.display(), e);
                    result.errors += 1;
                    continue;
                }
            };

            let (body, tags) = strip_frontmatter(&content);
            if body.trim().len() < 20 {
                continue;
            }

            let filename = file_path
                .file_stem()
                .and_then(|s| s.to_str())
                .unwrap_or("untitled");

            let relative_path = file_path
                .strip_prefix(&vault_path)
                .unwrap_or(file_path)
                .to_string_lossy();

            let word_count = body.split_whitespace().count();

            let tags_label = if tags.is_empty() {
                String::new()
            } else {
                format!(" [{}]", tags)
            };

            if word_count > 500 {
                // Chunk long notes
                let chunks = chunker::chunk_text(&body, 400, 50);
                for (ci, chunk) in chunks.iter().take(5).enumerate() {
                    let memory_id = format!(
                        "bootstrap::obsidian::{}::{}",
                        deterministic_hash(&relative_path),
                        ci
                    );

                    let summary = if ci == 0 {
                        let preview: String = body.chars().take(200).collect();
                        format!("{}{}: {}", filename, tags_label, preview)
                    } else {
                        format!("{} (part {})", filename, ci + 1)
                    };

                    match store_as_memory_with_summary(
                        state,
                        &memory_id,
                        &summary,
                        chunk,
                        "semantic",
                        base_importance,
                    )
                    .await
                    {
                        Ok(true) => result.memories_created += 1,
                        Ok(false) => result.skipped_existing += 1,
                        Err(e) => {
                            tracing::debug!("Bootstrap obsidian chunk error: {}", e);
                            result.errors += 1;
                        }
                    }
                }
            } else {
                // Store whole note
                let memory_id = format!(
                    "bootstrap::obsidian::{}",
                    deterministic_hash(&relative_path)
                );

                let preview: String = body.chars().take(200).collect();
                let summary = format!("{}{}: {}", filename, tags_label, preview);

                match store_as_memory_with_summary(
                    state, &memory_id, &summary, &body, "semantic", base_importance,
                )
                .await
                {
                    Ok(true) => result.memories_created += 1,
                    Ok(false) => result.skipped_existing += 1,
                    Err(e) => {
                        tracing::debug!("Bootstrap obsidian error: {}", e);
                        result.errors += 1;
                    }
                }
            }

            if i % 20 == 0 {
                emit_progress(
                    app_handle,
                    &BootstrapProgress {
                        source: "obsidian".to_string(),
                        phase: "storing".to_string(),
                        current: i as u32 + 1,
                        total,
                        memories_created: result.memories_created,
                    },
                );
                tokio::task::yield_now().await;
            }
        }

        tracing::info!(
            "Bootstrap obsidian complete: {} created, {} skipped",
            result.memories_created,
            result.skipped_existing
        );
        result
    }
}
