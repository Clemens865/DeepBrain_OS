//! Knowledge Bootstrap System for DeepBrain
//!
//! Orchestrates imports from pluggable Connectors (see `connectors/` module).
//! This file retains shared helpers used by all connectors:
//! - `deterministic_hash()` for idempotent memory IDs
//! - `store_as_memory()` for embedding + graph integration
//! - `emit_progress()` for frontend progress events
//! - `copy_sqlite_to_temp()` for safe external DB reads
//! - `collect_files_with_ext()` / `collect_memory_md_files()` for recursive file scanning
//! - `parse_claude_jsonl()` shared JSONL parser for Claude conversations

use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use tauri::Emitter;

use crate::indexer::connectors::ConnectorRegistry;
use crate::state::AppState;

// ---- Types ----

/// Progress event emitted to the frontend during bootstrap.
#[derive(Clone, Serialize)]
pub struct BootstrapProgress {
    pub source: String,
    pub phase: String,
    pub current: u32,
    pub total: u32,
    pub memories_created: u32,
}

/// Result for a single source import.
#[derive(Clone, Debug, Serialize, Deserialize, Default)]
pub struct SourceResult {
    pub source: String,
    pub items_scanned: u32,
    pub memories_created: u32,
    pub skipped_existing: u32,
    pub errors: u32,
}

/// Overall bootstrap result.
#[derive(Clone, Serialize)]
pub struct BootstrapResult {
    pub total_memories_created: u32,
    pub total_skipped: u32,
    pub sources: Vec<SourceResult>,
    pub duration_secs: f64,
}

// ---- Document-type extensions for file index filter ----

pub const DOC_EXTENSIONS: &[&str] = &[
    "md", "pdf", "txt", "html", "json", "csv", "xlsx", "docx", "rtf", "org", "rst", "tex",
    "yaml", "yml", "xml", "plist", "adoc", "ods", "xls",
];

/// Core Data epoch offset (2001-01-01 00:00:00 UTC -> Unix epoch).
/// Used by WhatsApp connector.
pub const CORE_DATA_EPOCH_OFFSET: i64 = 978_307_200;

// ---- Orchestrator ----

/// Run the knowledge bootstrap for the selected sources.
///
/// `sources` can include: "file_index", "claude_code", "claude_memory",
/// "whatsapp", "browser", "deepnote", "claude_plans", "claude_desktop",
/// "claude_history", "all"
pub async fn bootstrap_knowledge(
    state: &AppState,
    app_handle: Option<&tauri::AppHandle>,
    sources: Vec<String>,
) -> Result<BootstrapResult, String> {
    let start = std::time::Instant::now();

    let registry = ConnectorRegistry::new();
    let user_config = state.settings.read().connector_config.clone();
    let results = registry
        .run_import(state, app_handle, &sources, &user_config)
        .await;

    let total_created: u32 = results.iter().map(|r| r.memories_created).sum();
    let total_skipped: u32 = results.iter().map(|r| r.skipped_existing).sum();

    // Flush persistence after all imports
    if total_created > 0 {
        let nodes = state.engine.memory.all_nodes();
        let _ = state.persistence.store_memories_batch(&nodes);
        tracing::info!(
            "Bootstrap complete: {} memories created, {} skipped (already exist)",
            total_created,
            total_skipped
        );
    }

    Ok(BootstrapResult {
        total_memories_created: total_created,
        total_skipped,
        sources: results,
        duration_secs: start.elapsed().as_secs_f64(),
    })
}

// ---- Shared Helpers ----

pub fn emit_progress(app_handle: Option<&tauri::AppHandle>, progress: &BootstrapProgress) {
    if let Some(handle) = app_handle {
        let _ = handle.emit("bootstrap-progress", progress);
    }
}

pub fn deterministic_hash(input: &str) -> String {
    let mut hasher = DefaultHasher::new();
    input.hash(&mut hasher);
    format!("{:016x}", hasher.finish())
}

/// Store a chunk as a brain memory with graph integration.
/// Returns true if newly created, false if skipped (existing).
pub async fn store_as_memory(
    state: &AppState,
    id: &str,
    content: &str,
    memory_type: &str,
    importance: f64,
) -> Result<bool, String> {
    // Skip empty content
    if content.trim().len() < 20 {
        return Ok(false);
    }

    // Check idempotency
    if state.engine.memory.has_memory(id) {
        return Ok(false);
    }

    // Embed
    let vector = state.embeddings.embed(content).await?;

    // Store with deterministic ID
    let created = state.engine.memory.store_f32_with_id(
        id.to_string(),
        content.to_string(),
        vector.clone(),
        memory_type.to_string(),
        importance,
    )?;

    if created {
        // Persist to SQLite
        if let Some(node) = state
            .engine
            .memory
            .all_nodes()
            .into_iter()
            .find(|n| n.id == id)
        {
            let _ = state.persistence.store_memory(&node);
        }

        // Add to knowledge graph
        let _ = state.graph.add_memory_node(id, memory_type, content);

        // Auto-connect to similar memories (top 5, threshold 0.5)
        let similar = state
            .engine
            .recall_f32(&vector, Some(5), None)
            .unwrap_or_default();
        let related_ids: Vec<String> = similar
            .iter()
            .filter(|s| s.id != id && s.similarity > 0.5)
            .map(|s| s.id.clone())
            .collect();
        if !related_ids.is_empty() {
            state.graph.auto_connect(id, &related_ids, 0.6);
        }
    }

    Ok(created)
}

/// Copy an external SQLite file to a temp location to avoid WAL lock conflicts.
pub fn copy_sqlite_to_temp(source: &Path, label: &str) -> Result<PathBuf, String> {
    let temp_dir = std::env::temp_dir().join("deepbrain_bootstrap");
    std::fs::create_dir_all(&temp_dir)
        .map_err(|e| format!("Failed to create temp dir: {}", e))?;

    let dest = temp_dir.join(format!("{}.db", label));
    std::fs::copy(source, &dest)
        .map_err(|e| format!("Failed to copy {} to temp: {}", source.display(), e))?;

    // Also copy WAL and SHM files if they exist
    let wal = source.with_extension("db-wal");
    if wal.exists() {
        let _ = std::fs::copy(&wal, dest.with_extension("db-wal"));
    }
    let shm = source.with_extension("db-shm");
    if shm.exists() {
        let _ = std::fs::copy(&shm, dest.with_extension("db-shm"));
    }

    Ok(dest)
}

// ---- Shared JSONL Parser ----

/// Parse a Claude JSONL conversation file into a single text string.
/// Shared by `claude_code` and `claude_desktop` connectors.
pub fn parse_claude_jsonl(content: &str) -> String {
    let mut conversation_text = String::new();
    for line in content.lines() {
        if line.trim().is_empty() {
            continue;
        }
        if let Ok(json) = serde_json::from_str::<serde_json::Value>(line) {
            let role = json
                .get("message")
                .and_then(|m| m.get("role"))
                .and_then(|r| r.as_str())
                .or_else(|| json.get("role").and_then(|r| r.as_str()));

            let role_prefix = match role {
                Some("user") => "User: ",
                Some("assistant") => "Assistant: ",
                _ => continue,
            };

            let msg_obj = json.get("message").unwrap_or(&json);
            if let Some(content_val) = msg_obj.get("content") {
                if let Some(text) = content_val.as_str() {
                    if !text.is_empty() {
                        conversation_text.push_str(role_prefix);
                        conversation_text.push_str(text);
                        conversation_text.push_str("\n\n");
                    }
                } else if let Some(arr) = content_val.as_array() {
                    for item in arr {
                        if let Some(text) = item.get("text").and_then(|t| t.as_str()) {
                            if !text.is_empty() {
                                conversation_text.push_str(role_prefix);
                                conversation_text.push_str(text);
                                conversation_text.push_str("\n\n");
                            }
                        }
                    }
                }
            }
        }
    }
    conversation_text
}

// ---- File collection helpers ----

/// Recursively collect files with a specific extension.
pub fn collect_files_with_ext(dir: &Path, ext: &str, files: &mut Vec<PathBuf>, max_depth: u32) {
    if max_depth == 0 {
        return;
    }

    let entries = match std::fs::read_dir(dir) {
        Ok(e) => e,
        Err(_) => return,
    };

    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            collect_files_with_ext(&path, ext, files, max_depth - 1);
        } else if path.extension().and_then(|e| e.to_str()) == Some(ext) {
            files.push(path);
        }
    }
}

/// Find MEMORY.md and other .md files in memory/ subdirectories.
pub fn collect_memory_md_files(dir: &Path, files: &mut Vec<PathBuf>, max_depth: u32) {
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

        if path.is_dir() {
            if name_str == "memory" {
                // Collect all .md files inside memory/ directories
                collect_files_with_ext(&path, "md", files, 1);
            } else {
                collect_memory_md_files(&path, files, max_depth - 1);
            }
        } else if name_str == "MEMORY.md" {
            files.push(path);
        }
    }
}
