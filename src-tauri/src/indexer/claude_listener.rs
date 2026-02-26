//! Claude conversation listener for DeepBrain
//!
//! Watches Claude Code projects and Claude Desktop agent-mode sessions
//! for new or modified JSONL conversation files. When a change is detected,
//! the conversation is parsed and stored as brain memories in real time.
//!
//! This runs as a background task alongside the file watcher.

use std::collections::HashSet;
use std::path::PathBuf;
use std::sync::Arc;

use notify::{EventKind, RecommendedWatcher, RecursiveMode, Watcher};
use tokio::sync::mpsc;

use crate::indexer::bootstrap::{self, BootstrapProgress};
use crate::state::AppState;

/// Start a background watcher that monitors Claude conversation directories
/// and auto-imports new conversations into DeepBrain's memory.
///
/// Returns a handle to keep the watcher alive (drop it to stop watching).
pub fn start_claude_listener(
    state: Arc<AppState>,
    app_handle: tauri::AppHandle,
) -> Result<RecommendedWatcher, String> {
    let (tx, rx) = mpsc::unbounded_channel::<PathBuf>();

    let mut watcher = notify::recommended_watcher(move |res: Result<notify::Event, notify::Error>| {
        if let Ok(event) = res {
            match event.kind {
                EventKind::Create(_) | EventKind::Modify(_) => {}
                _ => return,
            }

            for path in event.paths {
                // Only care about .jsonl files
                if path.extension().and_then(|e| e.to_str()) == Some("jsonl") {
                    let _ = tx.send(path);
                }
            }
        }
    })
    .map_err(|e| format!("Failed to create Claude listener: {}", e))?;

    // Watch directories
    let home = dirs::home_dir().ok_or("no home dir")?;

    // 1. Claude Code projects (JSONL conversations)
    let claude_projects = home.join(".claude").join("projects");
    if claude_projects.exists() {
        watcher
            .watch(&claude_projects, RecursiveMode::Recursive)
            .map_err(|e| format!("Failed to watch Claude projects: {}", e))?;
        tracing::info!("Claude listener: watching {:?}", claude_projects);
    }

    // 2. Claude Desktop agent-mode sessions
    let desktop_sessions = home
        .join("Library")
        .join("Application Support")
        .join("Claude")
        .join("local-agent-mode-sessions");
    if desktop_sessions.exists() {
        watcher
            .watch(&desktop_sessions, RecursiveMode::Recursive)
            .map_err(|e| format!("Failed to watch Claude Desktop sessions: {}", e))?;
        tracing::info!("Claude listener: watching {:?}", desktop_sessions);
    }

    // 3. Claude history.jsonl
    let history_file = home.join(".claude").join("history.jsonl");
    if history_file.exists() {
        // Watch the parent directory for changes to history.jsonl
        watcher
            .watch(&home.join(".claude"), RecursiveMode::NonRecursive)
            .map_err(|e| format!("Failed to watch .claude dir: {}", e))?;
        tracing::info!("Claude listener: watching history.jsonl");
    }

    // Spawn the processing loop
    tauri::async_runtime::spawn(process_claude_changes(state, app_handle, rx));

    Ok(watcher)
}

/// Background loop that processes changed JSONL files with debouncing.
async fn process_claude_changes(
    state: Arc<AppState>,
    app_handle: tauri::AppHandle,
    mut rx: mpsc::UnboundedReceiver<PathBuf>,
) {
    let debounce = tokio::time::Duration::from_secs(5);
    let mut pending = HashSet::<PathBuf>::new();

    loop {
        // Wait for the first event
        let path = match rx.recv().await {
            Some(p) => p,
            None => break,
        };
        pending.insert(path);

        // Drain events within the debounce window
        loop {
            match tokio::time::timeout(debounce, rx.recv()).await {
                Ok(Some(p)) => {
                    pending.insert(p);
                }
                _ => break,
            }
        }

        // Process all changed files
        let batch: Vec<_> = pending.drain().collect();
        tracing::info!(
            "Claude listener: processing {} changed conversation files",
            batch.len()
        );

        let mut total_created = 0u32;

        for path in &batch {
            let path_str = path.to_string_lossy();

            // Determine the source type based on path
            let (source_type, memory_type, importance) =
                if path_str.contains("local-agent-mode-sessions") {
                    ("claude_desktop", "conversation", 0.65)
                } else if path_str.contains("history.jsonl") {
                    // Skip history.jsonl here — it's append-only and better
                    // handled by a full re-import (bootstrap). Individual
                    // line-level watching would need offset tracking.
                    continue;
                } else {
                    ("claude_code", "conversation", 0.6)
                };

            let created = import_single_conversation(
                &state,
                path,
                source_type,
                memory_type,
                importance,
            )
            .await;

            total_created += created;
        }

        if total_created > 0 {
            // Persist new memories
            let nodes = state.engine.memory.all_nodes();
            let _ = state.persistence.store_memories_batch(&nodes);

            // Emit progress event
            let _ = tauri::Emitter::emit(
                &app_handle,
                "bootstrap-progress",
                &BootstrapProgress {
                    source: "claude_listener".to_string(),
                    phase: "complete".to_string(),
                    current: total_created,
                    total: total_created,
                    memories_created: total_created,
                },
            );

            tracing::info!(
                "Claude listener: stored {} new memories from {} files",
                total_created,
                batch.len()
            );
        }
    }
}

/// Import a single JSONL conversation file into brain memories.
/// Returns the number of memories created.
async fn import_single_conversation(
    state: &AppState,
    file_path: &PathBuf,
    source_type: &str,
    memory_type: &str,
    importance: f64,
) -> u32 {
    let content = match std::fs::read_to_string(file_path) {
        Ok(c) => c,
        Err(e) => {
            tracing::debug!("Claude listener: read error for {:?}: {}", file_path, e);
            return 0;
        }
    };

    let file_hash = bootstrap::deterministic_hash(&file_path.to_string_lossy());

    // Parse JSONL: extract both user and assistant messages for full conversation context
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

    if conversation_text.trim().len() < 50 {
        return 0;
    }

    // Chunk into segments
    let segments = crate::indexer::chunker::chunk_text(&conversation_text, 400, 50);
    let mut created = 0u32;

    for (si, segment) in segments.iter().enumerate().take(15) {
        let seg_id = format!("bootstrap::{}::{}::{}", source_type, file_hash, si);

        match bootstrap::store_as_memory(state, &seg_id, segment, memory_type, importance).await {
            Ok(true) => created += 1,
            Ok(false) => {} // Already exists
            Err(e) => {
                tracing::debug!("Claude listener error: {}", e);
            }
        }
    }

    if created > 0 {
        tracing::debug!(
            "Claude listener: {} new memories from {:?}",
            created,
            file_path.file_name().unwrap_or_default()
        );
    }

    created
}
