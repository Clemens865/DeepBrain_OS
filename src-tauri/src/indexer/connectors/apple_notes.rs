//! Apple Notes connector — imports notes via AppleScript.
//!
//! Uses `osascript` to read notes from the Notes app, avoiding direct SQLite
//! access to the NoteStore which requires TCC entitlements. AppleScript's `log`
//! command writes to stderr, so we parse structured output from there.

use std::process::Command;

use crate::indexer::bootstrap::{
    deterministic_hash, emit_progress, store_as_memory_with_summary, BootstrapProgress,
    SourceResult,
};
use crate::indexer::chunker;
use crate::state::AppState;

use super::{Connector, ConnectorConfig, DetectionResult};

pub struct AppleNotesConnector;

/// A parsed note from AppleScript output.
struct ParsedNote {
    note_id: String,
    title: String,
    folder: String,
    body: String,
}

/// Parse the structured AppleScript output from stderr.
fn parse_notes_output(stderr: &str) -> Vec<ParsedNote> {
    let mut notes = Vec::new();
    let mut current_id = String::new();
    let mut current_title = String::new();
    let mut current_folder = String::new();
    let mut current_body = String::new();
    let mut in_body = false;

    for line in stderr.lines() {
        if line.contains("<<<NOTE_START>>>") {
            current_id.clear();
            current_title.clear();
            current_folder.clear();
            current_body.clear();
            in_body = false;
            continue;
        }
        if line.contains("<<<NOTE_END>>>") {
            if !current_title.is_empty() && !current_body.trim().is_empty() {
                notes.push(ParsedNote {
                    note_id: current_id.clone(),
                    title: current_title.clone(),
                    folder: current_folder.clone(),
                    body: current_body.trim().to_string(),
                });
            }
            in_body = false;
            continue;
        }

        if in_body {
            if !current_body.is_empty() {
                current_body.push('\n');
            }
            current_body.push_str(line);
            continue;
        }

        if let Some(rest) = line.strip_prefix("ID:") {
            current_id = rest.trim().to_string();
        } else if let Some(rest) = line.strip_prefix("TITLE:") {
            current_title = rest.trim().to_string();
        } else if let Some(rest) = line.strip_prefix("FOLDER:") {
            current_folder = rest.trim().to_string();
        } else if let Some(rest) = line.strip_prefix("BODY:") {
            current_body = rest.trim().to_string();
            in_body = true;
        }
    }

    notes
}

#[async_trait::async_trait]
impl Connector for AppleNotesConnector {
    fn id(&self) -> &str {
        "apple_notes"
    }
    fn name(&self) -> &str {
        "Apple Notes"
    }
    fn description(&self) -> &str {
        "Notes from the Apple Notes app"
    }
    fn icon(&self) -> &str {
        "sticky-note"
    }
    fn default_enabled(&self) -> bool {
        true
    }
    fn memory_type(&self) -> &str {
        "semantic"
    }
    fn default_importance(&self) -> f64 {
        0.6
    }

    fn detect(&self) -> DetectionResult {
        // Quick check: can we talk to Notes.app?
        let output = Command::new("osascript")
            .arg("-e")
            .arg("tell application \"Notes\" to count of notes")
            .output();

        match output {
            Ok(out) => {
                let stdout = String::from_utf8_lossy(&out.stdout).trim().to_string();
                if let Ok(count) = stdout.parse::<u64>() {
                    DetectionResult {
                        available: count > 0,
                        path: None,
                        details: Some(format!("{} notes", count)),
                    }
                } else {
                    DetectionResult {
                        available: false,
                        path: None,
                        details: Some("Could not query Notes app".to_string()),
                    }
                }
            }
            Err(_) => DetectionResult {
                available: false,
                path: None,
                details: None,
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
            source: "apple_notes".to_string(),
            ..Default::default()
        };

        let script = r#"
tell application "Notes"
    repeat with n in notes
        log "<<<NOTE_START>>>"
        log "ID:" & (id of n)
        log "TITLE:" & (name of n)
        log "FOLDER:" & (name of container of n)
        log "BODY:" & (plaintext of n)
        log "<<<NOTE_END>>>"
    end repeat
end tell
"#;

        emit_progress(
            app_handle,
            &BootstrapProgress {
                source: "apple_notes".to_string(),
                phase: "reading notes via AppleScript".to_string(),
                current: 0,
                total: 0,
                memories_created: 0,
            },
        );

        let output = match Command::new("osascript").arg("-e").arg(script).output() {
            Ok(out) => out,
            Err(e) => {
                tracing::info!("Bootstrap apple_notes: osascript failed: {}", e);
                return result;
            }
        };

        // AppleScript `log` writes to stderr
        let stderr = String::from_utf8_lossy(&output.stderr);
        let notes = parse_notes_output(&stderr);

        let total = notes.len() as u32;
        tracing::info!("Bootstrap apple_notes: parsed {} notes", total);

        let base_importance = config.importance_override.unwrap_or(self.default_importance());

        for (i, note) in notes.iter().enumerate() {
            result.items_scanned += 1;

            let word_count = note.body.split_whitespace().count();

            if word_count > 500 {
                // Chunk long notes
                let chunks = chunker::chunk_text(&note.body, 400, 50);
                for (ci, chunk) in chunks.iter().take(5).enumerate() {
                    let memory_id = format!(
                        "bootstrap::apple_notes::{}::{}",
                        deterministic_hash(&note.note_id),
                        ci
                    );

                    let summary = if ci == 0 {
                        let preview: String = note.body.chars().take(200).collect();
                        format!("{} — {}: {}", note.title, note.folder, preview)
                    } else {
                        format!("{} (part {})", note.title, ci + 1)
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
                            tracing::debug!("Bootstrap apple_notes chunk error: {}", e);
                            result.errors += 1;
                        }
                    }
                }
            } else {
                // Store whole note
                let memory_id = format!(
                    "bootstrap::apple_notes::{}",
                    deterministic_hash(&note.note_id)
                );

                let preview: String = note.body.chars().take(200).collect();
                let summary = format!("{} — {}: {}", note.title, note.folder, preview);

                match store_as_memory_with_summary(
                    state,
                    &memory_id,
                    &summary,
                    &note.body,
                    "semantic",
                    base_importance,
                )
                .await
                {
                    Ok(true) => result.memories_created += 1,
                    Ok(false) => result.skipped_existing += 1,
                    Err(e) => {
                        tracing::debug!("Bootstrap apple_notes error: {}", e);
                        result.errors += 1;
                    }
                }
            }

            if i % 10 == 0 {
                emit_progress(
                    app_handle,
                    &BootstrapProgress {
                        source: "apple_notes".to_string(),
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
            "Bootstrap apple_notes complete: {} created, {} skipped",
            result.memories_created,
            result.skipped_existing
        );
        result
    }
}
