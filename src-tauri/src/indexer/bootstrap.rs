//! Knowledge Bootstrap System for DeepBrain
//!
//! Populates the brain's memory and knowledge graph from personal data sources:
//! - File index documents (md, pdf, txt, html, etc.)
//! - Claude Code conversations and memory files
//! - WhatsApp messages
//! - Comet browser history
//! - Deepnote AI data
//!
//! Design:
//! - Deterministic IDs for idempotency (running twice won't duplicate)
//! - Copy-then-read for external SQLite files (avoid lock conflicts)
//! - Batch embedding + store with periodic RVF flush
//! - Progress events emitted to frontend via Tauri events

use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::path::{Path, PathBuf};

use rusqlite::Connection;
use serde::{Deserialize, Serialize};

use tauri::Emitter;

use crate::indexer::chunker;
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
    let run_all = sources.contains(&"all".to_string());

    let mut results = Vec::new();

    if run_all || sources.contains(&"file_index".to_string()) {
        let r = import_file_index_docs(state, app_handle).await;
        results.push(r);
    }

    if run_all || sources.contains(&"claude_code".to_string()) {
        let r = import_claude_code(state, app_handle).await;
        results.push(r);
    }

    if run_all || sources.contains(&"claude_memory".to_string()) {
        let r = import_claude_memory(state, app_handle).await;
        results.push(r);
    }

    if run_all || sources.contains(&"whatsapp".to_string()) {
        let r = import_whatsapp(state, app_handle).await;
        results.push(r);
    }

    if run_all || sources.contains(&"browser".to_string()) {
        let r = import_browser_history(state, app_handle).await;
        results.push(r);
    }

    if run_all || sources.contains(&"deepnote".to_string()) {
        let r = import_deepnote(state, app_handle).await;
        results.push(r);
    }

    if run_all || sources.contains(&"claude_plans".to_string()) {
        let r = import_claude_plans(state, app_handle).await;
        results.push(r);
    }

    if run_all || sources.contains(&"claude_desktop".to_string()) {
        let r = import_claude_desktop(state, app_handle).await;
        results.push(r);
    }

    if run_all || sources.contains(&"claude_history".to_string()) {
        let r = import_claude_history(state, app_handle).await;
        results.push(r);
    }

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

// ---- Helpers ----

fn emit_progress(app_handle: Option<&tauri::AppHandle>, progress: &BootstrapProgress) {
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

// ---- Document-type extensions for file index filter ----

const DOC_EXTENSIONS: &[&str] = &[
    "md", "pdf", "txt", "html", "json", "csv", "xlsx", "docx", "rtf", "org", "rst", "tex",
    "yaml", "yml", "xml", "plist", "adoc", "ods", "xls",
];

fn is_doc_extension(ext: &str) -> bool {
    DOC_EXTENSIONS.contains(&ext.to_lowercase().as_str())
}

// ---- Synchronous SQLite readers (Connection is !Send) ----

/// Read document rows from files.db synchronously.
fn read_file_index_docs() -> Result<Vec<(String, String, String, String)>, String> {
    let db_path = dirs::data_dir()
        .ok_or("no data dir")?
        .join("DeepBrain")
        .join("files.db");
    if !db_path.exists() {
        return Err("files.db not found".to_string());
    }
    let conn = Connection::open(&db_path).map_err(|e| format!("open files.db: {}", e))?;
    let ext_placeholders: Vec<String> = DOC_EXTENSIONS.iter().map(|_| "?".to_string()).collect();
    let sql = format!(
        "SELECT fi.path, fi.name, fi.ext, GROUP_CONCAT(fc.content, ' ') as full_content
         FROM file_index fi
         JOIN file_chunks fc ON fc.file_path = fi.path
         WHERE fi.ext IN ({})
         GROUP BY fi.path
         ORDER BY fi.modified DESC",
        ext_placeholders.join(",")
    );
    let mut stmt = conn.prepare(&sql).map_err(|e| format!("prepare: {}", e))?;
    let params_vec: Vec<Box<dyn rusqlite::types::ToSql>> = DOC_EXTENSIONS
        .iter()
        .map(|e| Box::new(e.to_string()) as Box<dyn rusqlite::types::ToSql>)
        .collect();
    let params_refs: Vec<&dyn rusqlite::types::ToSql> =
        params_vec.iter().map(|p| p.as_ref()).collect();
    let rows = stmt
        .query_map(params_refs.as_slice(), |row| {
            Ok((
                row.get::<_, String>(0)?,
                row.get::<_, String>(1)?,
                row.get::<_, String>(2)?,
                row.get::<_, String>(3)?,
            ))
        })
        .map_err(|e| format!("query: {}", e))?
        .filter_map(|r| r.ok())
        .collect();
    Ok(rows)
}

/// A single WhatsApp message with sender context and timestamp.
struct WhatsAppMsg {
    text: String,
    is_from_me: bool,
    timestamp_unix: i64,
    sender_name: String,
    chat_jid: String,
    chat_name: String,
}

/// Core Data epoch offset (2001-01-01 00:00:00 UTC → Unix epoch).
const CORE_DATA_EPOCH_OFFSET: i64 = 978_307_200;

/// Read WhatsApp messages from ChatStorage.sqlite with sender info and timestamps.
fn read_whatsapp_messages() -> Result<(Vec<WhatsAppMsg>, PathBuf), String> {
    let home = dirs::home_dir().ok_or("no home dir")?;
    let wa_db = home.join("Library/Group Containers/group.net.whatsapp.WhatsApp.shared/ChatStorage.sqlite");
    if !wa_db.exists() {
        return Err("WhatsApp ChatStorage.sqlite not found".to_string());
    }
    let temp_db = copy_sqlite_to_temp(&wa_db, "whatsapp")?;
    let conn = Connection::open_with_flags(
        &temp_db,
        rusqlite::OpenFlags::SQLITE_OPEN_READ_ONLY | rusqlite::OpenFlags::SQLITE_OPEN_NO_MUTEX,
    )
    .map_err(|e| format!("open WhatsApp copy: {}", e))?;

    let sql = "SELECT
            m.ZTEXT,
            m.ZISFROMME,
            m.ZMESSAGEDATE,
            COALESCE(m.ZPUSHNAME, cs.ZPARTNERNAME, cs.ZCONTACTJID) AS sender_name,
            cs.ZCONTACTJID,
            COALESCE(cs.ZPARTNERNAME, cs.ZCONTACTJID) AS chat_name
        FROM ZWAMESSAGE m
        JOIN ZWACHATSESSION cs ON m.ZCHATSESSION = cs.Z_PK
        WHERE m.ZTEXT IS NOT NULL AND length(m.ZTEXT) > 20
        ORDER BY cs.ZCONTACTJID, m.ZMESSAGEDATE ASC";

    let mut stmt = conn.prepare(sql).map_err(|e| format!("prepare: {}", e))?;
    let messages: Vec<WhatsAppMsg> = stmt
        .query_map([], |row| {
            let text: String = row.get(0)?;
            let is_from_me: bool = row.get::<_, i32>(1).unwrap_or(0) == 1;
            let core_data_ts: f64 = row.get::<_, f64>(2).unwrap_or(0.0);
            let sender_name: String = row.get::<_, Option<String>>(3)?.unwrap_or_default();
            let chat_jid: String = row.get::<_, Option<String>>(4)?.unwrap_or_else(|| "unknown".to_string());
            let chat_name: String = row.get::<_, Option<String>>(5)?.unwrap_or_else(|| chat_jid.clone());

            let timestamp_unix = core_data_ts as i64 + CORE_DATA_EPOCH_OFFSET;

            Ok(WhatsAppMsg {
                text,
                is_from_me,
                timestamp_unix,
                sender_name: if is_from_me { "You".to_string() } else { sender_name },
                chat_jid,
                chat_name,
            })
        })
        .map_err(|e| format!("query: {}", e))?
        .filter_map(|r| r.ok())
        .collect();

    Ok((messages, temp_db))
}

/// Read browser history from Comet History DB.
fn read_browser_history() -> Result<(Vec<(String, String, u32)>, PathBuf), String> {
    let home = dirs::home_dir().ok_or("no home dir")?;
    let history_db = home.join("Library/Application Support/Comet/Default/History");
    if !history_db.exists() {
        return Err("Comet browser History not found".to_string());
    }
    let temp_db = copy_sqlite_to_temp(&history_db, "comet_history")?;
    let conn = Connection::open_with_flags(
        &temp_db,
        rusqlite::OpenFlags::SQLITE_OPEN_READ_ONLY | rusqlite::OpenFlags::SQLITE_OPEN_NO_MUTEX,
    )
    .map_err(|e| format!("open Comet copy: {}", e))?;
    let sql = "SELECT url, title, visit_count, last_visit_time
               FROM urls
               WHERE visit_count > 1 AND title IS NOT NULL AND title != ''
               ORDER BY visit_count DESC
               LIMIT 2000";
    let mut stmt = conn.prepare(sql).map_err(|e| format!("prepare: {}", e))?;
    let rows = stmt
        .query_map([], |row| {
            Ok((
                row.get::<_, String>(0)?,
                row.get::<_, String>(1)?,
                row.get::<_, u32>(2)?,
            ))
        })
        .map_err(|e| format!("query: {}", e))?
        .filter_map(|r| r.ok())
        .collect();
    Ok((rows, temp_db))
}

/// Read Deepnote AI data with schema discovery.
fn read_deepnote_data() -> Result<(Vec<String>, PathBuf), String> {
    let home = dirs::home_dir().ok_or("no home dir")?;
    let deepnote_db = home.join("Library/Application Support/deepnote-ai/deepnote-ai.db");
    if !deepnote_db.exists() {
        return Err("Deepnote AI db not found".to_string());
    }
    let temp_db = copy_sqlite_to_temp(&deepnote_db, "deepnote")?;
    let conn = Connection::open_with_flags(
        &temp_db,
        rusqlite::OpenFlags::SQLITE_OPEN_READ_ONLY | rusqlite::OpenFlags::SQLITE_OPEN_NO_MUTEX,
    )
    .map_err(|e| format!("open Deepnote copy: {}", e))?;
    let tables: Vec<String> = conn
        .prepare("SELECT name FROM sqlite_master WHERE type='table'")
        .ok()
        .and_then(|mut stmt| {
            stmt.query_map([], |row| row.get::<_, String>(0))
                .ok()
                .map(|iter| iter.filter_map(|r| r.ok()).collect())
        })
        .unwrap_or_default();
    tracing::info!("Bootstrap deepnote: tables found: {:?}", tables);
    let chat_tables = ["chat_messages", "messages", "conversations", "chat"];
    for table in &chat_tables {
        if !tables.iter().any(|t| t == *table) {
            continue;
        }
        let sql = format!(
            "SELECT content FROM {} WHERE content IS NOT NULL AND length(content) > 20 LIMIT 1000",
            table
        );
        let mut stmt = match conn.prepare(&sql) {
            Ok(s) => s,
            Err(_) => {
                let sql2 = format!(
                    "SELECT text FROM {} WHERE text IS NOT NULL AND length(text) > 20 LIMIT 1000",
                    table
                );
                match conn.prepare(&sql2) {
                    Ok(s) => s,
                    Err(_) => continue,
                }
            }
        };
        let rows = stmt
            .query_map([], |row| row.get::<_, String>(0))
            .ok()
            .map(|iter| iter.filter_map(|r| r.ok()).collect())
            .unwrap_or_default();
        return Ok((rows, temp_db));
    }
    Ok((Vec::new(), temp_db))
}

// ========================================================================
// Source: File Index Documents
// ========================================================================

async fn import_file_index_docs(
    state: &AppState,
    app_handle: Option<&tauri::AppHandle>,
) -> SourceResult {
    let mut result = SourceResult {
        source: "file_index".to_string(),
        ..Default::default()
    };

    // Read file index rows synchronously, collecting into owned data so the
    // rusqlite Connection (!Send) is dropped before any .await points.
    let rows: Vec<(String, String, String, String)> =
        match read_file_index_docs() {
            Ok(r) => r,
            Err(e) => {
                tracing::info!("Bootstrap file_index: {}", e);
                return result;
            }
        };

    let total = rows.len() as u32;
    tracing::info!("Bootstrap file_index: found {} document files", total);

    for (i, (path, name, ext, content)) in rows.iter().enumerate() {
        result.items_scanned += 1;

        // Deterministic ID
        let path_hash = deterministic_hash(path);
        let memory_id = format!("bootstrap::file::{}", path_hash);

        // Skip if already exists
        if state.engine.memory.has_memory(&memory_id) {
            result.skipped_existing += 1;
            continue;
        }

        // Truncate very large documents to ~4000 words
        let truncated: String = content
            .split_whitespace()
            .take(4000)
            .collect::<Vec<&str>>()
            .join(" ");

        // Prefix with file info for context
        let enriched = format!("[Document: {} ({})] {}", name, path, truncated);

        // Chunk and store
        let chunks = chunker::chunk_text(&enriched, 512, 128);
        if chunks.is_empty() {
            continue;
        }

        // For documents, store the first chunk as the main memory
        // (embedding the whole concatenation would lose specificity)
        let importance = match ext.as_str() {
            "md" | "txt" | "rtf" | "org" | "rst" | "adoc" => 0.5,
            "pdf" | "docx" | "xlsx" | "xls" => 0.55,
            _ => 0.45,
        };

        // Store each chunk as a separate memory for better retrieval
        for (ci, chunk) in chunks.iter().enumerate().take(5) {
            // Max 5 chunks per file
            let chunk_id = format!("{}::{}", memory_id, ci);
            match store_as_memory(state, &chunk_id, chunk, "semantic", importance).await {
                Ok(true) => result.memories_created += 1,
                Ok(false) => result.skipped_existing += 1,
                Err(e) => {
                    tracing::debug!("Bootstrap file_index error for {}: {}", name, e);
                    result.errors += 1;
                }
            }
        }

        emit_progress(
            app_handle,
            &BootstrapProgress {
                source: "file_index".to_string(),
                phase: "storing".to_string(),
                current: i as u32 + 1,
                total,
                memories_created: result.memories_created,
            },
        );

        // Yield to async runtime periodically
        if i % 10 == 9 {
            tokio::task::yield_now().await;
        }
    }

    tracing::info!(
        "Bootstrap file_index complete: {} created, {} skipped",
        result.memories_created,
        result.skipped_existing
    );
    result
}

// ========================================================================
// Source: Claude Code Conversations
// ========================================================================

async fn import_claude_code(
    state: &AppState,
    app_handle: Option<&tauri::AppHandle>,
) -> SourceResult {
    let mut result = SourceResult {
        source: "claude_code".to_string(),
        ..Default::default()
    };

    let home = match dirs::home_dir() {
        Some(h) => h,
        None => return result,
    };

    let claude_dir = home.join(".claude").join("projects");
    if !claude_dir.exists() {
        tracing::info!("Bootstrap: ~/.claude/projects not found, skipping claude_code");
        return result;
    }

    // Find all .jsonl files recursively
    let mut jsonl_files = Vec::new();
    collect_files_with_ext(&claude_dir, "jsonl", &mut jsonl_files, 3);

    let total = jsonl_files.len() as u32;
    tracing::info!("Bootstrap claude_code: found {} conversation files", total);

    for (i, file_path) in jsonl_files.iter().enumerate() {
        result.items_scanned += 1;

        let content = match std::fs::read_to_string(file_path) {
            Ok(c) => c,
            Err(_) => continue,
        };

        let file_hash = deterministic_hash(&file_path.to_string_lossy());

        // Parse JSONL: extract both user and assistant messages
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
            continue;
        }

        // Chunk into conversation segments (~2000 chars each)
        let segments = chunker::chunk_text(&conversation_text, 400, 50);

        for (si, segment) in segments.iter().enumerate().take(10) {
            let seg_id = format!("bootstrap::claude::{}::{}", file_hash, si);

            match store_as_memory(state, &seg_id, segment, "episodic", 0.6).await {
                Ok(true) => result.memories_created += 1,
                Ok(false) => result.skipped_existing += 1,
                Err(e) => {
                    tracing::debug!("Bootstrap claude_code error: {}", e);
                    result.errors += 1;
                }
            }
        }

        emit_progress(
            app_handle,
            &BootstrapProgress {
                source: "claude_code".to_string(),
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
        "Bootstrap claude_code complete: {} created, {} skipped",
        result.memories_created,
        result.skipped_existing
    );
    result
}

// ========================================================================
// Source: Claude Code Memory Files
// ========================================================================

async fn import_claude_memory(
    state: &AppState,
    app_handle: Option<&tauri::AppHandle>,
) -> SourceResult {
    let mut result = SourceResult {
        source: "claude_memory".to_string(),
        ..Default::default()
    };

    let home = match dirs::home_dir() {
        Some(h) => h,
        None => return result,
    };

    let claude_dir = home.join(".claude").join("projects");
    if !claude_dir.exists() {
        return result;
    }

    // Find all MEMORY.md and other .md files in memory/ subdirectories
    let mut memory_files = Vec::new();
    collect_memory_md_files(&claude_dir, &mut memory_files, 4);

    // Also check top-level MEMORY.md
    let top_memory = home.join(".claude").join("MEMORY.md");
    if top_memory.exists() {
        memory_files.push(top_memory);
    }

    let total = memory_files.len() as u32;
    tracing::info!("Bootstrap claude_memory: found {} memory files", total);

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

        // Store as high-importance knowledge
        let enriched = format!(
            "[Claude Memory: {}]\n{}",
            file_path.file_name().unwrap_or_default().to_string_lossy(),
            content
        );

        let chunks = chunker::chunk_text(&enriched, 512, 128);
        for (ci, chunk) in chunks.iter().enumerate() {
            let chunk_id = format!("{}::{}", memory_id, ci);
            match store_as_memory(state, &chunk_id, chunk, "semantic", 0.8).await {
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

// ========================================================================
// Source: WhatsApp Messages
// ========================================================================

async fn import_whatsapp(
    state: &AppState,
    app_handle: Option<&tauri::AppHandle>,
) -> SourceResult {
    let mut result = SourceResult {
        source: "whatsapp".to_string(),
        ..Default::default()
    };

    let (messages, temp_db) = match read_whatsapp_messages() {
        Ok(r) => r,
        Err(e) => {
            tracing::info!("Bootstrap whatsapp: {}", e);
            return result;
        }
    };

    // Group messages by chat_jid (preserving order from SQL)
    let mut chats: std::collections::HashMap<String, Vec<&WhatsAppMsg>> =
        std::collections::HashMap::new();
    for msg in &messages {
        chats.entry(msg.chat_jid.clone()).or_default().push(msg);
    }

    let total = chats.len() as u32;
    tracing::info!(
        "Bootstrap whatsapp: {} messages in {} chats",
        messages.len(),
        total
    );

    let mut chat_idx = 0u32;
    for (chat_jid, msgs) in &chats {
        result.items_scanned += msgs.len() as u32;

        // Format conversation with timestamps and sender names
        let chat_name = msgs.first().map(|m| m.chat_name.as_str()).unwrap_or(chat_jid);
        let mut lines = Vec::with_capacity(msgs.len() + 2);
        lines.push(format!("Conversation with {}:\n", chat_name));

        for msg in msgs {
            let dt = chrono::DateTime::from_timestamp(msg.timestamp_unix, 0)
                .map(|d| d.format("%Y-%m-%d %H:%M").to_string())
                .unwrap_or_else(|| "unknown".to_string());
            lines.push(format!("[{}] {}: {}", dt, msg.sender_name, msg.text));
        }

        let combined = lines.join("\n");
        let windows = chunker::chunk_text(&combined, 200, 30);
        let chat_hash = deterministic_hash(chat_jid);

        for (wi, window) in windows.iter().enumerate().take(20) {
            let win_id = format!("bootstrap::whatsapp::{}::{}", chat_hash, wi);

            match store_as_memory(state, &win_id, window, "episodic", 0.4).await {
                Ok(true) => result.memories_created += 1,
                Ok(false) => result.skipped_existing += 1,
                Err(e) => {
                    tracing::debug!("Bootstrap whatsapp error: {}", e);
                    result.errors += 1;
                }
            }
        }

        chat_idx += 1;
        emit_progress(
            app_handle,
            &BootstrapProgress {
                source: "whatsapp".to_string(),
                phase: "storing".to_string(),
                current: chat_idx,
                total,
                memories_created: result.memories_created,
            },
        );

        if chat_idx % 5 == 0 {
            tokio::task::yield_now().await;
        }
    }

    // Clean up temp file
    let _ = std::fs::remove_file(&temp_db);

    tracing::info!(
        "Bootstrap whatsapp complete: {} created, {} skipped",
        result.memories_created,
        result.skipped_existing
    );
    result
}

// ========================================================================
// Source: Comet Browser History
// ========================================================================

async fn import_browser_history(
    state: &AppState,
    app_handle: Option<&tauri::AppHandle>,
) -> SourceResult {
    let mut result = SourceResult {
        source: "browser".to_string(),
        ..Default::default()
    };

    let (rows, temp_db) = match read_browser_history() {
        Ok(r) => r,
        Err(e) => {
            tracing::info!("Bootstrap browser: {}", e);
            return result;
        }
    };

    let total = rows.len() as u32;
    tracing::info!("Bootstrap browser: found {} URLs", total);

    for (i, (url, title, visit_count)) in rows.iter().enumerate() {
        result.items_scanned += 1;

        let url_hash = deterministic_hash(url);
        let memory_id = format!("bootstrap::comet::{}", url_hash);

        // Importance scales with visit count: 1-5 visits=0.3, 5-20=0.5, 20+=0.7
        let importance = if *visit_count > 20 {
            0.7
        } else if *visit_count > 5 {
            0.5
        } else {
            0.3
        };

        let content = format!(
            "[Bookmark] {} (visited {} times)\n{}",
            title, visit_count, url
        );

        match store_as_memory(state, &memory_id, &content, "semantic", importance).await {
            Ok(true) => result.memories_created += 1,
            Ok(false) => result.skipped_existing += 1,
            Err(e) => {
                tracing::debug!("Bootstrap browser error: {}", e);
                result.errors += 1;
            }
        }

        if i % 50 == 0 {
            emit_progress(
                app_handle,
                &BootstrapProgress {
                    source: "browser".to_string(),
                    phase: "storing".to_string(),
                    current: i as u32 + 1,
                    total,
                    memories_created: result.memories_created,
                },
            );
            tokio::task::yield_now().await;
        }
    }

    let _ = std::fs::remove_file(&temp_db);

    tracing::info!(
        "Bootstrap browser complete: {} created, {} skipped",
        result.memories_created,
        result.skipped_existing
    );
    result
}

// ========================================================================
// Source: Deepnote AI
// ========================================================================

async fn import_deepnote(
    state: &AppState,
    app_handle: Option<&tauri::AppHandle>,
) -> SourceResult {
    let mut result = SourceResult {
        source: "deepnote".to_string(),
        ..Default::default()
    };

    let (rows, temp_db) = match read_deepnote_data() {
        Ok(r) => r,
        Err(e) => {
            tracing::info!("Bootstrap deepnote: {}", e);
            return result;
        }
    };

    let total = rows.len() as u32;
    for (i, content) in rows.iter().enumerate() {
        result.items_scanned += 1;

        let content_hash = deterministic_hash(content);
        let memory_id = format!("bootstrap::deepnote::{}", content_hash);

        match store_as_memory(state, &memory_id, content, "procedural", 0.5).await {
            Ok(true) => result.memories_created += 1,
            Ok(false) => result.skipped_existing += 1,
            Err(e) => {
                tracing::debug!("Bootstrap deepnote error: {}", e);
                result.errors += 1;
            }
        }

        if i % 20 == 0 {
            emit_progress(
                app_handle,
                &BootstrapProgress {
                    source: "deepnote".to_string(),
                    phase: "storing".to_string(),
                    current: i as u32 + 1,
                    total,
                    memories_created: result.memories_created,
                },
            );
            tokio::task::yield_now().await;
        }
    }

    let _ = std::fs::remove_file(&temp_db);

    tracing::info!(
        "Bootstrap deepnote complete: {} created, {} skipped",
        result.memories_created,
        result.skipped_existing
    );
    result
}

// ========================================================================
// Source: Claude Plans (implementation plans from ~/.claude/plans/)
// ========================================================================

async fn import_claude_plans(
    state: &AppState,
    app_handle: Option<&tauri::AppHandle>,
) -> SourceResult {
    let mut result = SourceResult {
        source: "claude_plans".to_string(),
        ..Default::default()
    };

    let home = match dirs::home_dir() {
        Some(h) => h,
        None => return result,
    };

    // Plans live in ~/.claude/plans/ as .md files
    let plans_dir = home.join(".claude").join("plans");
    if !plans_dir.exists() {
        tracing::info!("Bootstrap: ~/.claude/plans not found, skipping claude_plans");
        return result;
    }

    let mut plan_files = Vec::new();
    collect_files_with_ext(&plans_dir, "md", &mut plan_files, 2);

    let total = plan_files.len() as u32;
    tracing::info!("Bootstrap claude_plans: found {} plan files", total);

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
            match store_as_memory(state, &chunk_id, chunk, "goal", 0.75).await {
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

// ========================================================================
// Source: Claude Desktop agent-mode sessions
// ========================================================================

async fn import_claude_desktop(
    state: &AppState,
    app_handle: Option<&tauri::AppHandle>,
) -> SourceResult {
    let mut result = SourceResult {
        source: "claude_desktop".to_string(),
        ..Default::default()
    };

    let home = match dirs::home_dir() {
        Some(h) => h,
        None => return result,
    };

    let sessions_dir = home
        .join("Library")
        .join("Application Support")
        .join("Claude")
        .join("local-agent-mode-sessions");
    if !sessions_dir.exists() {
        tracing::info!(
            "Bootstrap: Claude Desktop local-agent-mode-sessions not found, skipping"
        );
        return result;
    }

    // Collect all .jsonl files from agent-mode sessions
    let mut session_files = Vec::new();
    collect_files_with_ext(&sessions_dir, "jsonl", &mut session_files, 3);

    let total = session_files.len() as u32;
    tracing::info!(
        "Bootstrap claude_desktop: found {} session files",
        total
    );

    for (i, file_path) in session_files.iter().enumerate() {
        result.items_scanned += 1;

        let content = match std::fs::read_to_string(file_path) {
            Ok(c) => c,
            Err(_) => continue,
        };

        let file_hash = deterministic_hash(&file_path.to_string_lossy());

        // Parse JSONL: extract both user and assistant messages
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
                            if let Some(text) =
                                item.get("text").and_then(|t| t.as_str())
                            {
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
            continue;
        }

        // Chunk into segments
        let segments = chunker::chunk_text(&conversation_text, 400, 50);

        for (si, segment) in segments.iter().enumerate().take(15) {
            let seg_id = format!("bootstrap::claude_desktop::{}::{}", file_hash, si);

            match store_as_memory(state, &seg_id, segment, "episodic", 0.65).await {
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

// ========================================================================
// Source: Claude history.jsonl (user queries / session summaries)
// ========================================================================

async fn import_claude_history(
    state: &AppState,
    app_handle: Option<&tauri::AppHandle>,
) -> SourceResult {
    let mut result = SourceResult {
        source: "claude_history".to_string(),
        ..Default::default()
    };

    let home = match dirs::home_dir() {
        Some(h) => h,
        None => return result,
    };

    let history_file = home.join(".claude").join("history.jsonl");
    if !history_file.exists() {
        tracing::info!("Bootstrap: ~/.claude/history.jsonl not found, skipping");
        return result;
    }

    let content = match std::fs::read_to_string(&history_file) {
        Ok(c) => c,
        Err(e) => {
            tracing::info!("Bootstrap claude_history: read error: {}", e);
            return result;
        }
    };

    // Each line is a JSON entry with session metadata
    // Typical fields: "command", "cwd", "timestamp", "session_id", "prompt"
    let mut entries: Vec<String> = Vec::new();
    for line in content.lines() {
        if line.trim().is_empty() {
            continue;
        }
        if let Ok(json) = serde_json::from_str::<serde_json::Value>(line) {
            // Extract the user's prompt/command — the most valuable field
            let mut entry_text = String::new();

            if let Some(prompt) = json.get("prompt").and_then(|p| p.as_str()) {
                entry_text.push_str(prompt);
            }
            if let Some(command) = json.get("command").and_then(|c| c.as_str()) {
                if !command.is_empty() && entry_text.is_empty() {
                    entry_text.push_str(command);
                }
            }
            // Include working directory for context
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

    // Group entries in batches of 10 for better semantic embedding
    // (individual prompts are often too short for good embeddings)
    let batch_size = 10;
    let batches: Vec<String> = entries
        .chunks(batch_size)
        .map(|batch| batch.join("\n---\n"))
        .collect();

    for (i, batch) in batches.iter().enumerate() {
        let batch_hash = deterministic_hash(batch);
        let memory_id = format!("bootstrap::claude_history::{}", batch_hash);

        match store_as_memory(state, &memory_id, batch, "episodic", 0.5).await {
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

// ---- File collection helpers ----

/// Recursively collect files with a specific extension.
fn collect_files_with_ext(dir: &Path, ext: &str, files: &mut Vec<PathBuf>, max_depth: u32) {
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
fn collect_memory_md_files(dir: &Path, files: &mut Vec<PathBuf>, max_depth: u32) {
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
