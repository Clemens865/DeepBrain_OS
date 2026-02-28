//! WhatsApp connector — imports chat conversations from ChatStorage.sqlite.
//!
//! Each chat becomes one memory (or monthly segments for long chats).
//! The embedding is computed from a descriptive summary while the full
//! conversation text is stored for browsing in the detail panel.

use std::collections::HashMap;
use std::path::PathBuf;

use rusqlite::Connection;

use crate::indexer::bootstrap::{
    copy_sqlite_to_temp, deterministic_hash, emit_progress, store_as_memory_with_summary,
    BootstrapProgress, SourceResult, CORE_DATA_EPOCH_OFFSET,
};
use crate::state::AppState;

use super::{Connector, ConnectorConfig, DetectionResult};

pub struct WhatsAppConnector;

/// A single WhatsApp message with sender context and timestamp.
struct WhatsAppMsg {
    text: String,
    is_from_me: bool,
    timestamp_unix: i64,
    sender_name: String,
    chat_jid: String,
    chat_name: String,
}

/// Maximum messages per memory before splitting into monthly segments.
const MONTHLY_SPLIT_THRESHOLD: usize = 100;

/// Maximum topic snippets to include in the embedding summary.
const MAX_SUMMARY_SNIPPETS: usize = 8;

/// Maximum characters per snippet in the embedding summary.
const SNIPPET_MAX_CHARS: usize = 80;

impl WhatsAppConnector {
    fn default_path() -> Option<PathBuf> {
        dirs::home_dir().map(|h| {
            h.join("Library")
                .join("Group Containers")
                .join("group.net.whatsapp.WhatsApp.shared")
                .join("ChatStorage.sqlite")
        })
    }
}

/// Build a concise embedding summary for a set of messages in a chat.
///
/// Format: "WhatsApp conversation with {name} ({date_range}, {n} messages).
///          Topics: {snippet1}; {snippet2}; ..."
///
/// This gives the embedding model a semantically rich, short text to vectorize
/// instead of raw message dumps that would get truncated.
fn build_embedding_summary(chat_name: &str, msgs: &[&WhatsAppMsg], period_label: &str) -> String {
    let msg_count = msgs.len();

    // Collect unique topic snippets from the longest/most substantive messages.
    // Prefer messages that aren't from "You" for better topic representation,
    // but fall back to all messages.
    let mut candidates: Vec<&WhatsAppMsg> = msgs.iter().copied().collect();
    candidates.sort_by(|a, b| b.text.len().cmp(&a.text.len()));

    let mut snippets: Vec<String> = Vec::new();
    let mut seen = std::collections::HashSet::new();
    for msg in &candidates {
        if snippets.len() >= MAX_SUMMARY_SNIPPETS {
            break;
        }
        // Take first N chars as a snippet, deduplicate by prefix
        let snippet: String = msg.text.chars().take(SNIPPET_MAX_CHARS).collect();
        let key = snippet.to_lowercase();
        if seen.contains(&key) {
            continue;
        }
        seen.insert(key);
        snippets.push(snippet);
    }

    let topics = if snippets.is_empty() {
        String::new()
    } else {
        format!(" Topics: {}", snippets.join("; "))
    };

    format!(
        "WhatsApp conversation with {} ({}, {} messages).{}",
        chat_name, period_label, msg_count, topics
    )
}

/// Format messages into a readable conversation transcript.
fn format_conversation(chat_name: &str, msgs: &[&WhatsAppMsg], period_label: &str) -> String {
    let mut lines = Vec::with_capacity(msgs.len() + 3);
    lines.push(format!(
        "WhatsApp conversation with {} — {}\n",
        chat_name, period_label
    ));

    for msg in msgs {
        let dt = chrono::DateTime::from_timestamp(msg.timestamp_unix, 0)
            .map(|d| d.format("%Y-%m-%d %H:%M").to_string())
            .unwrap_or_else(|| "unknown".to_string());
        lines.push(format!("[{}] {}: {}", dt, msg.sender_name, msg.text));
    }

    lines.join("\n")
}

/// Group messages by year-month key (e.g. "2026-02").
fn group_by_month<'a>(msgs: &[&'a WhatsAppMsg]) -> Vec<(String, Vec<&'a WhatsAppMsg>)> {
    let mut months: HashMap<String, Vec<&'a WhatsAppMsg>> = HashMap::new();
    for msg in msgs {
        let key = chrono::DateTime::from_timestamp(msg.timestamp_unix, 0)
            .map(|d| d.format("%Y-%m").to_string())
            .unwrap_or_else(|| "unknown".to_string());
        months.entry(key).or_default().push(msg);
    }
    let mut sorted: Vec<_> = months.into_iter().collect();
    sorted.sort_by(|a, b| a.0.cmp(&b.0));
    sorted
}

/// Build a human-readable label for a date range.
fn date_range_label(msgs: &[&WhatsAppMsg]) -> String {
    let first = msgs.first().and_then(|m| {
        chrono::DateTime::from_timestamp(m.timestamp_unix, 0)
            .map(|d| d.format("%b %Y").to_string())
    });
    let last = msgs.last().and_then(|m| {
        chrono::DateTime::from_timestamp(m.timestamp_unix, 0)
            .map(|d| d.format("%b %Y").to_string())
    });
    match (first, last) {
        (Some(f), Some(l)) if f == l => f,
        (Some(f), Some(l)) => format!("{} – {}", f, l),
        _ => "unknown dates".to_string(),
    }
}

/// Read WhatsApp messages synchronously (Connection is !Send).
fn read_whatsapp_messages(
    db_path: &std::path::Path,
) -> Result<(Vec<WhatsAppMsg>, PathBuf), String> {
    if !db_path.exists() {
        return Err("WhatsApp ChatStorage.sqlite not found".to_string());
    }
    let temp_db = copy_sqlite_to_temp(db_path, "whatsapp")?;
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

    let mut stmt = conn
        .prepare(sql)
        .map_err(|e| format!("prepare: {}", e))?;
    let messages: Vec<WhatsAppMsg> = stmt
        .query_map([], |row| {
            let text: String = row.get(0)?;
            let is_from_me: bool = row.get::<_, i32>(1).unwrap_or(0) == 1;
            let core_data_ts: f64 = row.get::<_, f64>(2).unwrap_or(0.0);
            let sender_name: String = row.get::<_, Option<String>>(3)?.unwrap_or_default();
            let chat_jid: String = row
                .get::<_, Option<String>>(4)?
                .unwrap_or_else(|| "unknown".to_string());
            let chat_name: String = row
                .get::<_, Option<String>>(5)?
                .unwrap_or_else(|| chat_jid.clone());

            let timestamp_unix = core_data_ts as i64 + CORE_DATA_EPOCH_OFFSET;

            Ok(WhatsAppMsg {
                text,
                is_from_me,
                timestamp_unix,
                sender_name: if is_from_me {
                    "You".to_string()
                } else {
                    sender_name
                },
                chat_jid,
                chat_name,
            })
        })
        .map_err(|e| format!("query: {}", e))?
        .filter_map(|r| r.ok())
        .collect();

    Ok((messages, temp_db))
}

#[async_trait::async_trait]
impl Connector for WhatsAppConnector {
    fn id(&self) -> &str {
        "whatsapp"
    }
    fn name(&self) -> &str {
        "WhatsApp"
    }
    fn description(&self) -> &str {
        "Chat conversations"
    }
    fn icon(&self) -> &str {
        "chat"
    }
    fn default_enabled(&self) -> bool {
        true
    }
    fn memory_type(&self) -> &str {
        "episodic"
    }
    fn default_importance(&self) -> f64 {
        0.4
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
            source: "whatsapp".to_string(),
            ..Default::default()
        };

        let db_path = config
            .path_override
            .as_ref()
            .map(PathBuf::from)
            .or_else(Self::default_path);

        let db_path = match db_path {
            Some(p) => p,
            None => return result,
        };

        let (messages, temp_db) = match read_whatsapp_messages(&db_path) {
            Ok(r) => r,
            Err(e) => {
                tracing::info!("Bootstrap whatsapp: {}", e);
                return result;
            }
        };

        // Group messages by chat_jid
        let mut chats: HashMap<String, Vec<&WhatsAppMsg>> = HashMap::new();
        for msg in &messages {
            chats.entry(msg.chat_jid.clone()).or_default().push(msg);
        }

        let total = chats.len() as u32;
        tracing::info!(
            "Bootstrap whatsapp: {} messages in {} chats",
            messages.len(),
            total
        );
        let importance = config
            .importance_override
            .unwrap_or(self.default_importance());

        let mut chat_idx = 0u32;
        for (chat_jid, msgs) in &chats {
            result.items_scanned += msgs.len() as u32;

            let chat_name = msgs
                .first()
                .map(|m| m.chat_name.as_str())
                .unwrap_or(chat_jid);
            let chat_hash = deterministic_hash(chat_jid);

            // Decide: single memory vs monthly segments
            if msgs.len() <= MONTHLY_SPLIT_THRESHOLD {
                // ── Single memory for the whole chat ──
                let period = date_range_label(msgs);
                let summary = build_embedding_summary(chat_name, msgs, &period);
                let content = format_conversation(chat_name, msgs, &period);
                let mem_id = format!("bootstrap::whatsapp::{}", chat_hash);

                match store_as_memory_with_summary(
                    state, &mem_id, &summary, &content, "episodic", importance,
                )
                .await
                {
                    Ok(true) => result.memories_created += 1,
                    Ok(false) => result.skipped_existing += 1,
                    Err(e) => {
                        tracing::debug!("Bootstrap whatsapp error: {}", e);
                        result.errors += 1;
                    }
                }
            } else {
                // ── Split into monthly segments ──
                let monthly = group_by_month(msgs);
                for (month_key, month_msgs) in &monthly {
                    let period = date_range_label(month_msgs);
                    let summary = build_embedding_summary(chat_name, month_msgs, &period);
                    let content = format_conversation(chat_name, month_msgs, &period);
                    let mem_id =
                        format!("bootstrap::whatsapp::{}::{}", chat_hash, month_key);

                    match store_as_memory_with_summary(
                        state, &mem_id, &summary, &content, "episodic", importance,
                    )
                    .await
                    {
                        Ok(true) => result.memories_created += 1,
                        Ok(false) => result.skipped_existing += 1,
                        Err(e) => {
                            tracing::debug!("Bootstrap whatsapp error: {}", e);
                            result.errors += 1;
                        }
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

        let _ = std::fs::remove_file(&temp_db);

        tracing::info!(
            "Bootstrap whatsapp complete: {} memories created, {} skipped ({} chats, {} messages)",
            result.memories_created,
            result.skipped_existing,
            total,
            messages.len()
        );
        result
    }
}
