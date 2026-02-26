//! WhatsApp connector — imports chat messages from ChatStorage.sqlite.

use std::path::PathBuf;

use rusqlite::Connection;

use crate::indexer::bootstrap::{
    copy_sqlite_to_temp, deterministic_hash, emit_progress, store_as_memory, BootstrapProgress,
    SourceResult, CORE_DATA_EPOCH_OFFSET,
};
use crate::indexer::chunker;
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

/// Read WhatsApp messages synchronously (Connection is !Send).
fn read_whatsapp_messages(db_path: &std::path::Path) -> Result<(Vec<WhatsAppMsg>, PathBuf), String> {
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

    let mut stmt = conn.prepare(sql).map_err(|e| format!("prepare: {}", e))?;
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
        "Chat messages"
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
        let importance = config.importance_override.unwrap_or(self.default_importance());

        let mut chat_idx = 0u32;
        for (chat_jid, msgs) in &chats {
            result.items_scanned += msgs.len() as u32;

            let chat_name = msgs
                .first()
                .map(|m| m.chat_name.as_str())
                .unwrap_or(chat_jid);
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

                match store_as_memory(state, &win_id, window, "episodic", importance).await {
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

        let _ = std::fs::remove_file(&temp_db);

        tracing::info!(
            "Bootstrap whatsapp complete: {} created, {} skipped",
            result.memories_created,
            result.skipped_existing
        );
        result
    }
}
