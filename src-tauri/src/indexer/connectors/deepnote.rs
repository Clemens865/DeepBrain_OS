//! Deepnote AI connector — imports chat/conversation data from deepnote-ai.db.

use std::path::PathBuf;

use rusqlite::Connection;

use crate::indexer::bootstrap::{
    copy_sqlite_to_temp, deterministic_hash, emit_progress, store_as_memory, BootstrapProgress,
    SourceResult,
};
use crate::state::AppState;

use super::{Connector, ConnectorConfig, DetectionResult};

pub struct DeepnoteConnector;

impl DeepnoteConnector {
    fn default_path() -> Option<PathBuf> {
        dirs::home_dir().map(|h| {
            h.join("Library")
                .join("Application Support")
                .join("deepnote-ai")
                .join("deepnote-ai.db")
        })
    }
}

/// Read Deepnote AI data with schema discovery (synchronous — Connection is !Send).
fn read_deepnote_data(db_path: &std::path::Path) -> Result<(Vec<String>, PathBuf), String> {
    if !db_path.exists() {
        return Err("Deepnote AI db not found".to_string());
    }
    let temp_db = copy_sqlite_to_temp(db_path, "deepnote")?;
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

#[async_trait::async_trait]
impl Connector for DeepnoteConnector {
    fn id(&self) -> &str {
        "deepnote"
    }
    fn name(&self) -> &str {
        "Deepnote"
    }
    fn description(&self) -> &str {
        "AI notebooks"
    }
    fn icon(&self) -> &str {
        "notebook"
    }
    fn default_enabled(&self) -> bool {
        true
    }
    fn memory_type(&self) -> &str {
        "procedural"
    }
    fn default_importance(&self) -> f64 {
        0.5
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
            source: "deepnote".to_string(),
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

        let (rows, temp_db) = match read_deepnote_data(&db_path) {
            Ok(r) => r,
            Err(e) => {
                tracing::info!("Bootstrap deepnote: {}", e);
                return result;
            }
        };

        let importance = config.importance_override.unwrap_or(self.default_importance());
        let total = rows.len() as u32;

        for (i, content) in rows.iter().enumerate() {
            result.items_scanned += 1;

            let content_hash = deterministic_hash(content);
            let memory_id = format!("bootstrap::deepnote::{}", content_hash);

            match store_as_memory(state, &memory_id, content, "procedural", importance).await {
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
}
