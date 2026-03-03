//! Safari connector — imports browsing history from Safari's History.db.
//!
//! Safari's database is protected by macOS TCC (Transparency, Consent, and Control).
//! If the user hasn't granted Full Disk Access to DeepBrain, detection will succeed
//! but import will fail gracefully with a helpful message.

use std::path::PathBuf;

use rusqlite::Connection;

use crate::indexer::bootstrap::{
    copy_sqlite_to_temp, deterministic_hash, emit_progress, store_as_memory, BootstrapProgress,
    SourceResult,
};
use crate::state::AppState;

use super::{Connector, ConnectorConfig, DetectionResult};

pub struct SafariConnector;

impl SafariConnector {
    fn default_path() -> Option<PathBuf> {
        dirs::home_dir().map(|h| h.join("Library").join("Safari").join("History.db"))
    }
}

/// Safari URL record.
struct SafariUrl {
    url: String,
    title: String,
    visit_count: i64,
}

/// Read Safari history synchronously (Connection is !Send).
fn read_safari_history(
    db_path: &std::path::Path,
) -> Result<(Vec<SafariUrl>, PathBuf), String> {
    if !db_path.exists() {
        return Err("Safari History.db not found".to_string());
    }
    let temp_db = copy_sqlite_to_temp(db_path, "safari_history")?;
    let conn = Connection::open_with_flags(
        &temp_db,
        rusqlite::OpenFlags::SQLITE_OPEN_READ_ONLY | rusqlite::OpenFlags::SQLITE_OPEN_NO_MUTEX,
    )
    .map_err(|e| format!("open safari history copy: {}", e))?;

    let sql = "SELECT hi.url, hv.title, hi.visit_count
               FROM history_items hi
               JOIN history_visits hv ON hi.id = hv.history_item
               WHERE hv.title IS NOT NULL AND hv.title != ''
               GROUP BY hi.url
               ORDER BY hi.visit_count DESC
               LIMIT 2000";
    let mut stmt = conn.prepare(sql).map_err(|e| format!("prepare: {}", e))?;
    let rows: Vec<SafariUrl> = stmt
        .query_map([], |row| {
            Ok(SafariUrl {
                url: row.get(0)?,
                title: row.get(1)?,
                visit_count: row.get(2)?,
            })
        })
        .map_err(|e| format!("query: {}", e))?
        .filter_map(|r| r.ok())
        .collect();

    Ok((rows, temp_db))
}

#[async_trait::async_trait]
impl Connector for SafariConnector {
    fn id(&self) -> &str {
        "safari"
    }
    fn name(&self) -> &str {
        "Safari"
    }
    fn description(&self) -> &str {
        "Safari browsing history"
    }
    fn icon(&self) -> &str {
        "compass"
    }
    fn default_enabled(&self) -> bool {
        true
    }
    fn memory_type(&self) -> &str {
        "semantic"
    }
    fn default_importance(&self) -> f64 {
        0.3
    }

    fn detect(&self) -> DetectionResult {
        let path = Self::default_path();
        let exists = path.as_ref().map(|p| p.exists()).unwrap_or(false);

        if !exists {
            return DetectionResult {
                available: false,
                path: None,
                details: None,
            };
        }

        // Check if we can actually read the file (TCC may block us)
        let can_read = path
            .as_ref()
            .map(|p| std::fs::metadata(p).is_ok())
            .unwrap_or(false);

        let details = if can_read {
            Some("Safari history accessible".to_string())
        } else {
            Some(
                "Grant Full Disk Access to DeepBrain in System Settings \u{2192} Privacy".to_string(),
            )
        };

        DetectionResult {
            available: true,
            path: path.map(|p| p.to_string_lossy().to_string()),
            details,
        }
    }

    async fn import(
        &self,
        state: &AppState,
        app_handle: Option<&tauri::AppHandle>,
        config: &ConnectorConfig,
    ) -> SourceResult {
        let mut result = SourceResult {
            source: "safari".to_string(),
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

        let (rows, temp_db) = match read_safari_history(&db_path) {
            Ok(r) => r,
            Err(e) => {
                tracing::info!("Bootstrap safari: {}", e);
                return result;
            }
        };

        let total = rows.len() as u32;
        tracing::info!("Bootstrap safari: found {} URLs", total);
        let base_importance = config.importance_override.unwrap_or(self.default_importance());

        for (i, row) in rows.iter().enumerate() {
            result.items_scanned += 1;

            let url_hash = deterministic_hash(&row.url);
            let memory_id = format!("bootstrap::safari::{}", url_hash);

            let importance = if row.visit_count > 20 {
                (base_importance + 0.4).min(1.0)
            } else if row.visit_count > 5 {
                (base_importance + 0.2).min(1.0)
            } else if row.visit_count > 2 {
                (base_importance + 0.1).min(1.0)
            } else {
                base_importance
            };

            let content = format!("{} — {}", row.title, row.url);

            match store_as_memory(state, &memory_id, &content, "semantic", importance).await {
                Ok(true) => result.memories_created += 1,
                Ok(false) => result.skipped_existing += 1,
                Err(e) => {
                    tracing::debug!("Bootstrap safari error: {}", e);
                    result.errors += 1;
                }
            }

            if i % 50 == 0 {
                emit_progress(
                    app_handle,
                    &BootstrapProgress {
                        source: "safari".to_string(),
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
            "Bootstrap safari complete: {} created, {} skipped",
            result.memories_created,
            result.skipped_existing
        );
        result
    }
}
