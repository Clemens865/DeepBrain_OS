//! Browser connector — imports browsing history from Comet (Chromium-based) browser.

use std::path::PathBuf;

use rusqlite::Connection;

use crate::indexer::bootstrap::{
    copy_sqlite_to_temp, deterministic_hash, emit_progress, store_as_memory, BootstrapProgress,
    SourceResult,
};
use crate::state::AppState;

use super::{Connector, ConnectorConfig, DetectionResult};

pub struct BrowserConnector;

impl BrowserConnector {
    fn default_path() -> Option<PathBuf> {
        dirs::home_dir().map(|h| {
            h.join("Library")
                .join("Application Support")
                .join("Comet")
                .join("Default")
                .join("History")
        })
    }
}

/// Read browser history synchronously (Connection is !Send).
fn read_browser_history(
    db_path: &std::path::Path,
) -> Result<(Vec<(String, String, u32)>, PathBuf), String> {
    if !db_path.exists() {
        return Err("Browser History DB not found".to_string());
    }
    let temp_db = copy_sqlite_to_temp(db_path, "browser_history")?;
    let conn = Connection::open_with_flags(
        &temp_db,
        rusqlite::OpenFlags::SQLITE_OPEN_READ_ONLY | rusqlite::OpenFlags::SQLITE_OPEN_NO_MUTEX,
    )
    .map_err(|e| format!("open browser history copy: {}", e))?;

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

#[async_trait::async_trait]
impl Connector for BrowserConnector {
    fn id(&self) -> &str {
        "browser"
    }
    fn name(&self) -> &str {
        "Browser"
    }
    fn description(&self) -> &str {
        "Browsing history"
    }
    fn icon(&self) -> &str {
        "globe"
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
        let available = path.as_ref().map(|p| p.exists()).unwrap_or(false);
        let details = if available {
            Some("Comet browser".to_string())
        } else {
            // Check Chrome as fallback hint
            let chrome = dirs::home_dir().map(|h| {
                h.join("Library")
                    .join("Application Support")
                    .join("Google")
                    .join("Chrome")
                    .join("Default")
                    .join("History")
            });
            if chrome.as_ref().map(|p| p.exists()).unwrap_or(false) {
                Some("Chrome detected (use path override to import)".to_string())
            } else {
                None
            }
        };
        DetectionResult {
            available,
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
            source: "browser".to_string(),
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

        let (rows, temp_db) = match read_browser_history(&db_path) {
            Ok(r) => r,
            Err(e) => {
                tracing::info!("Bootstrap browser: {}", e);
                return result;
            }
        };

        let total = rows.len() as u32;
        tracing::info!("Bootstrap browser: found {} URLs", total);
        let base_importance = config.importance_override.unwrap_or(self.default_importance());

        for (i, (url, title, visit_count)) in rows.iter().enumerate() {
            result.items_scanned += 1;

            let url_hash = deterministic_hash(url);
            let memory_id = format!("bootstrap::comet::{}", url_hash);

            let importance = if *visit_count > 20 {
                (base_importance + 0.4).min(1.0)
            } else if *visit_count > 5 {
                (base_importance + 0.2).min(1.0)
            } else {
                base_importance
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
}
