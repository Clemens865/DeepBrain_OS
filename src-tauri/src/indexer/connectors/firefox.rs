//! Firefox connector — imports browsing history from places.sqlite.
//!
//! Firefox stores history in per-profile `places.sqlite` databases under
//! `~/Library/Application Support/Firefox/Profiles/`. Profile directories
//! are named like `abc123.default-release`.

use std::path::{Path, PathBuf};

use rusqlite::Connection;

use crate::indexer::bootstrap::{
    copy_sqlite_to_temp, deterministic_hash, emit_progress, store_as_memory, BootstrapProgress,
    SourceResult,
};
use crate::state::AppState;

use super::{Connector, ConnectorConfig, DetectionResult};

pub struct FirefoxConnector;

/// Discover all Firefox `places.sqlite` files across profiles.
fn discover_firefox_profiles() -> Vec<PathBuf> {
    let home = match dirs::home_dir() {
        Some(h) => h,
        None => return Vec::new(),
    };

    let profiles_dir = home
        .join("Library")
        .join("Application Support")
        .join("Firefox")
        .join("Profiles");

    if !profiles_dir.is_dir() {
        return Vec::new();
    }

    let mut dbs = Vec::new();
    if let Ok(entries) = std::fs::read_dir(&profiles_dir) {
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_dir() {
                let places = path.join("places.sqlite");
                if places.exists() {
                    dbs.push(places);
                }
            }
        }
    }
    dbs
}

/// Firefox URL record.
struct FirefoxUrl {
    url: String,
    title: String,
    visit_count: i64,
    frecency: i64,
}

/// Read Firefox history synchronously (Connection is !Send).
fn read_firefox_history(
    db_path: &Path,
    label: &str,
) -> Result<(Vec<FirefoxUrl>, PathBuf), String> {
    if !db_path.exists() {
        return Err("places.sqlite not found".to_string());
    }
    let temp_db = copy_sqlite_to_temp(db_path, label)?;
    let conn = Connection::open_with_flags(
        &temp_db,
        rusqlite::OpenFlags::SQLITE_OPEN_READ_ONLY | rusqlite::OpenFlags::SQLITE_OPEN_NO_MUTEX,
    )
    .map_err(|e| format!("open firefox places copy: {}", e))?;

    let sql = "SELECT url, title, visit_count, frecency
               FROM moz_places
               WHERE title IS NOT NULL AND title != ''
                 AND visit_count > 0
               ORDER BY frecency DESC
               LIMIT 2000";
    let mut stmt = conn.prepare(sql).map_err(|e| format!("prepare: {}", e))?;
    let rows: Vec<FirefoxUrl> = stmt
        .query_map([], |row| {
            Ok(FirefoxUrl {
                url: row.get(0)?,
                title: row.get(1)?,
                visit_count: row.get(2)?,
                frecency: row.get(3)?,
            })
        })
        .map_err(|e| format!("query: {}", e))?
        .filter_map(|r| r.ok())
        .collect();

    Ok((rows, temp_db))
}

#[async_trait::async_trait]
impl Connector for FirefoxConnector {
    fn id(&self) -> &str {
        "firefox"
    }
    fn name(&self) -> &str {
        "Firefox"
    }
    fn description(&self) -> &str {
        "Firefox browsing history"
    }
    fn icon(&self) -> &str {
        "flame"
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
        let dbs = discover_firefox_profiles();
        if dbs.is_empty() {
            return DetectionResult {
                available: false,
                path: None,
                details: None,
            };
        }

        let detail = if dbs.len() == 1 {
            "1 profile".to_string()
        } else {
            format!("{} profiles", dbs.len())
        };

        DetectionResult {
            available: true,
            path: dbs.first().map(|p| p.to_string_lossy().to_string()),
            details: Some(detail),
        }
    }

    async fn import(
        &self,
        state: &AppState,
        app_handle: Option<&tauri::AppHandle>,
        config: &ConnectorConfig,
    ) -> SourceResult {
        let mut result = SourceResult {
            source: "firefox".to_string(),
            ..Default::default()
        };

        let profile_dbs = discover_firefox_profiles();
        if profile_dbs.is_empty() {
            return result;
        }

        let base_importance = config.importance_override.unwrap_or(self.default_importance());

        for (pi, db_path) in profile_dbs.iter().enumerate() {
            let label = format!("firefox_profile_{}", pi);

            let (rows, temp_db) = match read_firefox_history(db_path, &label) {
                Ok(data) => data,
                Err(e) => {
                    tracing::info!("Bootstrap firefox profile {}: {}", pi, e);
                    continue;
                }
            };

            let total = rows.len() as u32;
            tracing::info!("Bootstrap firefox profile {}: {} URLs", pi, total);

            for (i, row) in rows.iter().enumerate() {
                result.items_scanned += 1;

                let url_hash = deterministic_hash(&row.url);
                let memory_id = format!("bootstrap::firefox::{}", url_hash);

                // Scale importance by frecency (Firefox's built-in relevance score)
                let importance = if row.frecency > 10_000 {
                    (base_importance + 0.3).min(1.0)
                } else if row.frecency > 1_000 {
                    (base_importance + 0.2).min(1.0)
                } else if row.frecency > 100 {
                    (base_importance + 0.1).min(1.0)
                } else {
                    base_importance
                };

                let content = format!("{} — {}", row.title, row.url);

                match store_as_memory(state, &memory_id, &content, "semantic", importance).await {
                    Ok(true) => result.memories_created += 1,
                    Ok(false) => result.skipped_existing += 1,
                    Err(e) => {
                        tracing::debug!("Bootstrap firefox error: {}", e);
                        result.errors += 1;
                    }
                }

                if i % 50 == 0 {
                    emit_progress(
                        app_handle,
                        &BootstrapProgress {
                            source: "firefox".to_string(),
                            phase: format!("profile {}", pi),
                            current: i as u32 + 1,
                            total,
                            memories_created: result.memories_created,
                        },
                    );
                    tokio::task::yield_now().await;
                }
            }

            let _ = std::fs::remove_file(&temp_db);
        }

        tracing::info!(
            "Bootstrap firefox complete: {} created, {} skipped",
            result.memories_created,
            result.skipped_existing
        );
        result
    }
}
