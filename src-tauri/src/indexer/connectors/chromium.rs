//! Chromium connector — imports browsing history and search terms from all
//! Chromium-based browsers: Chrome, Brave, Edge, Arc, Vivaldi, Opera, Comet.
//!
//! Discovers all profiles per browser and imports top URLs by visit count
//! plus recent keyword search terms as episodic memories.

use std::path::{Path, PathBuf};

use rusqlite::Connection;

use crate::indexer::bootstrap::{
    copy_sqlite_to_temp, deterministic_hash, emit_progress, store_as_memory, BootstrapProgress,
    SourceResult,
};
use crate::state::AppState;

use super::{Connector, ConnectorConfig, DetectionResult};

pub struct ChromiumConnector;

/// Known Chromium-based browsers and their Application Support subdirectories.
const CHROMIUM_BROWSERS: &[(&str, &str)] = &[
    ("Chrome", "Google/Chrome"),
    ("Brave", "BraveSoftware/Brave-Browser"),
    ("Edge", "Microsoft Edge"),
    ("Arc", "Arc/User Data"),
    ("Vivaldi", "Vivaldi"),
    ("Opera", "com.operasoftware.Opera"),
    ("Comet", "Comet"),
];

/// A discovered browser profile with its History DB path.
struct BrowserProfile {
    browser: String,
    profile: String,
    history_path: PathBuf,
}

/// Discover all Chromium profiles across all known browsers.
fn discover_chromium_profiles() -> Vec<BrowserProfile> {
    let home = match dirs::home_dir() {
        Some(h) => h,
        None => return Vec::new(),
    };
    let app_support = home.join("Library").join("Application Support");
    let mut profiles = Vec::new();

    for &(browser_name, subdir) in CHROMIUM_BROWSERS {
        let browser_dir = app_support.join(subdir);
        if !browser_dir.is_dir() {
            continue;
        }

        // Check Default profile
        let default_history = browser_dir.join("Default").join("History");
        if default_history.exists() {
            profiles.push(BrowserProfile {
                browser: browser_name.to_string(),
                profile: "Default".to_string(),
                history_path: default_history,
            });
        }

        // Check numbered profiles (Profile 1, Profile 2, etc.)
        if let Ok(entries) = std::fs::read_dir(&browser_dir) {
            for entry in entries.flatten() {
                let name = entry.file_name();
                let name_str = name.to_string_lossy();
                if name_str.starts_with("Profile ") {
                    let history = entry.path().join("History");
                    if history.exists() {
                        profiles.push(BrowserProfile {
                            browser: browser_name.to_string(),
                            profile: name_str.to_string(),
                            history_path: history,
                        });
                    }
                }
            }
        }
    }

    profiles
}

/// URL record from the Chromium `urls` table.
struct UrlRow {
    url: String,
    title: String,
    visit_count: u32,
}

/// Read URLs and search terms from a single Chromium History database.
/// Returns (urls, search_terms, temp_db_path).
fn read_chromium_history(
    db_path: &Path,
    label: &str,
) -> Result<(Vec<UrlRow>, Vec<String>, PathBuf), String> {
    if !db_path.exists() {
        return Err("History DB not found".to_string());
    }
    let temp_db = copy_sqlite_to_temp(db_path, label)?;
    let conn = Connection::open_with_flags(
        &temp_db,
        rusqlite::OpenFlags::SQLITE_OPEN_READ_ONLY | rusqlite::OpenFlags::SQLITE_OPEN_NO_MUTEX,
    )
    .map_err(|e| format!("open chromium history copy: {}", e))?;

    // Read top URLs
    let url_sql = "SELECT url, title, visit_count
                   FROM urls
                   WHERE hidden = 0 AND visit_count > 0
                     AND title IS NOT NULL AND title != ''
                   ORDER BY visit_count DESC
                   LIMIT 2000";
    let mut stmt = conn.prepare(url_sql).map_err(|e| format!("prepare urls: {}", e))?;
    let urls: Vec<UrlRow> = stmt
        .query_map([], |row| {
            Ok(UrlRow {
                url: row.get(0)?,
                title: row.get(1)?,
                visit_count: row.get(2)?,
            })
        })
        .map_err(|e| format!("query urls: {}", e))?
        .filter_map(|r| r.ok())
        .collect();

    // Read search terms (may not exist in all Chromium variants)
    let search_terms = {
        let mut terms = Vec::new();
        if let Ok(mut stmt) = conn.prepare(
            "SELECT DISTINCT term FROM keyword_search_terms ORDER BY rowid DESC LIMIT 500",
        ) {
            if let Ok(rows) = stmt.query_map([], |row| row.get::<_, String>(0)) {
                terms = rows.filter_map(|r| r.ok()).collect();
            }
        }
        terms
    };

    Ok((urls, search_terms, temp_db))
}

#[async_trait::async_trait]
impl Connector for ChromiumConnector {
    fn id(&self) -> &str {
        "chromium"
    }
    fn name(&self) -> &str {
        "Chromium Browsers"
    }
    fn description(&self) -> &str {
        "Chrome, Brave, Edge, Arc, Vivaldi, Opera — all profiles"
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
        let profiles = discover_chromium_profiles();
        if profiles.is_empty() {
            return DetectionResult {
                available: false,
                path: None,
                details: None,
            };
        }

        // Build summary: "Chrome (3 profiles), Brave (1 profile)"
        let mut browser_counts: std::collections::HashMap<&str, usize> =
            std::collections::HashMap::new();
        for p in &profiles {
            *browser_counts.entry(&p.browser).or_insert(0) += 1;
        }
        let mut parts: Vec<String> = browser_counts
            .iter()
            .map(|(b, c)| {
                if *c == 1 {
                    format!("{} (1 profile)", b)
                } else {
                    format!("{} ({} profiles)", b, c)
                }
            })
            .collect();
        parts.sort();

        DetectionResult {
            available: true,
            path: None,
            details: Some(parts.join(", ")),
        }
    }

    async fn import(
        &self,
        state: &AppState,
        app_handle: Option<&tauri::AppHandle>,
        config: &ConnectorConfig,
    ) -> SourceResult {
        let mut result = SourceResult {
            source: "chromium".to_string(),
            ..Default::default()
        };

        let profiles = discover_chromium_profiles();
        if profiles.is_empty() {
            return result;
        }

        let base_importance = config.importance_override.unwrap_or(self.default_importance());
        let mut all_temp_dbs = Vec::new();

        for profile in &profiles {
            let label = format!(
                "chromium_{}_{}",
                profile.browser.to_lowercase().replace(' ', "_"),
                profile.profile.to_lowercase().replace(' ', "_")
            );

            let (urls, search_terms, temp_db) =
                match read_chromium_history(&profile.history_path, &label) {
                    Ok(data) => data,
                    Err(e) => {
                        tracing::info!(
                            "Bootstrap chromium: {} {} — {}",
                            profile.browser,
                            profile.profile,
                            e
                        );
                        continue;
                    }
                };
            all_temp_dbs.push(temp_db);

            let total = urls.len() as u32;
            tracing::info!(
                "Bootstrap chromium: {} {} — {} URLs, {} search terms",
                profile.browser,
                profile.profile,
                total,
                search_terms.len()
            );

            // Import URLs
            for (i, row) in urls.iter().enumerate() {
                result.items_scanned += 1;

                let url_hash = deterministic_hash(&row.url);
                let memory_id = format!(
                    "bootstrap::chromium::{}::{}::{}",
                    profile.browser, profile.profile, url_hash
                );

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
                        tracing::debug!("Bootstrap chromium URL error: {}", e);
                        result.errors += 1;
                    }
                }

                if i % 50 == 0 {
                    emit_progress(
                        app_handle,
                        &BootstrapProgress {
                            source: "chromium".to_string(),
                            phase: format!(
                                "urls — {} {}",
                                profile.browser, profile.profile
                            ),
                            current: i as u32 + 1,
                            total,
                            memories_created: result.memories_created,
                        },
                    );
                    tokio::task::yield_now().await;
                }
            }

            // Import search terms in batches of 20
            if !search_terms.is_empty() {
                let batches: Vec<&[String]> = search_terms.chunks(20).collect();
                for (bi, batch) in batches.iter().enumerate() {
                    result.items_scanned += batch.len() as u32;

                    let batch_text = batch.join(", ");
                    let batch_hash = deterministic_hash(&batch_text);
                    let memory_id = format!(
                        "bootstrap::chromium::searches::{}::{}",
                        profile.browser, batch_hash
                    );

                    let content = format!(
                        "[Search terms from {}] {}",
                        profile.browser, batch_text
                    );

                    match store_as_memory(state, &memory_id, &content, "episodic", 0.5).await {
                        Ok(true) => result.memories_created += 1,
                        Ok(false) => result.skipped_existing += 1,
                        Err(e) => {
                            tracing::debug!("Bootstrap chromium search error: {}", e);
                            result.errors += 1;
                        }
                    }

                    if bi % 5 == 0 {
                        tokio::task::yield_now().await;
                    }
                }
            }
        }

        // Clean up all temp databases
        for temp in all_temp_dbs {
            let _ = std::fs::remove_file(&temp);
        }

        tracing::info!(
            "Bootstrap chromium complete: {} created, {} skipped",
            result.memories_created,
            result.skipped_existing
        );
        result
    }
}
