//! Activity Observer for DeepBrain
//!
//! Passively tracks the frontmost application and window title on macOS,
//! storing events in both an in-memory ring buffer (fast queries) and
//! SQLite (persistent history). This enables DeepNote to be context-aware
//! of what the user is currently doing.

use std::collections::VecDeque;
use std::path::{Path, PathBuf};

use parking_lot::{Mutex, RwLock};
use rusqlite::params;
use serde::{Deserialize, Serialize};

use crate::brain::utils::now_millis;

// ---------------------------------------------------------------------------
// Data model
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ActivityEvent {
    pub id: String,
    pub timestamp: i64,
    pub event_type: String,
    pub app_name: Option<String>,
    pub window_title: Option<String>,
    pub file_path: Option<String>,
    pub url: Option<String>,
    pub content_preview: Option<String>,
    pub metadata: Option<String>,
    pub project: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ActivitySettings {
    pub enabled: bool,
    pub excluded_apps: Vec<String>,
    pub track_browser: bool,
    pub track_terminal: bool,
    pub retention_days: u32,
}

impl Default for ActivitySettings {
    fn default() -> Self {
        Self {
            enabled: true,
            excluded_apps: vec![
                "1Password".to_string(),
                "Keychain Access".to_string(),
            ],
            track_browser: true,
            track_terminal: false,
            retention_days: 30,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RecentFile {
    pub path: String,
    pub timestamp: i64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CurrentActivity {
    pub active_app: Option<String>,
    pub window_title: Option<String>,
    pub project: Option<String>,
    pub idle_seconds: u64,
    pub recent_files: Vec<RecentFile>,
    pub recent_clipboard: Option<String>,
}

// ---------------------------------------------------------------------------
// ActivityObserver
// ---------------------------------------------------------------------------

const RING_BUFFER_CAPACITY: usize = 1000;

pub struct ActivityObserver {
    db: Mutex<rusqlite::Connection>,
    ring_buffer: RwLock<VecDeque<ActivityEvent>>,
    settings: RwLock<ActivitySettings>,
    last_app: RwLock<Option<String>>,
    last_title: RwLock<Option<String>>,
    last_event_time: RwLock<i64>,
}

impl ActivityObserver {
    /// Create a new ActivityObserver. Opens (or creates) `activity.db` in the
    /// given data directory.
    pub fn new(data_dir: &Path) -> Result<Self, String> {
        std::fs::create_dir_all(data_dir)
            .map_err(|e| format!("Failed to create data dir: {}", e))?;

        let db_path = data_dir.join("activity.db");
        let conn = rusqlite::Connection::open(&db_path)
            .map_err(|e| format!("Failed to open activity.db: {}", e))?;

        // WAL mode for concurrent reads
        conn.execute_batch("PRAGMA journal_mode=WAL;")
            .map_err(|e| format!("Failed to set WAL mode: {}", e))?;

        conn.execute_batch(
            "CREATE TABLE IF NOT EXISTS activity_events (
                id TEXT PRIMARY KEY,
                timestamp INTEGER NOT NULL,
                event_type TEXT NOT NULL,
                app_name TEXT,
                window_title TEXT,
                file_path TEXT,
                url TEXT,
                content_preview TEXT,
                metadata TEXT,
                project TEXT
            );
            CREATE INDEX IF NOT EXISTS idx_activity_timestamp ON activity_events(timestamp);
            CREATE INDEX IF NOT EXISTS idx_activity_event_type ON activity_events(event_type);
            CREATE INDEX IF NOT EXISTS idx_activity_project ON activity_events(project);",
        )
        .map_err(|e| format!("Failed to create activity tables: {}", e))?;

        tracing::info!("Activity observer ready at {:?}", db_path);

        Ok(Self {
            db: Mutex::new(conn),
            ring_buffer: RwLock::new(VecDeque::with_capacity(RING_BUFFER_CAPACITY)),
            settings: RwLock::new(ActivitySettings::default()),
            last_app: RwLock::new(None),
            last_title: RwLock::new(None),
            last_event_time: RwLock::new(0),
        })
    }

    // ---- Settings ----

    pub fn is_enabled(&self) -> bool {
        self.settings.read().enabled
    }

    pub fn retention_days(&self) -> u32 {
        self.settings.read().retention_days
    }

    pub fn update_settings(&self, new: ActivitySettings) {
        *self.settings.write() = new;
    }

    pub fn settings(&self) -> ActivitySettings {
        self.settings.read().clone()
    }

    // ---- Observation ----

    /// Poll the frontmost application via AppleScript (System Events).
    /// Records an `app_switch` event when the app or window title changes.
    pub async fn observe_frontmost(&self) {
        let result = tokio::process::Command::new("osascript")
            .arg("-e")
            .arg(
                r#"tell application "System Events"
    set frontApp to name of first application process whose frontmost is true
    set frontWindow to ""
    try
        set frontWindow to name of front window of first application process whose frontmost is true
    end try
end tell
return frontApp & "\n" & frontWindow"#,
            )
            .output()
            .await;

        let output = match result {
            Ok(o) if o.status.success() => String::from_utf8_lossy(&o.stdout).to_string(),
            Ok(o) => {
                tracing::debug!(
                    "osascript stderr: {}",
                    String::from_utf8_lossy(&o.stderr)
                );
                return;
            }
            Err(e) => {
                tracing::debug!("osascript failed: {}", e);
                return;
            }
        };

        let mut lines = output.trim().splitn(2, '\n');
        let app_name = lines.next().unwrap_or("").trim().to_string();
        let window_title = lines.next().unwrap_or("").trim().to_string();

        if app_name.is_empty() {
            return;
        }

        // Check if anything changed
        let prev_app = self.last_app.read().clone();
        let prev_title = self.last_title.read().clone();

        let app_changed = prev_app.as_deref() != Some(&app_name);
        let title_changed = prev_title.as_deref() != Some(&window_title);

        if !app_changed && !title_changed {
            // Update last event time for idle tracking
            *self.last_event_time.write() = now_millis();
            return;
        }

        // Update tracked state
        *self.last_app.write() = Some(app_name.clone());
        *self.last_title.write() = Some(window_title.clone());
        *self.last_event_time.write() = now_millis();

        // Check exclusions
        let settings = self.settings.read().clone();
        let is_excluded = settings
            .excluded_apps
            .iter()
            .any(|ex| app_name.eq_ignore_ascii_case(ex));

        let effective_title = if is_excluded {
            "[private]".to_string()
        } else {
            window_title.clone()
        };

        // Detect project from window title
        let project = if is_excluded {
            None
        } else {
            detect_project_from_title(&effective_title)
        };

        // Detect file path from window title
        let file_path = if is_excluded {
            None
        } else {
            extract_file_path(&effective_title)
        };

        // Capture browser URL when Comet is frontmost
        let url = if !is_excluded && app_name == "Comet" {
            get_browser_url(&app_name).await
        } else {
            None
        };

        let event = ActivityEvent {
            id: uuid::Uuid::new_v4().to_string(),
            timestamp: now_millis(),
            event_type: "app_switch".to_string(),
            app_name: Some(app_name),
            window_title: Some(effective_title),
            file_path,
            url,
            content_preview: None,
            metadata: None,
            project,
        };

        self.record_event(event);
    }

    // ---- Storage ----

    /// Record an event into both the ring buffer and SQLite.
    fn record_event(&self, event: ActivityEvent) {
        // Ring buffer
        {
            let mut buf = self.ring_buffer.write();
            buf.push_front(event.clone());
            if buf.len() > RING_BUFFER_CAPACITY {
                buf.pop_back();
            }
        }

        // SQLite
        let db = self.db.lock();
        if let Err(e) = db.execute(
            "INSERT OR IGNORE INTO activity_events
                (id, timestamp, event_type, app_name, window_title, file_path, url, content_preview, metadata, project)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10)",
            params![
                event.id,
                event.timestamp,
                event.event_type,
                event.app_name,
                event.window_title,
                event.file_path,
                event.url,
                event.content_preview,
                event.metadata,
                event.project,
            ],
        ) {
            tracing::warn!("Failed to insert activity event: {}", e);
        }
    }

    // ---- Queries ----

    /// Return the current activity state (most recent event + idle estimate + recent files).
    pub fn current(&self) -> CurrentActivity {
        let buf = self.ring_buffer.read();
        let latest = buf.front();

        let now = now_millis();
        let last_time = *self.last_event_time.read();
        let idle_seconds = if last_time > 0 {
            ((now - last_time).max(0) / 1000) as u64
        } else {
            0
        };

        // Collect recent files from ring buffer events that have file_path
        let recent_files: Vec<RecentFile> = buf
            .iter()
            .filter_map(|e| {
                e.file_path.as_ref().map(|fp| RecentFile {
                    path: fp.clone(),
                    timestamp: e.timestamp,
                })
            })
            .take(10)
            .collect();

        CurrentActivity {
            active_app: latest.and_then(|e| e.app_name.clone()),
            window_title: latest.and_then(|e| e.window_title.clone()),
            project: latest.and_then(|e| e.project.clone()),
            idle_seconds,
            recent_files,
            recent_clipboard: None, // Filled by API layer from ContextManager
        }
    }

    /// Query activity events from SQLite with optional filters.
    pub fn stream(
        &self,
        since_ms: i64,
        until_ms: Option<i64>,
        types: Option<&[String]>,
        project: Option<&str>,
        limit: u32,
    ) -> Vec<ActivityEvent> {
        let db = self.db.lock();

        let mut sql = String::from(
            "SELECT id, timestamp, event_type, app_name, window_title, file_path, url, content_preview, metadata, project
             FROM activity_events WHERE timestamp >= ?1",
        );
        let mut param_idx = 2u32;

        if until_ms.is_some() {
            sql.push_str(&format!(" AND timestamp <= ?{}", param_idx));
            param_idx += 1;
        }

        if let Some(types) = types {
            if !types.is_empty() {
                let placeholders: Vec<String> = types
                    .iter()
                    .enumerate()
                    .map(|(i, _)| format!("?{}", param_idx + i as u32))
                    .collect();
                sql.push_str(&format!(
                    " AND event_type IN ({})",
                    placeholders.join(", ")
                ));
                param_idx += types.len() as u32;
            }
        }

        if project.is_some() {
            sql.push_str(&format!(" AND project = ?{}", param_idx));
            #[allow(unused_assignments)]
            { param_idx += 1; }
        }

        sql.push_str(" ORDER BY timestamp DESC LIMIT ?");

        // Build parameter vector dynamically using rusqlite::types::Value
        let mut values: Vec<rusqlite::types::Value> = Vec::new();
        values.push(rusqlite::types::Value::Integer(since_ms));

        if let Some(until) = until_ms {
            values.push(rusqlite::types::Value::Integer(until));
        }

        if let Some(types) = types {
            for t in types {
                values.push(rusqlite::types::Value::Text(t.clone()));
            }
        }

        if let Some(proj) = project {
            values.push(rusqlite::types::Value::Text(proj.to_string()));
        }

        values.push(rusqlite::types::Value::Integer(limit as i64));

        let params_ref: Vec<&dyn rusqlite::types::ToSql> =
            values.iter().map(|v| v as &dyn rusqlite::types::ToSql).collect();

        let mut stmt = match db.prepare(&sql) {
            Ok(s) => s,
            Err(e) => {
                tracing::warn!("Failed to prepare activity stream query: {}", e);
                return Vec::new();
            }
        };

        let rows = stmt
            .query_map(params_ref.as_slice(), |row| {
                Ok(ActivityEvent {
                    id: row.get(0)?,
                    timestamp: row.get(1)?,
                    event_type: row.get(2)?,
                    app_name: row.get(3)?,
                    window_title: row.get(4)?,
                    file_path: row.get(5)?,
                    url: row.get(6)?,
                    content_preview: row.get(7)?,
                    metadata: row.get(8)?,
                    project: row.get(9)?,
                })
            })
            .ok();

        match rows {
            Some(mapped) => mapped.filter_map(|r| r.ok()).collect(),
            None => Vec::new(),
        }
    }

    // ---- Projects ----

    /// Query distinct projects from activity events.
    pub fn projects(&self) -> Vec<ProjectInfo> {
        let db = self.db.lock();
        let mut stmt = match db.prepare(
            "SELECT project, MAX(timestamp) as last_ts
             FROM activity_events
             WHERE project IS NOT NULL
             GROUP BY project
             ORDER BY last_ts DESC",
        ) {
            Ok(s) => s,
            Err(e) => {
                tracing::warn!("Failed to query activity projects: {}", e);
                return Vec::new();
            }
        };

        let rows = stmt
            .query_map([], |row| {
                let name: String = row.get(0)?;
                let last_ts: i64 = row.get(1)?;
                Ok((name, last_ts))
            })
            .ok();

        match rows {
            Some(mapped) => mapped
                .filter_map(|r| r.ok())
                .map(|(name, last_ts)| {
                    let last_active = chrono::DateTime::from_timestamp_millis(last_ts)
                        .map(|dt| dt.format("%Y-%m-%dT%H:%M:%SZ").to_string());
                    ProjectInfo {
                        name,
                        path: None,
                        last_active,
                        indexed_files: 0,
                        indexed_chunks: 0,
                    }
                })
                .collect(),
            None => Vec::new(),
        }
    }

    // ---- Maintenance ----

    /// Delete events older than `retention_days` from SQLite.
    pub fn prune(&self, retention_days: u32) {
        let cutoff_ms = now_millis() - (retention_days as i64 * 24 * 60 * 60 * 1000);
        let db = self.db.lock();
        match db.execute(
            "DELETE FROM activity_events WHERE timestamp < ?1",
            params![cutoff_ms],
        ) {
            Ok(deleted) => {
                if deleted > 0 {
                    tracing::info!(
                        "Pruned {} activity events older than {} days",
                        deleted,
                        retention_days
                    );
                }
            }
            Err(e) => {
                tracing::warn!("Failed to prune activity events: {}", e);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Project detection helpers
// ---------------------------------------------------------------------------

/// Try to detect a project name from a window title.
///
/// Common patterns:
/// - "auth.ts — Notebook_LLM"  (VS Code / editors with em-dash)
/// - "main.rs - DeepBrain [~/Projects/deepbrain]"
/// - "~/Projects/myapp/src/foo.rs"
fn detect_project_from_title(title: &str) -> Option<String> {
    // Pattern: "filename — ProjectName" or "filename - ProjectName"
    for sep in &[" \u{2014} ", " - ", " \u{2013} "] {
        if let Some(idx) = title.rfind(sep) {
            let right = title[idx + sep.len()..].trim();
            // Take the first word/token (strip brackets, paths)
            let project = right
                .split(|c: char| c == '[' || c == '(' || c == ' ')
                .next()
                .unwrap_or(right)
                .trim();
            if !project.is_empty() && project.len() < 80 {
                return Some(project.to_string());
            }
        }
    }

    // Pattern: path-like title containing a project directory
    if title.contains('/') {
        if let Some(proj) = detect_project_from_path(title.trim()) {
            return Some(proj);
        }
    }

    None
}

/// Walk up from a file path looking for project markers.
pub fn detect_project_from_path(path_str: &str) -> Option<String> {
    // Clean up path (may start with ~)
    let expanded = if path_str.starts_with('~') {
        if let Some(home) = dirs::home_dir() {
            path_str.replacen('~', &home.to_string_lossy(), 1)
        } else {
            path_str.to_string()
        }
    } else {
        path_str.to_string()
    };

    let path = PathBuf::from(&expanded);
    let markers = [
        ".git",
        "Cargo.toml",
        "package.json",
        "pyproject.toml",
        ".xcodeproj",
        "go.mod",
    ];

    // Walk up from the path (or its parent if it's a file)
    let mut current = if path.is_file() {
        path.parent().map(|p| p.to_path_buf())
    } else {
        Some(path.clone())
    };

    while let Some(dir) = current {
        for marker in &markers {
            let candidate = dir.join(marker);
            // Check for both files and directories (e.g. .git is a dir, .xcodeproj is a dir)
            if candidate.exists() {
                return dir
                    .file_name()
                    .map(|n| n.to_string_lossy().to_string());
            }
        }
        current = dir.parent().map(|p| p.to_path_buf());
        // Stop at home directory to avoid scanning too far up
        if let Some(home) = dirs::home_dir() {
            if current.as_deref() == Some(home.as_path()) {
                break;
            }
        }
    }

    None
}

/// Fetch the current URL from a Chromium-based browser via AppleScript.
async fn get_browser_url(app_name: &str) -> Option<String> {
    let script = match app_name {
        "Comet" => r#"tell application "Comet" to return URL of active tab of window 1"#,
        _ => return None,
    };
    let output = tokio::process::Command::new("osascript")
        .arg("-e")
        .arg(script)
        .output()
        .await
        .ok()?;
    if output.status.success() {
        let url = String::from_utf8_lossy(&output.stdout).trim().to_string();
        if url.starts_with("http") {
            Some(url)
        } else {
            None
        }
    } else {
        None
    }
}

/// Extract a file path from a window title if one is present.
fn extract_file_path(title: &str) -> Option<String> {
    // Look for path-like segments (starting with / or ~/)
    for token in title.split_whitespace() {
        let clean = token.trim_matches(|c: char| c == '[' || c == ']' || c == '(' || c == ')');
        if (clean.starts_with('/') || clean.starts_with("~/")) && clean.len() > 2 {
            return Some(clean.to_string());
        }
    }
    None
}

// ---------------------------------------------------------------------------
// Shared types
// ---------------------------------------------------------------------------

/// Aggregated project info, used by both ActivityObserver and FileIndexer.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProjectInfo {
    pub name: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub path: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub last_active: Option<String>,
    pub indexed_files: u32,
    pub indexed_chunks: u32,
}
