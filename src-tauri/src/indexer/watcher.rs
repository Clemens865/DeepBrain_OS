//! File system watcher for DeepBrain
//!
//! Monitors directories for changes and triggers re-indexing.
//! Applies the same filtering rules as `collect_files_recursive` (SKIP_DIRS,
//! hidden files) plus additional macOS ~/Library noise filtering.

use std::path::{Path, PathBuf};

use notify::{Event, EventKind, RecommendedWatcher, RecursiveMode, Watcher};
use tokio::sync::mpsc;

use super::parser;

/// File change event
#[derive(Debug, Clone)]
pub enum FileChange {
    Created(PathBuf),
    Modified(PathBuf),
    Deleted(PathBuf),
}

/// Directories to skip — mirrors SKIP_DIRS from mod.rs and adds more.
/// Matched against every path component.
const SKIP_DIRS: &[&str] = &[
    "node_modules",
    "target",
    ".git",
    ".svn",
    ".hg",
    "__pycache__",
    ".venv",
    "venv",
    ".cache",
    "build",
    "dist",
    ".Trash",
    ".npm",
    ".cargo",
    ".rustup",
    "Pods",
    ".next",
    ".nuxt",
    "vendor",
    "bower_components",
];

/// ~/Library subdirectories that are noisy system internals.
const LIBRARY_SKIP_DIRS: &[&str] = &[
    "Caches",
    "Logs",
    "WebKit",
    "Metadata",
    "Cookies",
    "HTTPStorages",
    "Saved Application State",
    "Group Containers",
    "KeyboardServices",
    "Accounts",
    "Assistant",
    "Biome",
    "DES",
    "Trial",
    "StatusKit",
    "IntelligencePlatform",
    "Daemon Containers",
    "Developer",
];

/// Application Support sub-paths to skip (internal databases, caches).
const APP_SUPPORT_SKIP: &[&str] = &[
    "FileProvider",
    "Comet",
    "Antigravity",
    "AddressBook",
    "CallHistoryDB",
    "Knowledge",
    "CrashReporter",
    "Google",
    "CloudDocs",
];

/// Maximum file size to index (1 MB). Larger files are skipped.
const MAX_FILE_SIZE: u64 = 1024 * 1024;

/// Check whether a path should be indexed or silently skipped.
///
/// This is the single source of truth for watcher filtering, applying:
/// 1. Extension check (only supported file types)
/// 2. Hidden file/dir check (dotfiles)
/// 3. SKIP_DIRS check (node_modules, target, .git, etc.)
/// 4. ~/Library noise filtering
/// 5. File size check
pub fn should_index(path: &Path) -> bool {
    // 1. Only supported file extensions
    let ext = path
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or("")
        .to_lowercase();
    if !parser::is_supported(&ext) {
        return false;
    }

    // 2. Skip hidden files (filename starts with '.')
    if let Some(name) = path.file_name().and_then(|n| n.to_str()) {
        if name.starts_with('.') {
            return false;
        }
    }

    // 3. Skip paths containing any SKIP_DIRS component
    for component in path.components() {
        if let std::path::Component::Normal(name) = component {
            let name_str = name.to_string_lossy();
            if name_str.starts_with('.') {
                return false;
            }
            if SKIP_DIRS.contains(&name_str.as_ref()) {
                return false;
            }
        }
    }

    // 4. ~/Library noise filtering
    let path_str = path.to_string_lossy();
    if path_str.contains("/Library/") {
        // Allow iCloud Drive
        if path_str.contains("/Mobile Documents/") {
            // But still apply SKIP_DIRS check (already done above)
        } else {
            // Block noisy ~/Library subdirectories
            for skip in LIBRARY_SKIP_DIRS {
                if path_str.contains(&format!("/Library/{}/", skip)) {
                    return false;
                }
            }
            for skip in APP_SUPPORT_SKIP {
                if path_str.contains(&format!("/Application Support/{}/", skip)) {
                    return false;
                }
            }
            // Block DeepBrain's own data files
            if path_str.contains("/Application Support/DeepBrain/") {
                return false;
            }
            // Block database files under Library
            if ext == "db" {
                return false;
            }
        }
    }

    // 5. Skip large files (avoid indexing multi-MB binaries/logs)
    if let Ok(meta) = std::fs::metadata(path) {
        if meta.len() > MAX_FILE_SIZE {
            return false;
        }
    }

    true
}

/// Start watching directories for changes
/// Returns a channel receiver that emits file change events
pub fn start_watcher(
    dirs: Vec<PathBuf>,
) -> Result<(RecommendedWatcher, mpsc::UnboundedReceiver<FileChange>), String> {
    let (tx, rx) = mpsc::unbounded_channel();

    let mut watcher = notify::recommended_watcher(move |res: Result<Event, notify::Error>| {
        if let Ok(event) = res {
            let paths: Vec<PathBuf> = event
                .paths
                .into_iter()
                .filter(|p| p.is_file() && should_index(p))
                .collect();

            for path in paths {
                let change = match event.kind {
                    EventKind::Create(_) => FileChange::Created(path),
                    EventKind::Modify(_) => FileChange::Modified(path),
                    EventKind::Remove(_) => FileChange::Deleted(path),
                    _ => continue,
                };
                let _ = tx.send(change);
            }
        }
    })
    .map_err(|e| format!("Failed to create watcher: {}", e))?;

    for dir in &dirs {
        if dir.exists() {
            watcher
                .watch(dir, RecursiveMode::Recursive)
                .map_err(|e| format!("Failed to watch {:?}: {}", dir, e))?;
            tracing::info!("Watching directory: {:?}", dir);
        }
    }

    Ok((watcher, rx))
}

/// Get default directories to watch
pub fn default_watch_dirs() -> Vec<PathBuf> {
    let home = dirs::home_dir().unwrap_or_else(|| PathBuf::from("/"));

    vec![
        home.join("Documents"),
        home.join("Desktop"),
        home.join("Downloads"),
    ]
    .into_iter()
    .filter(|p| p.exists())
    .collect()
}
