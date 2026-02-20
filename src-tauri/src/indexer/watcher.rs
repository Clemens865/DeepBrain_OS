//! File system watcher for DeepBrain
//!
//! Monitors directories for changes and triggers re-indexing.

use std::path::PathBuf;

use notify::{Event, EventKind, RecommendedWatcher, RecursiveMode, Watcher};
use tokio::sync::mpsc;

/// File change event
#[derive(Debug, Clone)]
pub enum FileChange {
    Created(PathBuf),
    Modified(PathBuf),
    Deleted(PathBuf),
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
                .filter(|p| p.is_file())
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
