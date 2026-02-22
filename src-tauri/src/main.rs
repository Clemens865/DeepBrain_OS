// Prevents additional console window on Windows in release
#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]
#![allow(dead_code)]

mod activity;
mod ai;
mod api;
mod autostart;
mod brain;
mod commands;
mod context;
mod deepbrain;
mod indexer;
mod keychain;
mod overlay;
mod state;
mod tray;
mod workflows;

use std::sync::Arc;
use state::AppState;
use tauri::Manager;

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "deepbrain_app=info".into()),
        )
        .init();

    tracing::info!("DeepBrain starting...");

    tauri::Builder::default()
        .plugin(tauri_plugin_global_shortcut::Builder::new().build())
        .plugin(tauri_plugin_shell::init())
        // Auto-update: uncomment after generating signing key with `cargo tauri signer generate`
        // .plugin(tauri_plugin_updater::Builder::new().build())
        .setup(|app| {
            // Set activation policy to accessory (menu bar only, no dock icon)
            #[cfg(target_os = "macos")]
            {
                app.set_activation_policy(tauri::ActivationPolicy::Accessory);
            }

            // Initialize application state (Arc-wrapped for sharing with HTTP API)
            let app_state = Arc::new(AppState::new().expect("Failed to initialize DeepBrain"));

            // Try to initialize Ollama embeddings in background
            let embeddings = app_state.embeddings.clone();
            tauri::async_runtime::spawn(async move {
                embeddings.try_init_ollama().await;
            });

            // Spawn HTTP API server (shares the same AppState)
            let api_state = Arc::clone(&app_state);
            let api_port = app_state.settings.read().api_port.unwrap_or(19519);
            tauri::async_runtime::spawn(async move {
                crate::api::start_api_server(api_state, api_port).await;
            });

            app.manage(app_state);

            // Setup system tray
            tray::setup_tray(app.handle())?;

            // Setup global shortcut (Cmd+Shift+Space)
            use tauri_plugin_global_shortcut::{GlobalShortcutExt, Shortcut, ShortcutState};
            let shortcut: Shortcut = "CmdOrCtrl+Shift+Space"
                .parse()
                .expect("Failed to parse shortcut");

            let handle = app.handle().clone();
            app.global_shortcut().on_shortcut(shortcut, move |_app, _shortcut, event| {
                // Only toggle on key press, not release.
                // Without this check, the window shows on press then immediately
                // hides on release, making it appear to "flash".
                if event.state == ShortcutState::Pressed {
                    overlay::toggle(&handle);
                }
            })?;

            // Start file watcher for indexed directories
            let indexer_ref = app.state::<Arc<AppState>>().indexer.clone();
            let custom_dirs: Vec<std::path::PathBuf> = app
                .state::<Arc<AppState>>()
                .settings
                .read()
                .indexed_folders
                .iter()
                .map(std::path::PathBuf::from)
                .filter(|p| p.exists())
                .collect();
            let watch_dirs = if custom_dirs.is_empty() {
                indexer::watcher::default_watch_dirs()
            } else {
                // Merge defaults with custom dirs
                let mut all = indexer::watcher::default_watch_dirs();
                for d in custom_dirs {
                    if !all.contains(&d) {
                        all.push(d);
                    }
                }
                all
            };
            indexer_ref.add_watch_dirs(watch_dirs.clone());
            match indexer::watcher::start_watcher(watch_dirs) {
                Ok((_watcher, mut rx)) => {
                    let idx = indexer_ref;
                    tauri::async_runtime::spawn(async move {
                        // Keep _watcher alive by moving it into the task
                        let _keep_alive = _watcher;

                        // Debounce: collect changed paths and flush after a quiet window.
                        // This coalesces bursts of events for the same file (macOS often
                        // emits 3-6 events per file change).
                        let mut pending = std::collections::HashSet::<std::path::PathBuf>::new();
                        let debounce = tokio::time::Duration::from_secs(2);

                        loop {
                            // Wait for the first event (blocking)
                            let change = match rx.recv().await {
                                Some(c) => c,
                                None => break,
                            };
                            if let indexer::watcher::FileChange::Created(p)
                                | indexer::watcher::FileChange::Modified(p) = change
                            {
                                pending.insert(p);
                            }

                            // Drain all events that arrive within the debounce window
                            loop {
                                match tokio::time::timeout(debounce, rx.recv()).await {
                                    Ok(Some(change)) => {
                                        if let indexer::watcher::FileChange::Created(p)
                                            | indexer::watcher::FileChange::Modified(p) = change
                                        {
                                            pending.insert(p);
                                        }
                                    }
                                    _ => break, // Timeout or channel closed
                                }
                            }

                            // Flush all unique pending paths
                            let batch: Vec<_> = pending.drain().collect();
                            tracing::debug!("Debounced watcher: indexing {} files", batch.len());
                            for path in batch {
                                let _ = idx.index_file(&path).await;
                            }
                        }
                    });
                    tracing::info!("File watcher started");
                }
                Err(e) => {
                    tracing::warn!("Failed to start file watcher: {}", e);
                }
            }

            // Start background cognitive cycle task (battery-aware)
            let engine = app
                .state::<Arc<AppState>>()
                .engine
                .clone();
            let persistence = app
                .state::<Arc<AppState>>()
                .persistence
                .clone();
            let vector_store = app
                .state::<Arc<AppState>>()
                .vector_store
                .clone();
            let cycle_handle = app.handle().clone();
            let rvtext_path = dirs::data_dir()
                .expect("No data dir")
                .join("DeepBrain")
                .join("knowledge.rvtext");

            tauri::async_runtime::spawn(async move {
                loop {
                    // Check battery state: use longer interval when on battery
                    let on_battery = is_on_battery();
                    let delay = if on_battery {
                        tracing::debug!("On battery — using 5min cycle interval");
                        tokio::time::Duration::from_secs(300)
                    } else {
                        tokio::time::Duration::from_secs(60)
                    };
                    tokio::time::sleep(delay).await;

                    // Show learning status
                    tray::set_status(&cycle_handle, tray::TrayStatus::Learning);

                    // Run a cognitive cycle
                    let _ = engine.cycle();
                    // Periodic flush
                    let nodes = engine.memory.all_nodes();
                    let _ = persistence.store_memories_batch(&nodes);
                    tracing::debug!("Background cycle completed (battery={})", on_battery);

                    // Auto-compaction: reclaim dead space when ratio exceeds 30%
                    let metrics = vector_store.metrics();
                    if metrics.dead_space_ratio > 0.3 {
                        match vector_store.compact() {
                            Ok(bytes) => tracing::info!("Auto-compaction reclaimed {} bytes", bytes),
                            Err(e) => tracing::warn!("Auto-compaction failed: {}", e),
                        }
                    }

                    // File size watchdog: force compaction if .rvtext exceeds 2 GB
                    if let Ok(meta) = std::fs::metadata(&rvtext_path) {
                        let size_gb = meta.len() as f64 / (1024.0 * 1024.0 * 1024.0);
                        if size_gb > 2.0 {
                            tracing::warn!("knowledge.rvtext is {:.1} GB — forcing compaction", size_gb);
                            match vector_store.compact() {
                                Ok(bytes) => tracing::info!("Emergency compaction reclaimed {} bytes", bytes),
                                Err(e) => tracing::error!("Emergency compaction failed: {}", e),
                            }
                            // Check if still over limit after compaction
                            if let Ok(meta) = std::fs::metadata(&rvtext_path) {
                                let size_gb = meta.len() as f64 / (1024.0 * 1024.0 * 1024.0);
                                if size_gb > 2.0 {
                                    tracing::error!(
                                        "knowledge.rvtext still {:.1} GB after compaction — manual intervention needed",
                                        size_gb
                                    );
                                }
                            }
                        }
                    }

                    tray::set_status(&cycle_handle, tray::TrayStatus::Idle);
                }
            });

            // Start clipboard monitoring (poll every 2s)
            let context_ref = app.state::<Arc<AppState>>().context.clone();
            tauri::async_runtime::spawn(async move {
                let mut last_clipboard = String::new();
                loop {
                    tokio::time::sleep(tokio::time::Duration::from_secs(2)).await;
                    if let Some(current) = get_clipboard_text() {
                        let trimmed = current.trim().to_string();
                        if !trimmed.is_empty() && trimmed != last_clipboard {
                            last_clipboard = trimmed.clone();
                            context_ref.record_clipboard(trimmed);
                            tracing::debug!("Clipboard captured");
                        }
                    }
                }
            });

            // Start activity observation loop (poll every 2s)
            let activity_ref = app.state::<Arc<AppState>>().activity.clone();
            tauri::async_runtime::spawn(async move {
                loop {
                    tokio::time::sleep(tokio::time::Duration::from_secs(2)).await;
                    if activity_ref.is_enabled() {
                        activity_ref.observe_frontmost().await;
                    }
                }
            });

            // Start hourly activity pruning task
            let activity_prune_ref = app.state::<Arc<AppState>>().activity.clone();
            tauri::async_runtime::spawn(async move {
                loop {
                    tokio::time::sleep(tokio::time::Duration::from_secs(3600)).await;
                    let retention = activity_prune_ref.retention_days();
                    activity_prune_ref.prune(retention);
                }
            });

            // Start background email indexing (battery-aware)
            let email_indexer = app
                .state::<Arc<AppState>>()
                .email_indexer
                .clone();
            let email_handle = app.handle().clone();

            tauri::async_runtime::spawn(async move {
                // Wait 30s before first pass to let the app fully settle
                tokio::time::sleep(tokio::time::Duration::from_secs(30)).await;
                tracing::info!("Email indexing background task started");

                loop {
                    tray::set_status(&email_handle, tray::TrayStatus::Learning);

                    match email_indexer.index_pass(50).await {
                        Ok(count) => {
                            if count > 0 {
                                tracing::info!("Email indexer: {} new emails indexed", count);
                            }
                        }
                        Err(e) => {
                            tracing::warn!("Email indexing error: {}", e);
                        }
                    }

                    tray::set_status(&email_handle, tray::TrayStatus::Idle);

                    // On battery: index every 15 min; plugged in: every 5 min
                    let on_battery = is_on_battery();
                    let delay = if on_battery {
                        tokio::time::Duration::from_secs(900)
                    } else {
                        tokio::time::Duration::from_secs(300)
                    };
                    tokio::time::sleep(delay).await;
                }
            });

            // One-time startup scan: index all existing files in fast local directories.
            // Only scans Documents, Desktop, Downloads — the watcher handles iCloud
            // and ~/ in real-time. Without this, existing content (md, code, configs)
            // would never be indexed until manually triggered or files change.
            let startup_indexer = app.state::<Arc<AppState>>().indexer.clone();
            let startup_handle = app.handle().clone();
            tauri::async_runtime::spawn(async move {
                // Wait 45s for the app and file watcher to fully settle
                tokio::time::sleep(tokio::time::Duration::from_secs(45)).await;
                tracing::info!("Startup file scan beginning...");
                tray::set_status(&startup_handle, tray::TrayStatus::Learning);

                // Only scan local dirs (fast). iCloud + ~/ are handled by watcher.
                let scan_dirs = indexer::watcher::default_watch_dirs();
                match startup_indexer.scan_dirs(&scan_dirs).await {
                    Ok(count) => {
                        if count > 0 {
                            tracing::info!("Startup scan complete: {} chunks indexed", count);
                        } else {
                            tracing::info!("Startup scan complete: all files already up to date");
                        }
                    }
                    Err(e) => tracing::warn!("Startup scan failed: {}", e),
                }

                tray::set_status(&startup_handle, tray::TrayStatus::Idle);
            });

            // Start background SONA learning tick
            let sona_ref = app.state::<Arc<AppState>>().sona.clone();
            let sona_persistence = app.state::<Arc<AppState>>().persistence.clone();
            tauri::async_runtime::spawn(async move {
                // Wait 2 minutes before first tick
                tokio::time::sleep(tokio::time::Duration::from_secs(120)).await;

                loop {
                    // Tick every 5 minutes — SONA internally checks its own interval
                    // (default 1 hour) and only runs pattern extraction when due
                    tokio::time::sleep(tokio::time::Duration::from_secs(300)).await;

                    if let Some(msg) = sona_ref.tick() {
                        tracing::info!("SONA background learning: {}", msg);

                        // Persist updated patterns after a learning cycle
                        let patterns = sona_ref.export_patterns();
                        if let Ok(bytes) = serde_json::to_vec(&patterns) {
                            let _ = sona_persistence.store_blob("sona_patterns", &bytes);
                        }
                    }
                }
            });

            // Background compression scan (every 24 hours, delayed 10 minutes after startup)
            let compress_engine = app.state::<Arc<AppState>>().engine.clone();
            let compress_compressor = app.state::<Arc<AppState>>().compressor.clone();
            tauri::async_runtime::spawn(async move {
                // Wait 10 minutes before first scan
                tokio::time::sleep(tokio::time::Duration::from_secs(600)).await;

                loop {
                    let nodes = compress_engine.memory.all_nodes();
                    let mem_stats = compress_engine.memory.stats();
                    let total_accesses = mem_stats.total_accesses as u64;

                    let memories: Vec<crate::deepbrain::compress_bridge::MemoryForCompression> = nodes
                        .iter()
                        .map(|n| crate::deepbrain::compress_bridge::MemoryForCompression {
                            id: n.id.clone(),
                            access_count: n.access_count,
                            vector_len: n.vector.len(),
                        })
                        .collect();

                    let stats = compress_compressor.scan(&memories, total_accesses);
                    tracing::info!(
                        "Compression scan: {} memories ({} hot, {} warm, {} cold), est. savings {:.1}%",
                        stats.total_memories,
                        stats.hot_count,
                        stats.warm_count,
                        stats.cold_count,
                        stats.estimated_savings_pct,
                    );

                    // Run every 24 hours
                    tokio::time::sleep(tokio::time::Duration::from_secs(86400)).await;
                }
            });

            // Start overlay hidden
            if let Some(window) = app.get_webview_window("main") {
                let _ = window.hide();
            }

            tracing::info!("DeepBrain initialized successfully");
            Ok(())
        })
        .on_window_event(|window, event| {
            // Hide window on blur (click outside), but ignore blur events
            // that fire immediately after show (caused by shortcut key release)
            if let tauri::WindowEvent::Focused(false) = event {
                if overlay::should_hide_on_blur() {
                    let _ = window.hide();
                }
            }
        })
        .invoke_handler(tauri::generate_handler![
            commands::think,
            commands::remember,
            commands::recall,
            commands::get_status,
            commands::get_settings,
            commands::update_settings,
            commands::get_thoughts,
            commands::get_stats,
            commands::evolve,
            commands::cycle,
            commands::search_files,
            commands::index_files,
            commands::run_workflow,
            commands::check_ollama,
            commands::get_clipboard_history,
            commands::add_indexed_folder,
            commands::flush,
            commands::list_memories,
            commands::get_memory,
            commands::delete_memory,
            commands::list_indexed_files,
            commands::update_memory,
            commands::get_file_chunks,
            commands::spotlight_search,
            commands::mail_search,
            commands::get_common_folders,
            commands::list_folders,
            commands::pick_folder,
            commands::get_recent_emails,
            commands::search_emails,
            commands::email_stats,
            commands::index_emails,
            commands::set_pinned,
            commands::get_pinned,
            commands::get_storage_metrics,
            commands::verify_storage,
            commands::migrate_from_v1,
            commands::load_local_model,
            commands::unload_local_model,
            commands::local_model_status,
            commands::get_sona_stats,
            commands::get_nervous_stats,
            commands::verify_all,
            commands::bootstrap_sona,
            commands::graph_stats,
            commands::graph_neighbors,
            commands::gnn_rerank,
            commands::gnn_save_weights,
            commands::compression_scan,
            commands::compression_stats,
        ])
        .run(tauri::generate_context!())
        .expect("Error while running DeepBrain");
}

fn main() {
    run();
}

/// Read the current clipboard text via macOS pasteboard
fn get_clipboard_text() -> Option<String> {
    std::process::Command::new("pbpaste")
        .output()
        .ok()
        .and_then(|out| {
            if out.status.success() {
                String::from_utf8(out.stdout).ok()
            } else {
                None
            }
        })
}

/// Check if the system is running on battery power
fn is_on_battery() -> bool {
    let manager = battery::Manager::new();
    match manager {
        Ok(manager) => {
            if let Some(Ok(bat)) = manager.batteries().ok().and_then(|mut b| b.next()) {
                use battery::State;
                matches!(bat.state(), State::Discharging)
            } else {
                false // No battery = desktop = always plugged in
            }
        }
        Err(_) => false,
    }
}
