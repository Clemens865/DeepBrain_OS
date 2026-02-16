// Prevents additional console window on Windows in release
#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]
#![allow(dead_code)]

mod ai;
mod autostart;
mod brain;
mod commands;
mod context;
mod indexer;
mod keychain;
mod overlay;
mod state;
mod tray;
mod workflows;

use state::AppState;
use tauri::Manager;

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "superbrain_app=info".into()),
        )
        .init();

    tracing::info!("SuperBrain starting...");

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

            // Initialize application state
            let app_state = AppState::new().expect("Failed to initialize SuperBrain");

            // Try to initialize Ollama embeddings in background
            let embeddings = app_state.embeddings.clone();
            tauri::async_runtime::spawn(async move {
                embeddings.try_init_ollama().await;
            });

            app.manage(app_state);

            // Setup system tray
            tray::setup_tray(app.handle())?;

            // Setup global shortcut (Cmd+Shift+Space)
            use tauri_plugin_global_shortcut::{GlobalShortcutExt, Shortcut};
            let shortcut: Shortcut = "CmdOrCtrl+Shift+Space"
                .parse()
                .expect("Failed to parse shortcut");

            let handle = app.handle().clone();
            app.global_shortcut().on_shortcut(shortcut, move |_app, _shortcut, _event| {
                overlay::toggle(&handle);
            })?;

            // Start file watcher for indexed directories
            let indexer_ref = app.state::<AppState>().indexer.clone();
            let watch_dirs = indexer::watcher::default_watch_dirs();
            indexer_ref.add_watch_dirs(watch_dirs.clone());
            match indexer::watcher::start_watcher(watch_dirs) {
                Ok((_watcher, mut rx)) => {
                    let idx = indexer_ref;
                    tauri::async_runtime::spawn(async move {
                        // Keep _watcher alive by moving it into the task
                        let _keep_alive = _watcher;
                        while let Some(change) = rx.recv().await {
                            let path = match &change {
                                indexer::watcher::FileChange::Created(p)
                                | indexer::watcher::FileChange::Modified(p) => Some(p.clone()),
                                indexer::watcher::FileChange::Deleted(_) => None,
                            };
                            if let Some(path) = path {
                                tracing::debug!("File changed, re-indexing: {:?}", path);
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
                .state::<AppState>()
                .engine
                .clone();
            let persistence = app
                .state::<AppState>()
                .persistence
                .clone();
            let cycle_handle = app.handle().clone();

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

                    tray::set_status(&cycle_handle, tray::TrayStatus::Idle);
                }
            });

            // Start clipboard monitoring (poll every 2s)
            let context_ref = app.state::<AppState>().context.clone();
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

            // Start overlay hidden
            if let Some(window) = app.get_webview_window("main") {
                let _ = window.hide();
            }

            tracing::info!("SuperBrain initialized successfully");
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
            commands::flush,
        ])
        .run(tauri::generate_context!())
        .expect("Error while running SuperBrain");
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
