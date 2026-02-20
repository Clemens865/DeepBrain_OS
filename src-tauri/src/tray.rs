//! System tray management for DeepBrain

use tauri::{
    image::Image,
    menu::{Menu, MenuItem},
    tray::{MouseButton, MouseButtonState, TrayIconBuilder, TrayIconEvent},
    AppHandle, Emitter, Manager,
};

/// Tray icon status variants
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TrayStatus {
    Idle,     // green
    Thinking, // yellow
    Learning, // blue
}

/// Set up the system tray icon and menu
pub fn setup_tray(app: &AppHandle) -> Result<(), Box<dyn std::error::Error>> {
    let show = MenuItem::with_id(app, "show", "Show DeepBrain", true, None::<&str>)?;
    let pin_label = if crate::overlay::is_pinned() { "Unpin Window" } else { "Pin Window" };
    let pin = MenuItem::with_id(app, "pin", pin_label, true, None::<&str>)?;
    let status = MenuItem::with_id(app, "status", "Status: Running", false, None::<&str>)?;
    let separator = MenuItem::with_id(app, "sep1", "---", false, None::<&str>)?;
    let settings = MenuItem::with_id(app, "settings", "Settings...", true, None::<&str>)?;
    let quit = MenuItem::with_id(app, "quit", "Quit DeepBrain", true, None::<&str>)?;

    let menu = Menu::with_items(app, &[&show, &pin, &status, &separator, &settings, &quit])?;

    let pin_item = pin.clone();
    let _tray = TrayIconBuilder::with_id("main-tray")
        .menu(&menu)
        .tooltip("DeepBrain - Cognitive Assistant")
        .icon(make_status_icon(TrayStatus::Idle))
        .on_menu_event(move |app, event| {
            match event.id.as_ref() {
                "show" => {
                    toggle_overlay(app);
                }
                "pin" => {
                    let new_state = !crate::overlay::is_pinned();
                    crate::overlay::set_pinned(new_state);
                    let label = if new_state { "Unpin Window" } else { "Pin Window" };
                    let _ = pin_item.set_text(label);
                    // Notify frontend
                    let _ = app.emit("pin-changed", new_state);
                }
                "settings" => {
                    if let Some(window) = app.get_webview_window("main") {
                        let _ = window.show();
                        let _ = window.set_focus();
                        // Emit event to navigate to settings
                        let _ = window.emit("navigate", "settings");
                    }
                }
                "quit" => {
                    // Flush state before quitting
                    if let Some(state) = app.try_state::<std::sync::Arc<crate::state::AppState>>() {
                        let _ = state.flush();
                    }
                    app.exit(0);
                }
                _ => {}
            }
        })
        .on_tray_icon_event(|tray, event| {
            if let TrayIconEvent::Click {
                button: MouseButton::Left,
                button_state: MouseButtonState::Up,
                ..
            } = event
            {
                toggle_overlay(tray.app_handle());
            }
        })
        .build(app)?;

    Ok(())
}

/// Update the tray icon to reflect current status
pub fn set_status(app: &AppHandle, status: TrayStatus) {
    if let Some(tray) = app.tray_by_id("main-tray") {
        let _ = tray.set_icon(Some(make_status_icon(status)));
        let tooltip = match status {
            TrayStatus::Idle => "DeepBrain - Idle",
            TrayStatus::Thinking => "DeepBrain - Thinking...",
            TrayStatus::Learning => "DeepBrain - Learning...",
        };
        let _ = tray.set_tooltip(Some(tooltip));
    }
}

/// Generate a 22x22 RGBA tray icon with a colored brain-dot indicator
fn make_status_icon(status: TrayStatus) -> Image<'static> {
    const SIZE: u32 = 22;
    let mut rgba = vec![0u8; (SIZE * SIZE * 4) as usize];

    let (r, g, b) = match status {
        TrayStatus::Idle => (64, 192, 87),     // green
        TrayStatus::Thinking => (250, 176, 5),  // yellow
        TrayStatus::Learning => (124, 92, 252),  // blue/purple (accent)
    };

    let cx = SIZE as f32 / 2.0;
    let cy = SIZE as f32 / 2.0;
    let outer_r = 9.0f32;
    let inner_r = 6.0f32;

    for y in 0..SIZE {
        for x in 0..SIZE {
            let dx = x as f32 - cx;
            let dy = y as f32 - cy;
            let dist = (dx * dx + dy * dy).sqrt();
            let offset = ((y * SIZE + x) * 4) as usize;

            if dist <= inner_r {
                // Solid color center
                rgba[offset] = r;
                rgba[offset + 1] = g;
                rgba[offset + 2] = b;
                rgba[offset + 3] = 230;
            } else if dist <= outer_r {
                // Antialiased edge
                let alpha = ((outer_r - dist) / (outer_r - inner_r) * 180.0) as u8;
                rgba[offset] = r;
                rgba[offset + 1] = g;
                rgba[offset + 2] = b;
                rgba[offset + 3] = alpha;
            }
        }
    }

    Image::new_owned(rgba, SIZE, SIZE)
}

/// Toggle the overlay window visibility
fn toggle_overlay(app: &AppHandle) {
    if let Some(window) = app.get_webview_window("main") {
        if window.is_visible().unwrap_or(false) {
            let _ = window.hide();
        } else {
            let _ = window.show();
            let _ = window.set_focus();
            let _ = window.center();
        }
    }
}
