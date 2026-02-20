//! Overlay window management for DeepBrain

use std::sync::atomic::{AtomicBool, AtomicI64, Ordering};
use tauri::{AppHandle, Emitter, Manager};

/// Timestamp (ms) of the last show() call — used to debounce blur events
static LAST_SHOW_MS: AtomicI64 = AtomicI64::new(0);

/// Whether the window is pinned (should not auto-hide on blur)
static PINNED: AtomicBool = AtomicBool::new(false);

/// Minimum time (ms) the window must be visible before blur can hide it.
/// This prevents the global-shortcut key-release from immediately dismissing
/// the overlay on macOS.
const BLUR_DEBOUNCE_MS: i64 = 300;

/// Toggle the overlay window
pub fn toggle(app: &AppHandle) {
    if let Some(window) = app.get_webview_window("main") {
        if window.is_visible().unwrap_or(false) {
            hide(app);
        } else {
            show(app);
        }
    }
}

/// Show the overlay window
pub fn show(app: &AppHandle) {
    if let Some(window) = app.get_webview_window("main") {
        let _ = window.center();
        let _ = window.show();
        let _ = window.set_focus();
        LAST_SHOW_MS.store(now_ms(), Ordering::Relaxed);
        let _ = window.emit("overlay-shown", ());
    }
}

/// Hide the overlay window
pub fn hide(app: &AppHandle) {
    if let Some(window) = app.get_webview_window("main") {
        let _ = window.hide();
        let _ = window.emit("overlay-hidden", ());
    }
}

/// Returns true if the window is pinned (should not auto-hide on blur).
pub fn is_pinned() -> bool {
    PINNED.load(Ordering::Relaxed)
}

/// Set the pinned state.
pub fn set_pinned(pinned: bool) {
    PINNED.store(pinned, Ordering::Relaxed);
}

/// Returns true if enough time has passed since the last show() that a blur
/// event should be honoured, and the window is not pinned.
/// Called from the `on_window_event` handler.
pub fn should_hide_on_blur() -> bool {
    if is_pinned() {
        return false;
    }
    let shown_at = LAST_SHOW_MS.load(Ordering::Relaxed);
    now_ms() - shown_at > BLUR_DEBOUNCE_MS
}

fn now_ms() -> i64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_millis() as i64)
        .unwrap_or(0)
}
