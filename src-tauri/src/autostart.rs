//! Auto-start on login for macOS
//!
//! Manages a LaunchAgent plist to start SuperBrain at login.

use std::path::PathBuf;

const PLIST_LABEL: &str = "com.superbrain.app";

/// Get the LaunchAgents directory path
fn launch_agents_dir() -> Option<PathBuf> {
    dirs::home_dir().map(|h| h.join("Library").join("LaunchAgents"))
}

/// Get the plist file path
fn plist_path() -> Option<PathBuf> {
    launch_agents_dir().map(|d| d.join(format!("{}.plist", PLIST_LABEL)))
}

/// Get the path to the current executable
fn app_executable() -> Option<String> {
    std::env::current_exe()
        .ok()
        .and_then(|p| p.to_str().map(|s| s.to_string()))
}

/// Enable or disable auto-start at login
pub fn set_auto_start(enabled: bool) -> Result<(), String> {
    let plist = plist_path().ok_or("Cannot determine LaunchAgents path")?;

    if enabled {
        let exe = app_executable().ok_or("Cannot determine executable path")?;
        let dir = launch_agents_dir().ok_or("Cannot determine LaunchAgents dir")?;

        std::fs::create_dir_all(&dir)
            .map_err(|e| format!("Failed to create LaunchAgents dir: {}", e))?;

        let content = format!(
            r#"<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>{}</string>
    <key>ProgramArguments</key>
    <array>
        <string>{}</string>
    </array>
    <key>RunAtLoad</key>
    <true/>
    <key>KeepAlive</key>
    <false/>
</dict>
</plist>"#,
            PLIST_LABEL, exe
        );

        std::fs::write(&plist, content)
            .map_err(|e| format!("Failed to write plist: {}", e))?;

        tracing::info!("Auto-start enabled: {}", plist.display());
    } else if plist.exists() {
        std::fs::remove_file(&plist)
            .map_err(|e| format!("Failed to remove plist: {}", e))?;

        tracing::info!("Auto-start disabled");
    }

    Ok(())
}
