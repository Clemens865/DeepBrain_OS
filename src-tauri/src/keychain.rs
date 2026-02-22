//! macOS Keychain integration for secure credential storage
//!
//! Keychain access is DISABLED during development because each rebuild changes
//! the binary's code signature, causing macOS to show frozen authorization
//! dialogs.  Secrets are stored in files instead (see state.rs).
//!
//! To re-enable Keychain for production (code-signed) builds, restore the
//! `security_framework` calls guarded behind `#[cfg(feature = "keychain")]`.

/// Store a secret — no-op (Keychain disabled).
pub fn store_secret(_key: &str, _value: &str) -> Result<(), String> {
    Ok(())
}

/// Retrieve a secret — always returns None (Keychain disabled).
pub fn get_secret(_key: &str) -> Result<Option<String>, String> {
    Ok(None)
}

/// Delete a secret — no-op (Keychain disabled).
pub fn delete_secret(_key: &str) -> Result<(), String> {
    Ok(())
}
