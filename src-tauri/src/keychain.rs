//! macOS Keychain integration for secure credential storage

use security_framework::passwords::{
    delete_generic_password, get_generic_password, set_generic_password,
};

const SERVICE_NAME: &str = "DeepBrain";
const LEGACY_SERVICE_NAME: &str = "SuperBrain";

/// Store a secret in the macOS Keychain
pub fn store_secret(key: &str, value: &str) -> Result<(), String> {
    // Delete existing entry first (set_generic_password fails if it already exists)
    let _ = delete_generic_password(SERVICE_NAME, key);
    set_generic_password(SERVICE_NAME, key, value.as_bytes())
        .map_err(|e| format!("Keychain store failed: {}", e))
}

/// Retrieve a secret from the macOS Keychain.
/// Falls back to the legacy "SuperBrain" service name to migrate existing users.
pub fn get_secret(key: &str) -> Result<Option<String>, String> {
    match get_generic_password(SERVICE_NAME, key) {
        Ok(bytes) => {
            let value = String::from_utf8(bytes.to_vec())
                .map_err(|e| format!("Invalid UTF-8 in keychain: {}", e))?;
            Ok(Some(value))
        }
        Err(e) => {
            if e.code() == -25300 {
                // Not found under new name — try legacy service name
                match get_generic_password(LEGACY_SERVICE_NAME, key) {
                    Ok(bytes) => {
                        let value = String::from_utf8(bytes.to_vec())
                            .map_err(|e| format!("Invalid UTF-8 in keychain: {}", e))?;
                        // Migrate: store under new name and remove old entry
                        let _ = store_secret(key, &value);
                        let _ = delete_generic_password(LEGACY_SERVICE_NAME, key);
                        tracing::info!("Migrated keychain entry '{}' from SuperBrain to DeepBrain", key);
                        Ok(Some(value))
                    }
                    Err(legacy_err) => {
                        if legacy_err.code() == -25300 {
                            Ok(None)
                        } else {
                            Err(format!("Keychain read failed: {}", legacy_err))
                        }
                    }
                }
            } else {
                Err(format!("Keychain read failed: {}", e))
            }
        }
    }
}

/// Delete a secret from the macOS Keychain
pub fn delete_secret(key: &str) -> Result<(), String> {
    // Delete from both service names to be thorough
    let _ = delete_generic_password(LEGACY_SERVICE_NAME, key);
    match delete_generic_password(SERVICE_NAME, key) {
        Ok(()) => Ok(()),
        Err(e) => {
            if e.code() == -25300 {
                Ok(()) // Not found is fine
            } else {
                Err(format!("Keychain delete failed: {}", e))
            }
        }
    }
}
