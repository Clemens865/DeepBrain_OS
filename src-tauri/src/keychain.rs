//! macOS Keychain integration for secure credential storage

use security_framework::passwords::{
    delete_generic_password, get_generic_password, set_generic_password,
};

const SERVICE_NAME: &str = "SuperBrain";

/// Store a secret in the macOS Keychain
pub fn store_secret(key: &str, value: &str) -> Result<(), String> {
    // Delete existing entry first (set_generic_password fails if it already exists)
    let _ = delete_generic_password(SERVICE_NAME, key);
    set_generic_password(SERVICE_NAME, key, value.as_bytes())
        .map_err(|e| format!("Keychain store failed: {}", e))
}

/// Retrieve a secret from the macOS Keychain
pub fn get_secret(key: &str) -> Result<Option<String>, String> {
    match get_generic_password(SERVICE_NAME, key) {
        Ok(bytes) => {
            let value = String::from_utf8(bytes.to_vec())
                .map_err(|e| format!("Invalid UTF-8 in keychain: {}", e))?;
            Ok(Some(value))
        }
        Err(e) => {
            // errSecItemNotFound is expected when key doesn't exist
            if e.code() == -25300 {
                Ok(None)
            } else {
                Err(format!("Keychain read failed: {}", e))
            }
        }
    }
}

/// Delete a secret from the macOS Keychain
pub fn delete_secret(key: &str) -> Result<(), String> {
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
