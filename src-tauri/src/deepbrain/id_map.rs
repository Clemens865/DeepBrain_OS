//! Bidirectional String ↔ u64 ID mapping for RvfStore compatibility.
//!
//! RvfStore uses u64 vector IDs. DeepBrain uses string UUIDs.
//! This module bridges the two with a SQLite-backed persistent map.

use std::path::Path;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Mutex;

use rusqlite::{params, Connection, OptionalExtension};

use super::error::DeepBrainError;

/// Bidirectional mapping between string IDs and u64 numeric IDs.
///
/// Backed by SQLite for persistence, with an atomic counter for fast
/// allocation of new numeric IDs. Thread-safe via internal Mutex.
pub struct IdMap {
    conn: Mutex<Connection>,
    next_id: AtomicU64,
}

impl IdMap {
    /// Open or create the id_map database at the given path.
    pub fn open(db_path: &Path) -> Result<Self, DeepBrainError> {
        let conn = Connection::open(db_path)?;
        conn.execute_batch(
            "PRAGMA journal_mode=WAL;
             PRAGMA synchronous=NORMAL;
             CREATE TABLE IF NOT EXISTS vector_id_map (
                 str_id TEXT PRIMARY KEY,
                 num_id INTEGER UNIQUE NOT NULL
             );
             CREATE INDEX IF NOT EXISTS idx_num_id ON vector_id_map(num_id);",
        )?;

        // Find the current max numeric ID to continue from.
        let max_id: u64 = conn
            .query_row(
                "SELECT COALESCE(MAX(num_id), 0) FROM vector_id_map",
                [],
                |row| row.get(0),
            )?;

        Ok(Self {
            conn: Mutex::new(conn),
            next_id: AtomicU64::new(max_id + 1),
        })
    }

    /// Get the numeric ID for a string ID, or allocate a new one.
    ///
    /// Returns `(num_id, is_new)` where `is_new` is true if the mapping
    /// was just created.
    pub fn get_or_insert(&self, str_id: &str) -> Result<(u64, bool), DeepBrainError> {
        let conn = self.conn.lock().map_err(|_| {
            DeepBrainError::Io(std::io::Error::new(
                std::io::ErrorKind::Other,
                "IdMap mutex poisoned",
            ))
        })?;

        // Check existing mapping first.
        let existing: Option<i64> = conn
            .query_row(
                "SELECT num_id FROM vector_id_map WHERE str_id = ?1",
                params![str_id],
                |row| row.get(0),
            )
            .optional()?;

        if let Some(num_id) = existing {
            return Ok((num_id as u64, false));
        }

        // Allocate a new numeric ID.
        let num_id = self.next_id.fetch_add(1, Ordering::Relaxed);
        conn.execute(
            "INSERT OR IGNORE INTO vector_id_map (str_id, num_id) VALUES (?1, ?2)",
            params![str_id, num_id as i64],
        )?;

        Ok((num_id, true))
    }

    /// Look up the numeric ID for a string ID.
    pub fn get_num_id(&self, str_id: &str) -> Result<Option<u64>, DeepBrainError> {
        let conn = self.conn.lock().map_err(|_| {
            DeepBrainError::Io(std::io::Error::new(
                std::io::ErrorKind::Other,
                "IdMap mutex poisoned",
            ))
        })?;

        let result: Option<i64> = conn
            .query_row(
                "SELECT num_id FROM vector_id_map WHERE str_id = ?1",
                params![str_id],
                |row| row.get(0),
            )
            .optional()?;

        Ok(result.map(|v| v as u64))
    }

    /// Look up the string ID for a numeric ID.
    pub fn get_str_id(&self, num_id: u64) -> Result<Option<String>, DeepBrainError> {
        let conn = self.conn.lock().map_err(|_| {
            DeepBrainError::Io(std::io::Error::new(
                std::io::ErrorKind::Other,
                "IdMap mutex poisoned",
            ))
        })?;

        let result: Option<String> = conn
            .query_row(
                "SELECT str_id FROM vector_id_map WHERE num_id = ?1",
                params![num_id as i64],
                |row| row.get(0),
            )
            .optional()?;

        Ok(result)
    }

    /// Remove a mapping by string ID.
    pub fn remove(&self, str_id: &str) -> Result<bool, DeepBrainError> {
        let conn = self.conn.lock().map_err(|_| {
            DeepBrainError::Io(std::io::Error::new(
                std::io::ErrorKind::Other,
                "IdMap mutex poisoned",
            ))
        })?;

        let changed = conn.execute(
            "DELETE FROM vector_id_map WHERE str_id = ?1",
            params![str_id],
        )?;
        Ok(changed > 0)
    }

    /// Batch insert multiple mappings in a single transaction.
    /// Returns the list of (str_id, num_id) pairs.
    pub fn insert_batch(
        &self,
        str_ids: &[&str],
    ) -> Result<Vec<(String, u64)>, DeepBrainError> {
        let conn = self.conn.lock().map_err(|_| {
            DeepBrainError::Io(std::io::Error::new(
                std::io::ErrorKind::Other,
                "IdMap mutex poisoned",
            ))
        })?;

        let mut results = Vec::with_capacity(str_ids.len());

        let tx = conn.unchecked_transaction()?;
        {
            let mut stmt = tx.prepare(
                "INSERT OR IGNORE INTO vector_id_map (str_id, num_id) VALUES (?1, ?2)",
            )?;

            for &str_id in str_ids {
                let num_id = self.next_id.fetch_add(1, Ordering::Relaxed);
                stmt.execute(params![str_id, num_id as i64])?;
                results.push((str_id.to_string(), num_id));
            }
        }
        tx.commit()?;

        Ok(results)
    }

    /// Total number of mappings.
    pub fn count(&self) -> Result<u64, DeepBrainError> {
        let conn = self.conn.lock().map_err(|_| {
            DeepBrainError::Io(std::io::Error::new(
                std::io::ErrorKind::Other,
                "IdMap mutex poisoned",
            ))
        })?;

        let count: i64 = conn
            .query_row("SELECT COUNT(*) FROM vector_id_map", [], |row| row.get(0))?;
        Ok(count as u64)
    }
}
