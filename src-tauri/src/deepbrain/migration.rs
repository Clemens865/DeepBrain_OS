//! SQLite → VectorDB migration tool.
//!
//! Migrates vectors from SuperBrain's brain.db (memories) and files.db
//! (file_chunks, including email chunks) into the DeepBrain VectorDB.
//!
//! Migration is non-destructive — the source SQLite databases are only read.

use std::path::Path;
use std::time::Instant;

use rusqlite::Connection;
use serde::Serialize;

use super::error::DeepBrainError;
use super::vector_store::{DeepBrainVectorStore, VectorMetadata};

/// Report from a migration run.
#[derive(Clone, Debug, Serialize)]
pub struct MigrationReport {
    /// Number of memory vectors migrated.
    pub memory_vectors: u64,
    /// Number of file chunk vectors migrated.
    pub file_vectors: u64,
    /// Number of email chunk vectors migrated.
    pub email_vectors: u64,
    /// Number of vectors that were skipped (wrong dimension).
    pub skipped: u64,
    /// Total migration time in milliseconds.
    pub duration_ms: u64,
    /// Whether verification passed.
    pub verified: bool,
}

/// Migrate vectors from SuperBrain SQLite databases to a DeepBrain VectorDB.
///
/// # Arguments
/// * `brain_db` - Path to SuperBrain's brain.db (contains `memories` table)
/// * `files_db` - Path to SuperBrain's files.db (contains `file_chunks` table)
/// * `target_dir` - Directory for the new DeepBrain store
///
/// The source databases are only read, never modified.
pub fn migrate_from_superbrain(
    brain_db: &Path,
    files_db: &Path,
    target_dir: &Path,
) -> Result<MigrationReport, DeepBrainError> {
    let start = Instant::now();
    let dimension = DeepBrainVectorStore::DEFAULT_DIMENSION as usize;

    // Create or open the target store.
    let store = DeepBrainVectorStore::open_or_create(target_dir)?;

    let mut memory_count: u64 = 0;
    let mut file_count: u64 = 0;
    let mut email_count: u64 = 0;
    let mut skipped: u64 = 0;

    // --- Phase 1: Migrate memory vectors from brain.db ---
    if brain_db.exists() {
        let conn = Connection::open(brain_db)
            .map_err(|e| DeepBrainError::Migration(format!("Failed to open brain.db: {}", e)))?;
        conn.execute_batch("PRAGMA journal_mode=WAL;")
            .map_err(|e| DeepBrainError::Migration(format!("PRAGMA failed: {}", e)))?;

        let mut stmt = conn.prepare(
            "SELECT id, vector, memory_type, importance, timestamp FROM memories",
        ).map_err(|e| DeepBrainError::Migration(format!("Prepare failed: {}", e)))?;

        let mut batch: Vec<(String, Vec<f32>, VectorMetadata)> = Vec::with_capacity(500);

        let rows = stmt.query_map([], |row| {
            let id: String = row.get(0)?;
            let vector_bytes: Vec<u8> = row.get(1)?;
            let memory_type: String = row.get(2)?;
            let importance: f64 = row.get(3)?;
            let timestamp: i64 = row.get(4)?;
            Ok((id, vector_bytes, memory_type, importance, timestamp))
        }).map_err(|e| DeepBrainError::Migration(format!("Query failed: {}", e)))?;

        for row in rows {
            let (id, vector_bytes, memory_type, importance, timestamp) = row
                .map_err(|e| DeepBrainError::Migration(format!("Row read failed: {}", e)))?;
            let vector = bytes_to_f32_vec(&vector_bytes);

            if vector.len() != dimension {
                skipped += 1;
                continue;
            }

            let metadata = VectorMetadata {
                source: "memory".to_string(),
                memory_type,
                importance,
                timestamp,
                path: String::new(),
            };

            batch.push((id, vector, metadata));

            // Flush in batches of 500.
            if batch.len() >= 500 {
                let accepted = store.store_batch(&batch)?;
                memory_count += accepted;
                batch.clear();
            }
        }

        // Flush remaining.
        if !batch.is_empty() {
            let accepted = store.store_batch(&batch)?;
            memory_count += accepted;
        }

        tracing::info!("Migrated {} memory vectors from brain.db", memory_count);
    }

    // --- Phase 2: Migrate file/email chunk vectors from files.db ---
    if files_db.exists() {
        let conn = Connection::open(files_db)
            .map_err(|e| DeepBrainError::Migration(format!("Failed to open files.db: {}", e)))?;
        conn.execute_batch("PRAGMA journal_mode=WAL;")
            .map_err(|e| DeepBrainError::Migration(format!("PRAGMA failed: {}", e)))?;

        let mut stmt = conn.prepare(
            "SELECT file_path, chunk_index, vector FROM file_chunks",
        ).map_err(|e| DeepBrainError::Migration(format!("Prepare failed: {}", e)))?;

        let mut batch: Vec<(String, Vec<f32>, VectorMetadata)> = Vec::with_capacity(500);

        let rows = stmt.query_map([], |row| {
            let file_path: String = row.get(0)?;
            let chunk_index: u32 = row.get(1)?;
            let vector_bytes: Vec<u8> = row.get(2)?;
            Ok((file_path, chunk_index, vector_bytes))
        }).map_err(|e| DeepBrainError::Migration(format!("Query failed: {}", e)))?;

        for row in rows {
            let (file_path, chunk_index, vector_bytes) = row
                .map_err(|e| DeepBrainError::Migration(format!("Row read failed: {}", e)))?;
            let vector = bytes_to_f32_vec(&vector_bytes);

            if vector.len() != dimension {
                skipped += 1;
                continue;
            }

            // Emails use synthetic paths like "mail://<message_id>".
            let is_email = file_path.starts_with("mail://");
            let (source, id_prefix) = if is_email {
                ("email", "email")
            } else {
                ("file", "file")
            };

            let str_id = format!("{}::{}::{}", id_prefix, file_path, chunk_index);

            let now_ms = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_millis() as i64;

            let metadata = VectorMetadata {
                source: source.to_string(),
                memory_type: String::new(),
                importance: 0.0,
                timestamp: now_ms,
                path: file_path,
            };

            batch.push((str_id, vector, metadata));

            if batch.len() >= 500 {
                let accepted = store.store_batch(&batch)?;
                if is_email {
                    email_count += accepted;
                } else {
                    file_count += accepted;
                }
                batch.clear();
            }
        }

        // Flush remaining — need to count per-source type.
        if !batch.is_empty() {
            let email_in_batch = batch.iter().filter(|(_, _, m)| m.source == "email").count() as u64;
            let file_in_batch = batch.len() as u64 - email_in_batch;
            let accepted = store.store_batch(&batch)?;
            // Approximate split based on batch composition.
            if accepted == batch.len() as u64 {
                email_count += email_in_batch;
                file_count += file_in_batch;
            } else {
                // Best-effort split.
                file_count += accepted.saturating_sub(email_in_batch);
                email_count += accepted.min(email_in_batch);
            }
        }

        tracing::info!(
            "Migrated {} file + {} email vectors from files.db",
            file_count,
            email_count,
        );
    }

    let duration_ms = start.elapsed().as_millis() as u64;

    // --- Phase 3: Verification ---
    let status = store.status();
    let expected_total = memory_count + file_count + email_count;
    let verified = status.total_vectors >= expected_total;

    if verified {
        tracing::info!(
            "Migration verified: {} vectors in store (expected {})",
            status.total_vectors,
            expected_total,
        );
    } else {
        tracing::warn!(
            "Migration verification mismatch: {} vectors in store, expected {}",
            status.total_vectors,
            expected_total,
        );
    }

    Ok(MigrationReport {
        memory_vectors: memory_count,
        file_vectors: file_count,
        email_vectors: email_count,
        skipped,
        duration_ms,
        verified,
    })
}

/// Check if migration is needed (SuperBrain data exists but DeepBrain store doesn't).
pub fn needs_migration(data_dir: &Path) -> bool {
    let superbrain_dir = dirs::data_dir()
        .map(|d| d.join("SuperBrain"))
        .unwrap_or_default();

    let brain_db = superbrain_dir.join("brain.db");
    let deepbrain_store = data_dir.join("knowledge.redb");

    brain_db.exists() && !deepbrain_store.exists()
}

/// Convert a little-endian byte slice to a Vec<f32>.
fn bytes_to_f32_vec(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect()
}
