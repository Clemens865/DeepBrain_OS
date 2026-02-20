//! SQLite persistence layer for DeepBrain
//!
//! Persists memories, Q-table, experiences, goals, and configuration
//! to ~/Library/Application Support/DeepBrain/brain.db

use std::path::PathBuf;

use rusqlite::{params, Connection};
use smallvec::SmallVec;

use crate::brain::memory::MemoryNode;
use crate::brain::types::MemoryType;

/// Persistence manager for the cognitive engine
pub struct BrainPersistence {
    db_path: PathBuf,
}

impl BrainPersistence {
    /// Create a new persistence manager
    pub fn new() -> Result<Self, String> {
        let data_dir = dirs::data_dir()
            .ok_or("Could not find Application Support directory")?
            .join("DeepBrain");

        std::fs::create_dir_all(&data_dir)
            .map_err(|e| format!("Failed to create data directory: {}", e))?;

        let db_path = data_dir.join("brain.db");

        let persistence = Self { db_path };
        persistence.initialize_db()?;

        Ok(persistence)
    }

    /// Create with custom path (for testing)
    pub fn with_path(db_path: PathBuf) -> Result<Self, String> {
        if let Some(parent) = db_path.parent() {
            std::fs::create_dir_all(parent)
                .map_err(|e| format!("Failed to create directory: {}", e))?;
        }

        let persistence = Self { db_path };
        persistence.initialize_db()?;

        Ok(persistence)
    }

    fn open_connection(&self) -> Result<Connection, String> {
        Connection::open(&self.db_path).map_err(|e| format!("Failed to open database: {}", e))
    }

    /// Initialize database tables
    fn initialize_db(&self) -> Result<(), String> {
        let conn = self.open_connection()?;

        // Enable WAL mode for better concurrent read performance
        conn.execute_batch("PRAGMA journal_mode=WAL;")
            .map_err(|e| format!("Failed to set WAL mode: {}", e))?;

        conn.execute_batch(
            "
            CREATE TABLE IF NOT EXISTS memories (
                id TEXT PRIMARY KEY,
                content TEXT NOT NULL,
                vector BLOB NOT NULL,
                memory_type TEXT NOT NULL,
                importance REAL NOT NULL DEFAULT 0.5,
                decay REAL NOT NULL DEFAULT 0.0,
                access_count INTEGER NOT NULL DEFAULT 0,
                timestamp INTEGER NOT NULL,
                connections TEXT NOT NULL DEFAULT '[]'
            );

            CREATE TABLE IF NOT EXISTS q_table (
                state_hash INTEGER PRIMARY KEY,
                values_json TEXT NOT NULL,
                visits INTEGER NOT NULL DEFAULT 0
            );

            CREATE TABLE IF NOT EXISTS experiences (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                state_json TEXT NOT NULL,
                action INTEGER NOT NULL,
                reward REAL NOT NULL,
                next_state_json TEXT NOT NULL,
                done INTEGER NOT NULL DEFAULT 0,
                timestamp INTEGER NOT NULL
            );

            CREATE TABLE IF NOT EXISTS goals (
                id TEXT PRIMARY KEY,
                description TEXT NOT NULL,
                priority REAL NOT NULL,
                progress REAL NOT NULL DEFAULT 0.0,
                status TEXT NOT NULL DEFAULT 'Pending',
                created_at INTEGER NOT NULL
            );

            CREATE TABLE IF NOT EXISTS config (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );

            CREATE INDEX IF NOT EXISTS idx_memories_type ON memories(memory_type);
            CREATE INDEX IF NOT EXISTS idx_memories_importance ON memories(importance);
            CREATE INDEX IF NOT EXISTS idx_memories_timestamp ON memories(timestamp);

            CREATE TABLE IF NOT EXISTS query_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                query_hash INTEGER NOT NULL,
                result_ids TEXT NOT NULL,
                selected_id TEXT,
                source TEXT NOT NULL,
                timestamp INTEGER NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_query_log_ts ON query_log(timestamp);

            CREATE TABLE IF NOT EXISTS blob_store (
                key TEXT PRIMARY KEY,
                data BLOB NOT NULL,
                updated_at INTEGER NOT NULL
            );
            ",
        )
        .map_err(|e| format!("Failed to create tables: {}", e))?;

        Ok(())
    }

    // ---- Memory Persistence ----

    /// Store a single memory
    pub fn store_memory(&self, node: &MemoryNode) -> Result<(), String> {
        let conn = self.open_connection()?;
        let vector_bytes = vector_to_bytes(&node.vector);
        let connections_json =
            serde_json::to_string(&node.connections.to_vec()).unwrap_or_else(|_| "[]".to_string());

        conn.execute(
            "INSERT OR REPLACE INTO memories (id, content, vector, memory_type, importance, decay, access_count, timestamp, connections)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9)",
            params![
                node.id,
                node.content,
                vector_bytes,
                format!("{:?}", node.memory_type),
                node.importance,
                node.decay,
                node.access_count,
                node.timestamp,
                connections_json,
            ],
        )
        .map_err(|e| format!("Failed to store memory: {}", e))?;

        Ok(())
    }

    /// Store multiple memories in a transaction
    pub fn store_memories_batch(&self, nodes: &[MemoryNode]) -> Result<(), String> {
        let conn = self.open_connection()?;

        conn.execute_batch("BEGIN TRANSACTION;")
            .map_err(|e| format!("Failed to begin transaction: {}", e))?;

        for node in nodes {
            let vector_bytes = vector_to_bytes(&node.vector);
            let connections_json = serde_json::to_string(&node.connections.to_vec())
                .unwrap_or_else(|_| "[]".to_string());

            if let Err(e) = conn.execute(
                "INSERT OR REPLACE INTO memories (id, content, vector, memory_type, importance, decay, access_count, timestamp, connections)
                 VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9)",
                params![
                    node.id,
                    node.content,
                    vector_bytes,
                    format!("{:?}", node.memory_type),
                    node.importance,
                    node.decay,
                    node.access_count,
                    node.timestamp,
                    connections_json,
                ],
            ) {
                let _ = conn.execute_batch("ROLLBACK;");
                return Err(format!("Failed to store memory: {}", e));
            }
        }

        conn.execute_batch("COMMIT;")
            .map_err(|e| format!("Failed to commit: {}", e))?;

        Ok(())
    }

    /// Load all memories from database
    pub fn load_memories(&self) -> Result<Vec<MemoryNode>, String> {
        let conn = self.open_connection()?;
        let mut stmt = conn
            .prepare("SELECT id, content, vector, memory_type, importance, decay, access_count, timestamp, connections FROM memories")
            .map_err(|e| format!("Failed to prepare query: {}", e))?;

        let memories = stmt
            .query_map([], |row| {
                let id: String = row.get(0)?;
                let content: String = row.get(1)?;
                let vector_bytes: Vec<u8> = row.get(2)?;
                let memory_type_str: String = row.get(3)?;
                let importance: f64 = row.get(4)?;
                let decay: f64 = row.get(5)?;
                let access_count: u32 = row.get(6)?;
                let timestamp: i64 = row.get(7)?;
                let connections_json: String = row.get(8)?;

                let vector = bytes_to_vector(&vector_bytes);
                let memory_type = parse_memory_type_from_debug(&memory_type_str);
                let connections: Vec<String> =
                    serde_json::from_str(&connections_json).unwrap_or_default();

                Ok(MemoryNode {
                    id,
                    content,
                    vector,
                    memory_type,
                    importance,
                    decay,
                    access_count,
                    timestamp,
                    connections: SmallVec::from_vec(connections),
                })
            })
            .map_err(|e| format!("Failed to query memories: {}", e))?
            .filter_map(|r| r.ok())
            .collect();

        Ok(memories)
    }

    /// Delete a memory by ID
    pub fn delete_memory(&self, id: &str) -> Result<(), String> {
        let conn = self.open_connection()?;
        conn.execute("DELETE FROM memories WHERE id = ?1", params![id])
            .map_err(|e| format!("Failed to delete memory: {}", e))?;
        Ok(())
    }

    /// Get memory count
    pub fn memory_count(&self) -> Result<u32, String> {
        let conn = self.open_connection()?;
        let count: u32 = conn
            .query_row("SELECT COUNT(*) FROM memories", [], |row| row.get(0))
            .map_err(|e| format!("Failed to count memories: {}", e))?;
        Ok(count)
    }

    // ---- Q-Table Persistence ----

    /// Store Q-table entries
    pub fn store_q_table(&self, entries: &[(u64, Vec<f64>, u32)]) -> Result<(), String> {
        let conn = self.open_connection()?;

        conn.execute_batch("BEGIN TRANSACTION;")
            .map_err(|e| format!("Failed to begin transaction: {}", e))?;

        for (state_hash, values, visits) in entries {
            let values_json = serde_json::to_string(values).unwrap_or_else(|_| "[]".to_string());

            if let Err(e) = conn.execute(
                "INSERT OR REPLACE INTO q_table (state_hash, values_json, visits) VALUES (?1, ?2, ?3)",
                params![*state_hash as i64, values_json, *visits],
            ) {
                let _ = conn.execute_batch("ROLLBACK;");
                return Err(format!("Failed to store Q-table entry: {}", e));
            }
        }

        conn.execute_batch("COMMIT;")
            .map_err(|e| format!("Failed to commit: {}", e))?;

        Ok(())
    }

    /// Load Q-table entries
    pub fn load_q_table(&self) -> Result<Vec<(u64, Vec<f64>, u32)>, String> {
        let conn = self.open_connection()?;
        let mut stmt = conn
            .prepare("SELECT state_hash, values_json, visits FROM q_table")
            .map_err(|e| format!("Failed to prepare query: {}", e))?;

        let entries = stmt
            .query_map([], |row| {
                let state_hash: i64 = row.get(0)?;
                let values_json: String = row.get(1)?;
                let visits: u32 = row.get(2)?;

                let values: Vec<f64> = serde_json::from_str(&values_json).unwrap_or_default();

                Ok((state_hash as u64, values, visits))
            })
            .map_err(|e| format!("Failed to query Q-table: {}", e))?
            .filter_map(|r| r.ok())
            .collect();

        Ok(entries)
    }

    // ---- Config Persistence ----

    /// Store a config value
    pub fn store_config(&self, key: &str, value: &str) -> Result<(), String> {
        let conn = self.open_connection()?;
        conn.execute(
            "INSERT OR REPLACE INTO config (key, value) VALUES (?1, ?2)",
            params![key, value],
        )
        .map_err(|e| format!("Failed to store config: {}", e))?;
        Ok(())
    }

    /// Load a config value
    pub fn load_config(&self, key: &str) -> Result<Option<String>, String> {
        let conn = self.open_connection()?;
        let result = conn.query_row(
            "SELECT value FROM config WHERE key = ?1",
            params![key],
            |row| row.get(0),
        );

        match result {
            Ok(value) => Ok(Some(value)),
            Err(rusqlite::Error::QueryReturnedNoRows) => Ok(None),
            Err(e) => Err(format!("Failed to load config: {}", e)),
        }
    }

    /// Get database path
    pub fn db_path(&self) -> &PathBuf {
        &self.db_path
    }

    // ---- Query Log ----

    /// Log a search query and its results for GNN training
    pub fn log_query(
        &self,
        query_hash: i64,
        result_ids: &[String],
        selected_id: Option<&str>,
        source: &str,
    ) -> Result<(), String> {
        let conn = self.open_connection()?;
        let ids_json =
            serde_json::to_string(result_ids).unwrap_or_else(|_| "[]".to_string());
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as i64;

        conn.execute(
            "INSERT INTO query_log (query_hash, result_ids, selected_id, source, timestamp)
             VALUES (?1, ?2, ?3, ?4, ?5)",
            params![query_hash, ids_json, selected_id, source, now],
        )
        .map_err(|e| format!("Failed to log query: {}", e))?;

        Ok(())
    }

    /// Load recent query log entries for training
    pub fn load_recent_query_logs(
        &self,
        limit: usize,
    ) -> Result<Vec<crate::brain::gnn::QueryLogEntry>, String> {
        let conn = self.open_connection()?;
        let mut stmt = conn
            .prepare(
                "SELECT query_hash, result_ids, selected_id, source, timestamp
                 FROM query_log ORDER BY timestamp DESC LIMIT ?1",
            )
            .map_err(|e| format!("Failed to prepare query: {}", e))?;

        let entries = stmt
            .query_map(params![limit as u32], |row| {
                let query_hash: i64 = row.get(0)?;
                let result_ids_json: String = row.get(1)?;
                let selected_id: Option<String> = row.get(2)?;
                let source: String = row.get(3)?;
                let timestamp: i64 = row.get(4)?;

                let result_ids: Vec<String> =
                    serde_json::from_str(&result_ids_json).unwrap_or_default();

                Ok(crate::brain::gnn::QueryLogEntry {
                    query_hash,
                    result_ids,
                    selected_id,
                    source,
                    timestamp,
                })
            })
            .map_err(|e| format!("Failed to query logs: {}", e))?
            .filter_map(|r| r.ok())
            .collect();

        Ok(entries)
    }

    /// Get the count of query log entries since a given timestamp
    pub fn query_log_count_since(&self, since_ms: i64) -> Result<u32, String> {
        let conn = self.open_connection()?;
        let count: u32 = conn
            .query_row(
                "SELECT COUNT(*) FROM query_log WHERE timestamp > ?1",
                params![since_ms],
                |row| row.get(0),
            )
            .map_err(|e| format!("Failed to count query logs: {}", e))?;
        Ok(count)
    }

    // ---- Blob Store (for HNSW index and GNN weights) ----

    /// Store a binary blob
    pub fn store_blob(&self, key: &str, data: &[u8]) -> Result<(), String> {
        let conn = self.open_connection()?;
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as i64;

        conn.execute(
            "INSERT OR REPLACE INTO blob_store (key, data, updated_at) VALUES (?1, ?2, ?3)",
            params![key, data, now],
        )
        .map_err(|e| format!("Failed to store blob: {}", e))?;

        Ok(())
    }

    /// Load a binary blob
    pub fn load_blob(&self, key: &str) -> Result<Option<Vec<u8>>, String> {
        let conn = self.open_connection()?;
        let result = conn.query_row(
            "SELECT data FROM blob_store WHERE key = ?1",
            params![key],
            |row| row.get(0),
        );

        match result {
            Ok(data) => Ok(Some(data)),
            Err(rusqlite::Error::QueryReturnedNoRows) => Ok(None),
            Err(e) => Err(format!("Failed to load blob: {}", e)),
        }
    }
}

// ---- Helper Functions ----

fn vector_to_bytes(vector: &[f32]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(vector.len() * 4);
    for &val in vector {
        bytes.extend_from_slice(&val.to_le_bytes());
    }
    bytes
}

fn bytes_to_vector(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect()
}

fn parse_memory_type_from_debug(s: &str) -> MemoryType {
    match s {
        "Episodic" => MemoryType::Episodic,
        "Semantic" => MemoryType::Semantic,
        "Procedural" => MemoryType::Procedural,
        "Working" => MemoryType::Working,
        "Meta" => MemoryType::Meta,
        "Causal" => MemoryType::Causal,
        "Goal" => MemoryType::Goal,
        "Emotional" => MemoryType::Emotional,
        _ => MemoryType::Semantic,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use smallvec::SmallVec;
    

    fn temp_persistence() -> BrainPersistence {
        let path = std::env::temp_dir().join(format!("superbrain_test_{}.db", uuid::Uuid::new_v4()));
        BrainPersistence::with_path(path).unwrap()
    }

    #[test]
    fn test_memory_round_trip() {
        let p = temp_persistence();

        let node = MemoryNode {
            id: "test-1".to_string(),
            content: "Hello world".to_string(),
            vector: vec![0.1, 0.2, 0.3, 0.4],
            memory_type: MemoryType::Semantic,
            importance: 0.8,
            decay: 0.0,
            access_count: 0,
            timestamp: 1000,
            connections: SmallVec::new(),
        };

        p.store_memory(&node).unwrap();

        let loaded = p.load_memories().unwrap();
        assert_eq!(loaded.len(), 1);
        assert_eq!(loaded[0].id, "test-1");
        assert_eq!(loaded[0].content, "Hello world");
        assert!((loaded[0].vector[0] - 0.1).abs() < 1e-6);
        assert!((loaded[0].importance - 0.8).abs() < 1e-6);

        // Cleanup
        let _ = std::fs::remove_file(p.db_path());
    }

    #[test]
    fn test_batch_store_and_count() {
        let p = temp_persistence();

        let nodes: Vec<MemoryNode> = (0..100)
            .map(|i| MemoryNode {
                id: format!("mem-{}", i),
                content: format!("Memory number {}", i),
                vector: vec![i as f32 / 100.0; 4],
                memory_type: MemoryType::Episodic,
                importance: 0.5,
                decay: 0.0,
                access_count: 0,
                timestamp: 1000 + i,
                connections: SmallVec::new(),
            })
            .collect();

        p.store_memories_batch(&nodes).unwrap();

        let count = p.memory_count().unwrap();
        assert_eq!(count, 100);

        let loaded = p.load_memories().unwrap();
        assert_eq!(loaded.len(), 100);

        let _ = std::fs::remove_file(p.db_path());
    }

    #[test]
    fn test_q_table_round_trip() {
        let p = temp_persistence();

        let entries = vec![
            (12345u64, vec![0.1, 0.2, 0.3], 5u32),
            (67890u64, vec![0.4, 0.5, 0.6], 10u32),
        ];

        p.store_q_table(&entries).unwrap();

        let loaded = p.load_q_table().unwrap();
        assert_eq!(loaded.len(), 2);

        let _ = std::fs::remove_file(p.db_path());
    }

    #[test]
    fn test_config_round_trip() {
        let p = temp_persistence();

        p.store_config("test_key", "test_value").unwrap();
        let value = p.load_config("test_key").unwrap();
        assert_eq!(value, Some("test_value".to_string()));

        let missing = p.load_config("nonexistent").unwrap();
        assert_eq!(missing, None);

        let _ = std::fs::remove_file(p.db_path());
    }
}
