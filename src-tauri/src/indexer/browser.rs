//! Browser history indexer for DeepBrain
//!
//! Periodically syncs new URLs from the Comet browser's Chromium SQLite
//! History database into searchable brain memories. Uses a watermark
//! (last_visit_time) to only import new entries on each pass.

use std::sync::Arc;

use parking_lot::RwLock;
use serde::{Deserialize, Serialize};

use crate::brain::cognitive::CognitiveEngine;
use crate::brain::embeddings::EmbeddingModel;
use crate::brain::persistence::BrainPersistence;
use crate::deepbrain::graph_bridge::GraphBridge;
use crate::deepbrain::vector_store::DeepBrainVectorStore;
use crate::indexer::bootstrap;

/// Stats about the browser indexer
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BrowserStats {
    pub indexed_count: u32,
    pub last_sync_unix: i64,
    pub is_syncing: bool,
}

/// A row from the Chromium urls table
struct UrlRow {
    url: String,
    title: String,
    visit_count: u32,
    last_visit_time: i64,
}

pub struct BrowserIndexer {
    engine: Arc<CognitiveEngine>,
    embeddings: Arc<EmbeddingModel>,
    _vector_store: Option<Arc<DeepBrainVectorStore>>,
    persistence: Arc<BrainPersistence>,
    graph: Arc<GraphBridge>,
    is_syncing: RwLock<bool>,
    last_synced_time: RwLock<i64>,
    indexed_count: RwLock<u32>,
}

impl BrowserIndexer {
    pub fn new(
        engine: Arc<CognitiveEngine>,
        embeddings: Arc<EmbeddingModel>,
        vector_store: Option<Arc<DeepBrainVectorStore>>,
        persistence: Arc<BrainPersistence>,
        graph: Arc<GraphBridge>,
    ) -> Self {
        Self {
            engine,
            embeddings,
            _vector_store: vector_store,
            persistence,
            graph,
            is_syncing: RwLock::new(false),
            last_synced_time: RwLock::new(0),
            indexed_count: RwLock::new(0),
        }
    }

    /// Run a single sync pass: import new URLs since the last watermark.
    /// Returns the number of new URLs indexed.
    pub async fn sync_pass(&self) -> Result<u32, String> {
        // Guard against concurrent syncs
        {
            let syncing = self.is_syncing.read();
            if *syncing {
                return Ok(0);
            }
        }
        *self.is_syncing.write() = true;

        let watermark = *self.last_synced_time.read();

        // Read rows synchronously (rusqlite::Connection is !Send)
        let rows = match read_new_urls(watermark) {
            Ok(r) => r,
            Err(e) => {
                *self.is_syncing.write() = false;
                return Err(e);
            }
        };

        if rows.is_empty() {
            *self.is_syncing.write() = false;
            return Ok(0);
        }

        let mut created = 0u32;
        let mut max_visit_time = watermark;

        for row in &rows {
            // Track the highest visit time for the watermark
            if row.last_visit_time > max_visit_time {
                max_visit_time = row.last_visit_time;
            }

            // Deterministic ID based on URL hash
            let url_hash = bootstrap::deterministic_hash(&row.url);
            let memory_id = format!("browser::comet::{}", url_hash);

            // Importance scales with visit count
            let importance = if row.visit_count > 20 {
                0.7
            } else if row.visit_count > 5 {
                0.5
            } else {
                0.3
            };

            let content = format!("[Page] {}\n{}", row.title, row.url);

            match store_browser_memory(
                &self.engine,
                &self.embeddings,
                &self.persistence,
                &self.graph,
                &memory_id,
                &content,
                importance,
            )
            .await
            {
                Ok(true) => created += 1,
                Ok(false) => {} // Already exists
                Err(e) => {
                    tracing::debug!("Browser sync error for {}: {}", row.url, e);
                }
            }

            // Yield periodically
            if created % 20 == 19 {
                tokio::task::yield_now().await;
            }
        }

        // Update watermark
        if max_visit_time > watermark {
            *self.last_synced_time.write() = max_visit_time;
        }

        // Update indexed count
        *self.indexed_count.write() += created;

        // Persist new memories
        if created > 0 {
            let nodes = self.engine.memory.all_nodes();
            let _ = self.persistence.store_memories_batch(&nodes);
        }

        *self.is_syncing.write() = false;
        Ok(created)
    }

    /// Get current stats
    pub fn stats(&self) -> BrowserStats {
        let last_sync_chromium = *self.last_synced_time.read();
        // Convert Chromium timestamp to Unix: (chromium_us / 1_000_000) - 11_644_473_600
        let last_sync_unix = if last_sync_chromium > 0 {
            (last_sync_chromium / 1_000_000) - 11_644_473_600
        } else {
            0
        };
        BrowserStats {
            indexed_count: *self.indexed_count.read(),
            last_sync_unix,
            is_syncing: *self.is_syncing.read(),
        }
    }
}

/// Read new URLs from the Comet History DB since the given Chromium timestamp watermark.
/// All SQLite work is done synchronously to avoid holding Connection across .await.
fn read_new_urls(since_chromium_time: i64) -> Result<Vec<UrlRow>, String> {
    let home = dirs::home_dir().ok_or("no home dir")?;
    let history_db = home.join("Library/Application Support/Comet/Default/History");
    if !history_db.exists() {
        return Err("Comet browser History not found".to_string());
    }

    let temp_db = bootstrap::copy_sqlite_to_temp(&history_db, "comet_sync")?;
    let conn = rusqlite::Connection::open_with_flags(
        &temp_db,
        rusqlite::OpenFlags::SQLITE_OPEN_READ_ONLY | rusqlite::OpenFlags::SQLITE_OPEN_NO_MUTEX,
    )
    .map_err(|e| format!("open Comet History copy: {}", e))?;

    let mut stmt = conn
        .prepare(
            "SELECT url, title, visit_count, last_visit_time
             FROM urls
             WHERE last_visit_time > ?1
               AND title IS NOT NULL AND title != ''
             ORDER BY last_visit_time ASC
             LIMIT 500",
        )
        .map_err(|e| format!("prepare: {}", e))?;

    let rows: Vec<UrlRow> = stmt
        .query_map(rusqlite::params![since_chromium_time], |row| {
            Ok(UrlRow {
                url: row.get(0)?,
                title: row.get(1)?,
                visit_count: row.get(2)?,
                last_visit_time: row.get(3)?,
            })
        })
        .map_err(|e| format!("query: {}", e))?
        .filter_map(|r| r.ok())
        .collect();

    // Clean up temp file
    let _ = std::fs::remove_file(&temp_db);

    Ok(rows)
}

/// Store a browser URL as a brain memory with graph integration.
/// Returns true if newly created, false if already exists.
async fn store_browser_memory(
    engine: &CognitiveEngine,
    embeddings: &EmbeddingModel,
    persistence: &BrainPersistence,
    graph: &GraphBridge,
    id: &str,
    content: &str,
    importance: f64,
) -> Result<bool, String> {
    if content.trim().len() < 20 {
        return Ok(false);
    }

    // Check idempotency
    if engine.memory.has_memory(id) {
        return Ok(false);
    }

    // Embed
    let vector = embeddings.embed(content).await?;

    // Store with deterministic ID
    let created = engine.memory.store_f32_with_id(
        id.to_string(),
        content.to_string(),
        vector.clone(),
        "bookmark".to_string(),
        importance,
    )?;

    if created {
        // Persist to SQLite
        if let Some(node) = engine.memory.all_nodes().into_iter().find(|n| n.id == id) {
            let _ = persistence.store_memory(&node);
        }

        // Add to knowledge graph
        let _ = graph.add_memory_node(id, "bookmark", content);

        // Auto-connect to similar memories
        let similar = engine.recall_f32(&vector, Some(5), None).unwrap_or_default();
        let related_ids: Vec<String> = similar
            .iter()
            .filter(|s| s.id != id && s.similarity > 0.5)
            .map(|s| s.id.clone())
            .collect();
        if !related_ids.is_empty() {
            graph.auto_connect(id, &related_ids, 0.6);
        }
    }

    Ok(created)
}
