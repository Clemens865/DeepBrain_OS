//! Application state management for SuperBrain
//!
//! Wraps CognitiveEngine + EmbeddingModel + Persistence in Arc for Tauri managed state.

use std::sync::Arc;

use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use tokio::sync::Notify;

use crate::ai::AiProvider;
use crate::brain::cognitive::CognitiveEngine;
use crate::brain::embeddings::EmbeddingModel;
use crate::brain::persistence::BrainPersistence;
use crate::brain::types::CognitiveConfig;
use crate::context::ContextManager;
use crate::indexer::FileIndexer;

/// Application settings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AppSettings {
    pub ai_provider: String,         // "ollama" | "claude" | "none"
    pub ollama_model: String,        // e.g. "llama3.2"
    pub claude_api_key: Option<String>,
    pub hotkey: String,              // e.g. "CmdOrCtrl+Shift+Space"
    pub indexed_folders: Vec<String>,
    pub theme: String,               // "dark" | "light" | "system"
    pub auto_start: bool,
    pub privacy_mode: bool,
    pub onboarded: bool,
}

impl Default for AppSettings {
    fn default() -> Self {
        Self {
            ai_provider: "ollama".to_string(),
            ollama_model: "llama3.2".to_string(),
            claude_api_key: None,
            hotkey: "CmdOrCtrl+Shift+Space".to_string(),
            indexed_folders: vec![],
            theme: "dark".to_string(),
            auto_start: false,
            privacy_mode: false,
            onboarded: false,
        }
    }
}

/// System status for the frontend
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SystemStatus {
    pub status: String,
    pub memory_count: u32,
    pub thought_count: u32,
    pub uptime_ms: i64,
    pub ai_provider: String,
    pub ai_available: bool,
    pub embedding_provider: String,
    pub learning_trend: String,
    pub indexed_files: u32,
    pub indexed_chunks: u32,
}

/// Main application state
pub struct AppState {
    pub engine: Arc<CognitiveEngine>,
    pub embeddings: Arc<EmbeddingModel>,
    pub persistence: Arc<BrainPersistence>,
    pub indexer: Arc<FileIndexer>,
    pub context: Arc<ContextManager>,
    pub ai_provider: RwLock<Option<Box<dyn AiProvider>>>,
    pub settings: RwLock<AppSettings>,
    pub shutdown: Notify,
}

impl AppState {
    /// Create a new application state
    pub fn new() -> Result<Self, String> {
        let persistence = BrainPersistence::new()?;
        let engine = CognitiveEngine::new(Some(CognitiveConfig::default()));
        let embeddings = EmbeddingModel::new();

        // Restore persisted memories
        match persistence.load_memories() {
            Ok(memories) => {
                let count = memories.len();
                for node in memories {
                    engine.memory.restore_node(node);
                }
                if count > 0 {
                    tracing::info!("Restored {} memories from database", count);
                }
            }
            Err(e) => {
                tracing::warn!("Failed to load memories: {}", e);
            }
        }

        // Restore Q-table
        match persistence.load_q_table() {
            Ok(entries) => {
                let count = entries.len();
                engine.learner.import_q_table(entries);
                if count > 0 {
                    tracing::info!("Restored {} Q-table entries", count);
                }
            }
            Err(e) => {
                tracing::warn!("Failed to load Q-table: {}", e);
            }
        }

        // Load settings
        let mut settings: AppSettings = match persistence.load_config("app_settings") {
            Ok(Some(json)) => serde_json::from_str(&json).unwrap_or_default(),
            _ => AppSettings::default(),
        };

        // Load Claude API key from Keychain (overrides any value in settings)
        if let Ok(Some(key)) = crate::keychain::get_secret("claude_api_key") {
            settings.claude_api_key = Some(key);
            tracing::info!("Loaded Claude API key from Keychain");
        }

        engine.set_running(true);

        let embeddings = Arc::new(embeddings);

        // Initialize file indexer
        let index_db = dirs::data_dir()
            .ok_or("No data dir")?
            .join("SuperBrain")
            .join("files.db");
        let indexer = FileIndexer::new(index_db, embeddings.clone())?;

        let ai_provider = Self::build_ai_provider(&settings);

        Ok(Self {
            engine: Arc::new(engine),
            embeddings,
            persistence: Arc::new(persistence),
            indexer: Arc::new(indexer),
            context: Arc::new(ContextManager::new()),
            ai_provider: RwLock::new(ai_provider),
            settings: RwLock::new(settings),
            shutdown: Notify::new(),
        })
    }

    /// Build an AI provider from current settings
    pub fn build_ai_provider(settings: &AppSettings) -> Option<Box<dyn AiProvider>> {
        match settings.ai_provider.as_str() {
            "ollama" => Some(Box::new(
                crate::ai::ollama::OllamaProvider::new(&settings.ollama_model),
            )),
            "claude" => {
                if let Some(ref key) = settings.claude_api_key {
                    if !key.is_empty() {
                        return Some(Box::new(crate::ai::claude::ClaudeProvider::new(key)));
                    }
                }
                None
            }
            _ => None,
        }
    }

    /// Refresh the AI provider (call after settings change)
    pub fn refresh_ai_provider(&self) {
        let settings = self.settings.read().clone();
        *self.ai_provider.write() = Self::build_ai_provider(&settings);
    }

    /// Persist current state to disk
    pub fn flush(&self) -> Result<(), String> {
        // Save memories
        let nodes = self.engine.memory.all_nodes();
        self.persistence.store_memories_batch(&nodes)?;

        // Save Q-table
        let q_entries = self.engine.learner.export_q_table();
        self.persistence.store_q_table(&q_entries)?;

        // Save settings
        let settings = self.settings.read().clone();
        let settings_json =
            serde_json::to_string(&settings).map_err(|e| format!("Serialize error: {}", e))?;
        self.persistence
            .store_config("app_settings", &settings_json)?;

        tracing::info!("State flushed to disk ({} memories)", nodes.len());
        Ok(())
    }
}
