//! Application state management for DeepBrain
//!
//! Wraps CognitiveEngine + EmbeddingModel + Persistence in Arc for Tauri managed state.

use std::sync::Arc;

use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use tokio::sync::Notify;

use std::collections::HashMap;

use crate::activity::ActivityObserver;
use crate::ai::AiProvider;
use crate::brain::cognitive::CognitiveEngine;
use crate::brain::embeddings::EmbeddingModel;
use crate::brain::persistence::BrainPersistence;
use crate::brain::types::CognitiveConfig;
use crate::context::ContextManager;
use crate::deepbrain::compress_bridge::CompressBridge;
use crate::deepbrain::gnn_bridge::GnnBridge;
use crate::deepbrain::graph_bridge::GraphBridge;
use crate::deepbrain::llm_bridge::LlmBridge;
use crate::deepbrain::nervous_bridge::NervousBridge;
use crate::deepbrain::sona_bridge::SonaBridge;
use crate::deepbrain::vector_store::DeepBrainVectorStore;
use crate::indexer::FileIndexer;
use crate::indexer::browser::BrowserIndexer;
use crate::indexer::connectors::ConnectorConfig;
use crate::indexer::email::EmailIndexer;

/// Application settings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AppSettings {
    pub ai_provider: String,         // "ollama" | "claude" | "ruvllm" | "none"
    pub ollama_model: String,        // e.g. "llama3.1:8b"
    pub ruvllm_model: Option<String>, // e.g. "Qwen/Qwen2.5-1.5B-Instruct" or local GGUF path
    pub claude_api_key: Option<String>,
    pub hotkey: String,              // e.g. "CmdOrCtrl+Shift+Space"
    pub indexed_folders: Vec<String>,
    pub theme: String,               // "dark" | "light" | "system"
    pub auto_start: bool,
    pub privacy_mode: bool,
    pub onboarded: bool,
    pub api_port: Option<u16>,
    pub api_token: Option<String>,
    /// Email accounts to index (empty = all accounts).
    #[serde(default)]
    pub email_accounts: Vec<String>,
    /// Activity tracking: master toggle
    #[serde(default = "default_true")]
    pub activity_tracking_enabled: bool,
    /// Activity tracking: apps to exclude from window title capture
    #[serde(default)]
    pub activity_excluded_apps: Vec<String>,
    /// Activity tracking: capture browser tab titles (future use)
    #[serde(default = "default_true")]
    pub activity_track_browser: bool,
    /// Activity tracking: capture terminal window titles
    #[serde(default)]
    pub activity_track_terminal: bool,
    /// Activity tracking: days to retain events before pruning
    #[serde(default = "default_30")]
    pub activity_retention_days: u32,
    /// Per-connector configuration (enable/disable, path overrides, importance)
    #[serde(default)]
    pub connector_config: HashMap<String, ConnectorConfig>,
}

fn default_true() -> bool {
    true
}

fn default_30() -> u32 {
    30
}

impl Default for AppSettings {
    fn default() -> Self {
        Self {
            ai_provider: "ollama".to_string(),
            ollama_model: "llama3.1:8b".to_string(),
            ruvllm_model: None,
            claude_api_key: None,
            hotkey: "CmdOrCtrl+Shift+Space".to_string(),
            indexed_folders: vec![],
            theme: "dark".to_string(),
            auto_start: false,
            privacy_mode: false,
            onboarded: false,
            api_port: None,
            api_token: None,
            email_accounts: vec![],
            activity_tracking_enabled: true,
            activity_excluded_apps: vec![],
            activity_track_browser: true,
            activity_track_terminal: false,
            activity_retention_days: 30,
            connector_config: HashMap::new(),
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
    pub indexed_emails: u32,
}

/// Main application state
pub struct AppState {
    pub engine: Arc<CognitiveEngine>,
    pub embeddings: Arc<EmbeddingModel>,
    pub persistence: Arc<BrainPersistence>,
    pub indexer: Arc<FileIndexer>,
    pub email_indexer: Arc<EmailIndexer>,
    pub browser_indexer: Arc<BrowserIndexer>,
    pub context: Arc<ContextManager>,
    pub sona: Arc<SonaBridge>,
    pub nervous: Arc<NervousBridge>,
    pub llm: Arc<LlmBridge>,
    pub graph: Arc<GraphBridge>,
    pub gnn: Arc<GnnBridge>,
    pub compressor: Arc<CompressBridge>,
    pub vector_store: Arc<DeepBrainVectorStore>,
    pub activity: Arc<ActivityObserver>,
    pub ai_provider: RwLock<Option<Box<dyn AiProvider>>>,
    pub settings: RwLock<AppSettings>,
    pub shutdown: Notify,
}

impl AppState {
    /// Create a new application state
    pub fn new() -> Result<Self, String> {
        let persistence = BrainPersistence::new()?;
        let embeddings = EmbeddingModel::new();

        // --- Initialize DeepBrain vector store (ruvector-core / redb) ---
        let data_dir = dirs::data_dir()
            .ok_or("No data dir")?
            .join("DeepBrain");
        let vector_store = Arc::new(
            DeepBrainVectorStore::open_or_create(&data_dir)
                .map_err(|e| format!("Failed to open DeepBrain vector store: {}", e))?,
        );
        tracing::info!(
            "DeepBrain vector store ready at {:?} ({} vectors)",
            data_dir,
            vector_store.status().total_vectors,
        );

        // Create shared SONA bridge
        let sona = Arc::new(SonaBridge::new());

        // Create nervous system bridge (Hopfield, routing, predictive coding)
        let nervous = Arc::new(NervousBridge::new());

        // Create local LLM bridge (ruvllm CandleBackend — model loaded on demand)
        let llm = Arc::new(LlmBridge::new());

        // Create knowledge graph bridge (persistent)
        let graph = Arc::new(
            GraphBridge::open(&data_dir).unwrap_or_else(|e| {
                tracing::warn!("Failed to open graph database, using in-memory: {}", e);
                GraphBridge::in_memory()
            }),
        );
        tracing::info!(
            "Knowledge graph ready ({} nodes, {} edges)",
            graph.stats().node_count,
            graph.stats().edge_count,
        );

        // Create GNN reranker bridge (restore weights if persisted)
        let gnn = Arc::new(match persistence.load_blob("gnn_weights") {
            Ok(Some(bytes)) => {
                match std::str::from_utf8(&bytes).ok().and_then(|json| GnnBridge::from_persisted(json).ok()) {
                    Some(bridge) => {
                        tracing::info!("Restored GNN reranker weights");
                        bridge
                    }
                    None => {
                        tracing::warn!("Failed to deserialize GNN weights, using fresh layer");
                        GnnBridge::new()
                    }
                }
            }
            _ => GnnBridge::new(),
        });

        // Create tensor compression bridge
        let compressor = Arc::new(CompressBridge::new());

        // Create cognitive engine with vector-store-backed memory + SONA + nervous system
        let engine = CognitiveEngine::with_all(
            Some(CognitiveConfig::default()),
            Arc::clone(&vector_store),
            Arc::clone(&sona),
            Arc::clone(&nervous),
        );

        // Restore persisted memories (batch insert into vector store + DashMap)
        match persistence.load_memories() {
            Ok(memories) => {
                let count = memories.len();
                engine.memory.restore_nodes_batch(memories);
                if count > 0 {
                    tracing::info!("Restored {} memories from database", count);
                }
            }
            Err(e) => {
                tracing::warn!("Failed to load memories: {}", e);
            }
        }

        // Restore SONA patterns from blob store
        match persistence.load_blob("sona_patterns") {
            Ok(Some(bytes)) => {
                match serde_json::from_slice::<Vec<crate::deepbrain::sona_bridge::SerializedPattern>>(&bytes) {
                    Ok(patterns) => {
                        let count = patterns.len();
                        sona.import_patterns(&patterns);
                        if count > 0 {
                            tracing::info!("Restored {} SONA patterns", count);
                        }
                    }
                    Err(e) => {
                        tracing::warn!("Failed to deserialize SONA patterns: {}", e);
                    }
                }
            }
            Ok(None) => {}
            Err(e) => {
                tracing::warn!("Failed to load SONA patterns: {}", e);
            }
        }

        // Load settings
        let mut settings: AppSettings = match persistence.load_config("app_settings") {
            Ok(Some(json)) => serde_json::from_str(&json).unwrap_or_default(),
            _ => AppSettings::default(),
        };

        // Load secrets from Keychain (overrides any value in settings)
        if let Ok(Some(key)) = crate::keychain::get_secret("claude_api_key") {
            settings.claude_api_key = Some(key);
            tracing::info!("Loaded Claude API key from Keychain");
        }
        if let Ok(Some(token)) = crate::keychain::get_secret("api_token") {
            settings.api_token = Some(token.clone());
            tracing::info!("Loaded API token from Keychain");
            // Sync token file so external tools (MCP servers, CLI) can read it
            let token_file = data_dir.join(".api_token");
            if let Err(e) = std::fs::write(&token_file, &token) {
                tracing::warn!("Failed to sync API token file: {}", e);
            }
            #[cfg(unix)]
            {
                use std::os::unix::fs::PermissionsExt;
                let _ = std::fs::set_permissions(&token_file, std::fs::Permissions::from_mode(0o600));
            }
        }

        // Ensure we always have an API token in settings (generate if Keychain was empty)
        if settings.api_token.is_none() {
            let generated = uuid::Uuid::new_v4().to_string();
            let _ = crate::keychain::store_secret("api_token", &generated);
            // Also write token to data directory as a fallback (Keychain may be unavailable)
            let token_file = data_dir.join(".api_token");
            if let Err(e) = std::fs::write(&token_file, &generated) {
                tracing::warn!("Failed to write API token file: {}", e);
            } else {
                // Restrict permissions to owner only (chmod 600)
                #[cfg(unix)]
                {
                    use std::os::unix::fs::PermissionsExt;
                    let _ = std::fs::set_permissions(&token_file, std::fs::Permissions::from_mode(0o600));
                }
                tracing::info!(
                    "API token written to {:?} (and Keychain if available)",
                    token_file,
                );
            }
            settings.api_token = Some(generated);
        }

        engine.set_running(true);

        let embeddings = Arc::new(embeddings);
        let engine = Arc::new(engine);
        let persistence = Arc::new(persistence);

        // Initialize file indexer with shared vector store
        let index_db = dirs::data_dir()
            .ok_or("No data dir")?
            .join("DeepBrain")
            .join("files.db");
        let indexer = FileIndexer::with_vector_store(
            index_db.clone(),
            embeddings.clone(),
            Arc::clone(&vector_store),
        )?;

        // Initialize email indexer with shared vector store
        let email_indexer = EmailIndexer::with_vector_store(
            index_db,
            embeddings.clone(),
            Arc::clone(&vector_store),
        )?;

        // Initialize browser history indexer
        let browser_indexer = Arc::new(BrowserIndexer::new(
            Arc::clone(&engine),
            embeddings.clone(),
            Some(Arc::clone(&vector_store)),
            Arc::clone(&persistence),
            Arc::clone(&graph),
        ));

        // Configure email account filter from settings
        if !settings.email_accounts.is_empty() {
            email_indexer.set_allowed_accounts(settings.email_accounts.clone());
            tracing::info!("Email indexer limited to accounts: {:?}", settings.email_accounts);
        }

        // Initialize activity observer
        let activity = Arc::new(ActivityObserver::new(&data_dir)?);
        // Apply activity settings from loaded AppSettings
        activity.update_settings(crate::activity::ActivitySettings {
            enabled: settings.activity_tracking_enabled,
            excluded_apps: settings.activity_excluded_apps.clone(),
            track_browser: settings.activity_track_browser,
            track_terminal: settings.activity_track_terminal,
            retention_days: settings.activity_retention_days,
        });

        let ai_provider = Self::build_ai_provider(&settings);

        let indexer = Arc::new(indexer);

        // HNSW index is rebuilt from redb on VectorDB::new().
        tracing::info!("File indexer ready (VectorDB-backed)");

        Ok(Self {
            engine,
            embeddings,
            persistence,
            indexer,
            email_indexer: Arc::new(email_indexer),
            browser_indexer,
            context: Arc::new(ContextManager::new()),
            sona,
            nervous,
            llm,
            graph,
            gnn,
            compressor,
            vector_store,
            activity,
            ai_provider: RwLock::new(ai_provider),
            settings: RwLock::new(settings),
            shutdown: Notify::new(),
        })
    }

    /// Build an AI provider from current settings.
    ///
    /// For "ruvllm", the LlmBridge must have a model loaded separately
    /// (via the `load_local_model` command). This returns None if no model is loaded.
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
            // "ruvllm" is handled specially — uses the shared LlmBridge
            // which implements AiProvider when wrapped in Arc.
            // Caller should use `self.build_ai_provider_with_llm()` instead.
            _ => None,
        }
    }

    /// Build an AI provider, including support for the local LLM backend.
    pub fn build_ai_provider_with_llm(&self) -> Option<Box<dyn AiProvider>> {
        let settings = self.settings.read();
        match settings.ai_provider.as_str() {
            "ruvllm" => {
                if self.llm.is_loaded() {
                    Some(Box::new(Arc::clone(&self.llm)))
                } else {
                    tracing::warn!("ruvllm selected but no model loaded");
                    None
                }
            }
            _ => Self::build_ai_provider(&settings),
        }
    }

    /// Refresh the AI provider (call after settings change)
    pub fn refresh_ai_provider(&self) {
        *self.ai_provider.write() = self.build_ai_provider_with_llm();
    }

    /// Persist current state to disk
    pub fn flush(&self) -> Result<(), String> {
        // Save memories
        let nodes = self.engine.memory.all_nodes();
        self.persistence.store_memories_batch(&nodes)?;

        // Save settings
        let settings = self.settings.read().clone();
        let settings_json =
            serde_json::to_string(&settings).map_err(|e| format!("Serialize error: {}", e))?;
        self.persistence
            .store_config("app_settings", &settings_json)?;

        // Save GNN reranker weights
        match self.gnn.to_json() {
            Ok(json) => {
                let _ = self.persistence.store_blob("gnn_weights", json.as_bytes());
            }
            Err(e) => {
                tracing::warn!("Failed to serialize GNN weights: {}", e);
            }
        }

        // Save SONA learned patterns
        let patterns = self.sona.export_patterns();
        if !patterns.is_empty() {
            match serde_json::to_vec(&patterns) {
                Ok(bytes) => {
                    let _ = self.persistence.store_blob("sona_patterns", &bytes);
                }
                Err(e) => {
                    tracing::warn!("Failed to serialize SONA patterns: {}", e);
                }
            }
        }

        tracing::info!("State flushed to disk ({} memories)", nodes.len());
        Ok(())
    }
}
