//! Pluggable connector architecture for bootstrap data sources.
//!
//! Each connector encapsulates detection, configuration, and import logic
//! for a single data source (WhatsApp, browser history, Claude conversations, etc.).

pub mod browser;
pub mod claude_code;
pub mod claude_desktop;
pub mod claude_history;
pub mod claude_memory;
pub mod claude_plans;
pub mod deepnote;
pub mod file_index;
pub mod whatsapp;

use std::collections::HashMap;

use serde::{Deserialize, Serialize};

use crate::indexer::bootstrap::SourceResult;
use crate::state::AppState;

// ---- Types ----

/// Result of auto-detecting whether a data source is available.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DetectionResult {
    pub available: bool,
    pub path: Option<String>,
    pub details: Option<String>,
}

/// Per-connector user configuration (persisted in AppSettings).
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ConnectorConfig {
    pub enabled: Option<bool>,
    pub path_override: Option<String>,
    pub importance_override: Option<f64>,
    #[serde(default)]
    pub extra: serde_json::Value,
}

/// Connector metadata exposed to the frontend.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConnectorInfo {
    pub id: String,
    pub name: String,
    pub description: String,
    pub icon: String,
    pub detected: DetectionResult,
    pub enabled: bool,
    pub memory_type: String,
    pub default_importance: f64,
}

// ---- Trait ----

#[async_trait::async_trait]
pub trait Connector: Send + Sync {
    /// Unique identifier (matches the source string used in bootstrap, e.g. "whatsapp").
    fn id(&self) -> &str;
    /// Human-readable display name.
    fn name(&self) -> &str;
    /// Short description shown in the UI.
    fn description(&self) -> &str;
    /// Emoji or icon identifier.
    fn icon(&self) -> &str;
    /// Whether this connector is enabled by default when detected.
    fn default_enabled(&self) -> bool;
    /// Memory type used for stored memories (e.g. "episodic", "semantic").
    fn memory_type(&self) -> &str;
    /// Default importance score for memories from this source.
    fn default_importance(&self) -> f64;
    /// Detect if this data source is available on the current machine.
    fn detect(&self) -> DetectionResult;
    /// Run the import, returning per-source stats.
    async fn import(
        &self,
        state: &AppState,
        app_handle: Option<&tauri::AppHandle>,
        config: &ConnectorConfig,
    ) -> SourceResult;
}

// ---- Registry ----

/// Stateless registry of all known connectors.
/// Created fresh per call — connectors are unit structs with no mutable state.
pub struct ConnectorRegistry {
    connectors: Vec<Box<dyn Connector>>,
}

impl ConnectorRegistry {
    /// Build a registry with all built-in connectors.
    pub fn new() -> Self {
        let connectors: Vec<Box<dyn Connector>> = vec![
            Box::new(file_index::FileIndexConnector),
            Box::new(claude_code::ClaudeCodeConnector),
            Box::new(claude_memory::ClaudeMemoryConnector),
            Box::new(claude_plans::ClaudePlansConnector),
            Box::new(claude_desktop::ClaudeDesktopConnector),
            Box::new(claude_history::ClaudeHistoryConnector),
            Box::new(whatsapp::WhatsAppConnector),
            Box::new(browser::BrowserConnector),
            Box::new(deepnote::DeepnoteConnector),
        ];
        Self { connectors }
    }

    /// List all connectors with their detection status and user config applied.
    pub fn list(&self, user_config: &HashMap<String, ConnectorConfig>) -> Vec<ConnectorInfo> {
        self.connectors
            .iter()
            .map(|c| {
                let detected = c.detect();
                let cfg = user_config.get(c.id());
                let enabled = cfg
                    .and_then(|c| c.enabled)
                    .unwrap_or_else(|| c.default_enabled() && detected.available);
                ConnectorInfo {
                    id: c.id().to_string(),
                    name: c.name().to_string(),
                    description: c.description().to_string(),
                    icon: c.icon().to_string(),
                    detected,
                    enabled,
                    memory_type: c.memory_type().to_string(),
                    default_importance: c.default_importance(),
                }
            })
            .collect()
    }

    /// Get a connector by ID.
    pub fn get(&self, id: &str) -> Option<&dyn Connector> {
        self.connectors.iter().find(|c| c.id() == id).map(|c| c.as_ref())
    }

    /// Run imports for all requested sources, respecting user config.
    ///
    /// `sources` works like before: vec of source IDs, or `["all"]` for everything.
    pub async fn run_import(
        &self,
        state: &AppState,
        app_handle: Option<&tauri::AppHandle>,
        sources: &[String],
        user_config: &HashMap<String, ConnectorConfig>,
    ) -> Vec<SourceResult> {
        let run_all = sources.contains(&"all".to_string());
        let mut results = Vec::new();

        for connector in &self.connectors {
            let id = connector.id();

            // Check if this source was requested
            if !run_all && !sources.iter().any(|s| s == id) {
                continue;
            }

            // Check if user has disabled this connector
            let cfg = user_config.get(id).cloned().unwrap_or_default();
            let enabled = cfg.enabled.unwrap_or(true);
            if !enabled {
                tracing::info!("Connector '{}' is disabled, skipping", id);
                continue;
            }

            let result = connector.import(state, app_handle, &cfg).await;
            results.push(result);
        }

        results
    }
}
