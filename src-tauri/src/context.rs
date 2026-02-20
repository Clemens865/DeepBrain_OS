//! Context awareness for DeepBrain
//!
//! Monitors clipboard and provides contextual boosts for search.

use parking_lot::RwLock;
use serde::{Deserialize, Serialize};

use crate::brain::utils::now_millis;

/// Recent clipboard entry
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClipboardEntry {
    pub content: String,
    pub timestamp: i64,
}

/// Context manager tracks recent activity for search relevance boosting
pub struct ContextManager {
    /// Recent clipboard entries
    clipboard_history: RwLock<Vec<ClipboardEntry>>,
    /// Maximum clipboard history entries
    max_history: usize,
}

impl ContextManager {
    pub fn new() -> Self {
        Self {
            clipboard_history: RwLock::new(Vec::new()),
            max_history: 50,
        }
    }

    /// Record a clipboard entry
    pub fn record_clipboard(&self, content: String) {
        let entry = ClipboardEntry {
            content,
            timestamp: now_millis(),
        };

        let mut history = self.clipboard_history.write();
        history.insert(0, entry);
        history.truncate(self.max_history);
    }

    /// Get recent clipboard entries
    pub fn recent_clipboard(&self, limit: usize) -> Vec<ClipboardEntry> {
        self.clipboard_history
            .read()
            .iter()
            .take(limit)
            .cloned()
            .collect()
    }

    /// Get the most recent clipboard content
    pub fn last_clipboard(&self) -> Option<String> {
        self.clipboard_history
            .read()
            .first()
            .map(|e| e.content.clone())
    }
}

impl Default for ContextManager {
    fn default() -> Self {
        Self::new()
    }
}
