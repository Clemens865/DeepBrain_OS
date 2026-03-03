//! Bridge adapters for integrating Brainwire cognitive memory with DeepBrain.
//!
//! Wraps DeepBrain's EmbeddingModel as Brainwire's EmbeddingProvider and
//! DeepBrain's AiProvider as Brainwire's LlmProvider.

use std::sync::Arc;

use async_trait::async_trait;
use parking_lot::RwLock;

use brainwire_core::{
    ConsolidationResult, ContextFrame, EncodingResult, LlmProvider, EmbeddingProvider,
    MemoryRecord, Result as BwResult, BrainwireError, SourceType, StmEntry,
};

use crate::ai::AiProvider;
use crate::brain::embeddings::EmbeddingModel;
use crate::deepbrain::llm_bridge::LlmBridge;
use crate::state::AppSettings;

// ---------------------------------------------------------------------------
// DeepBrainEmbedAdapter — wraps EmbeddingModel as Brainwire EmbeddingProvider
// ---------------------------------------------------------------------------

pub struct DeepBrainEmbedAdapter(pub Arc<EmbeddingModel>);

#[async_trait]
impl EmbeddingProvider for DeepBrainEmbedAdapter {
    async fn embed(&self, text: &str) -> BwResult<Vec<f32>> {
        self.0
            .embed(text)
            .await
            .map_err(|e| BrainwireError::Other(format!("Embedding error: {e}")))
    }

    fn dimensions(&self) -> usize {
        384
    }
}

// ---------------------------------------------------------------------------
// DeepBrainLlmAdapter — wraps AiProvider as Brainwire LlmProvider
// ---------------------------------------------------------------------------

/// Adapts DeepBrain's AI providers (Ollama/Claude/ruvllm) for Brainwire's LlmProvider trait.
///
/// Constructs a fresh provider for each call (matching think_impl's pattern)
/// to avoid holding a parking_lot lock across async boundaries.
pub struct DeepBrainLlmAdapter {
    settings: Arc<RwLock<AppSettings>>,
    llm: Arc<LlmBridge>,
}

impl DeepBrainLlmAdapter {
    pub fn new(settings: Arc<RwLock<AppSettings>>, llm: Arc<LlmBridge>) -> Self {
        Self { settings, llm }
    }

    /// Build a fresh AI provider from current settings and call generate.
    /// Returns None if no provider is available or the call fails.
    async fn try_generate(&self, prompt: &str) -> Option<String> {
        // Read settings (sync, fast) and drop the lock before any async work
        let provider: Option<Box<dyn AiProvider>> = {
            let settings = self.settings.read();
            match settings.ai_provider.as_str() {
                "ollama" => Some(Box::new(
                    crate::ai::ollama::OllamaProvider::new(&settings.ollama_model),
                )),
                "claude" => settings
                    .claude_api_key
                    .as_ref()
                    .filter(|k| !k.is_empty())
                    .map(|k| -> Box<dyn AiProvider> {
                        Box::new(crate::ai::claude::ClaudeProvider::new(k))
                    }),
                "ruvllm" => {
                    if self.llm.is_loaded() {
                        Some(Box::new(Arc::clone(&self.llm)))
                    } else {
                        None
                    }
                }
                _ => None,
            }
        };

        let prov = provider?;
        match prov.generate(prompt, &[], "").await {
            Ok(resp) => Some(resp.content),
            Err(e) => {
                tracing::debug!("Brainwire LLM adapter: AI provider error: {e}");
                None
            }
        }
    }
}

/// Deterministic fallback: extract key words from text as concepts.
fn fallback_extract_concepts(text: &str) -> Vec<String> {
    let stop_words: &[&str] = &[
        "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
        "have", "has", "had", "do", "does", "did", "will", "would", "could",
        "should", "may", "might", "shall", "can", "need", "dare", "ought",
        "used", "to", "of", "in", "for", "on", "with", "at", "by", "from",
        "as", "into", "through", "during", "before", "after", "above", "below",
        "between", "out", "off", "over", "under", "again", "further", "then",
        "once", "here", "there", "when", "where", "why", "how", "all", "both",
        "each", "few", "more", "most", "other", "some", "such", "no", "nor",
        "not", "only", "own", "same", "so", "than", "too", "very", "just",
        "because", "but", "and", "or", "if", "while", "about", "up", "it",
        "its", "this", "that", "these", "those", "i", "me", "my", "we", "our",
        "you", "your", "he", "him", "his", "she", "her", "they", "them", "their",
        "what", "which", "who", "whom",
    ];

    text.split_whitespace()
        .map(|w| w.trim_matches(|c: char| !c.is_alphanumeric()).to_lowercase())
        .filter(|w| w.len() > 2 && !stop_words.contains(&w.as_str()))
        .take(5)
        .collect()
}

/// Deterministic fallback: create a gist from text (first sentence or truncated).
fn fallback_gist(text: &str) -> String {
    let first_sentence = text
        .split_terminator(|c| c == '.' || c == '!' || c == '?')
        .next()
        .unwrap_or(text);
    if first_sentence.len() > 200 {
        // Walk back to a char boundary to avoid slicing into multi-byte characters
        let mut end = 197;
        while end > 0 && !first_sentence.is_char_boundary(end) {
            end -= 1;
        }
        format!("{}...", &first_sentence[..end])
    } else {
        first_sentence.to_string()
    }
}

#[async_trait]
impl LlmProvider for DeepBrainLlmAdapter {
    async fn encode(&self, entries: &[StmEntry]) -> BwResult<Vec<EncodingResult>> {
        let mut results = Vec::with_capacity(entries.len());

        for entry in entries {
            let text = &entry.raw_text;

            // Try AI provider for richer encoding
            let prompt = format!(
                "Analyze this text and return a JSON object with these fields:\n\
                 - \"gist\": a 1-sentence summary\n\
                 - \"concepts\": array of 3-5 key concepts (single words or short phrases)\n\
                 - \"salience\": importance score 0.0-1.0\n\
                 - \"emotional_valence\": emotional tone -1.0 to 1.0 (negative=sad, positive=happy)\n\
                 - \"learned_rules\": array of generalizable patterns or rules (can be empty)\n\n\
                 Text: \"{text}\"\n\nRespond with ONLY the JSON object, no markdown."
            );

            if let Some(response) = self.try_generate(&prompt).await {
                if let Ok(parsed) = serde_json::from_str::<serde_json::Value>(&response)
                    .or_else(|_| {
                        // Try to extract JSON from markdown code block
                        let trimmed = response
                            .trim()
                            .strip_prefix("```json")
                            .or_else(|| response.trim().strip_prefix("```"))
                            .unwrap_or(&response)
                            .strip_suffix("```")
                            .unwrap_or(&response)
                            .trim();
                        serde_json::from_str(trimmed)
                    })
                {
                    results.push(EncodingResult {
                        gist: parsed["gist"]
                            .as_str()
                            .unwrap_or(&fallback_gist(text))
                            .to_string(),
                        concepts: parsed["concepts"]
                            .as_array()
                            .map(|arr| {
                                arr.iter()
                                    .filter_map(|v| v.as_str().map(String::from))
                                    .collect()
                            })
                            .unwrap_or_else(|| fallback_extract_concepts(text)),
                        learned_rules: parsed["learned_rules"]
                            .as_array()
                            .map(|arr| {
                                arr.iter()
                                    .filter_map(|v| v.as_str().map(String::from))
                                    .collect()
                            })
                            .unwrap_or_default(),
                        salience: parsed["salience"]
                            .as_f64()
                            .unwrap_or(0.5) as f32,
                        emotional_valence: parsed["emotional_valence"]
                            .as_f64()
                            .unwrap_or(0.0) as f32,
                        source_type: SourceType::Observation,
                        confidence: 0.8,
                    });
                    continue;
                }
            }

            // Fallback: deterministic extraction
            results.push(EncodingResult {
                gist: fallback_gist(text),
                concepts: fallback_extract_concepts(text),
                learned_rules: vec![],
                salience: 0.5,
                emotional_valence: 0.0,
                source_type: SourceType::Observation,
                confidence: 0.5,
            });
        }

        Ok(results)
    }

    async fn consolidate(&self, memories: &[MemoryRecord]) -> BwResult<ConsolidationResult> {
        if memories.is_empty() {
            return Ok(ConsolidationResult {
                gist: String::new(),
                concepts: vec![],
                learned_rules: vec![],
                salience: 0.0,
                confidence: 0.0,
            });
        }

        let gists: Vec<&str> = memories.iter().map(|m| m.gist.as_str()).collect();
        let prompt = format!(
            "Merge these related memories into a single consolidated summary.\n\
             Return a JSON object with:\n\
             - \"gist\": merged summary (1-2 sentences)\n\
             - \"concepts\": combined key concepts (3-7 items)\n\
             - \"learned_rules\": generalizable patterns extracted\n\
             - \"salience\": importance 0.0-1.0\n\
             - \"confidence\": confidence 0.0-1.0\n\n\
             Memories:\n{}\n\nRespond with ONLY the JSON object.",
            gists.join("\n- ")
        );

        if let Some(response) = self.try_generate(&prompt).await {
            if let Ok(parsed) = serde_json::from_str::<serde_json::Value>(&response)
                .or_else(|_| {
                    let trimmed = response
                        .trim()
                        .strip_prefix("```json")
                        .or_else(|| response.trim().strip_prefix("```"))
                        .unwrap_or(&response)
                        .strip_suffix("```")
                        .unwrap_or(&response)
                        .trim();
                    serde_json::from_str(trimmed)
                })
            {
                return Ok(ConsolidationResult {
                    gist: parsed["gist"]
                        .as_str()
                        .unwrap_or(gists.first().unwrap_or(&""))
                        .to_string(),
                    concepts: parsed["concepts"]
                        .as_array()
                        .map(|arr| {
                            arr.iter()
                                .filter_map(|v| v.as_str().map(String::from))
                                .collect()
                        })
                        .unwrap_or_default(),
                    learned_rules: parsed["learned_rules"]
                        .as_array()
                        .map(|arr| {
                            arr.iter()
                                .filter_map(|v| v.as_str().map(String::from))
                                .collect()
                        })
                        .unwrap_or_default(),
                    salience: parsed["salience"].as_f64().unwrap_or(0.6) as f32,
                    confidence: parsed["confidence"].as_f64().unwrap_or(0.7) as f32,
                });
            }
        }

        // Fallback: simple merge
        let all_concepts: Vec<String> = memories
            .iter()
            .flat_map(|m| m.concepts.clone())
            .collect::<std::collections::HashSet<_>>()
            .into_iter()
            .take(7)
            .collect();

        Ok(ConsolidationResult {
            gist: gists.first().map(|g| g.to_string()).unwrap_or_default(),
            concepts: all_concepts,
            learned_rules: vec![],
            salience: memories.iter().map(|m| m.salience).sum::<f32>() / memories.len() as f32,
            confidence: 0.5,
        })
    }

    async fn extract_semantic(&self, episodes: &[MemoryRecord]) -> BwResult<ConsolidationResult> {
        if episodes.is_empty() {
            return Ok(ConsolidationResult {
                gist: String::new(),
                concepts: vec![],
                learned_rules: vec![],
                salience: 0.0,
                confidence: 0.0,
            });
        }

        let gists: Vec<&str> = episodes.iter().map(|m| m.gist.as_str()).collect();
        let prompt = format!(
            "Extract timeless, general knowledge from these episodic memories.\n\
             Focus on facts, patterns, and rules that are always true (not time-specific).\n\
             Return a JSON object with:\n\
             - \"gist\": the general knowledge statement (1-2 sentences)\n\
             - \"concepts\": key concepts (3-7 items)\n\
             - \"learned_rules\": generalizable rules and patterns\n\
             - \"salience\": importance 0.0-1.0\n\
             - \"confidence\": confidence 0.0-1.0\n\n\
             Episodes:\n{}\n\nRespond with ONLY the JSON object.",
            gists.join("\n- ")
        );

        if let Some(response) = self.try_generate(&prompt).await {
            if let Ok(parsed) = serde_json::from_str::<serde_json::Value>(&response)
                .or_else(|_| {
                    let trimmed = response
                        .trim()
                        .strip_prefix("```json")
                        .or_else(|| response.trim().strip_prefix("```"))
                        .unwrap_or(&response)
                        .strip_suffix("```")
                        .unwrap_or(&response)
                        .trim();
                    serde_json::from_str(trimmed)
                })
            {
                return Ok(ConsolidationResult {
                    gist: parsed["gist"]
                        .as_str()
                        .unwrap_or(gists.first().unwrap_or(&""))
                        .to_string(),
                    concepts: parsed["concepts"]
                        .as_array()
                        .map(|arr| {
                            arr.iter()
                                .filter_map(|v| v.as_str().map(String::from))
                                .collect()
                        })
                        .unwrap_or_default(),
                    learned_rules: parsed["learned_rules"]
                        .as_array()
                        .map(|arr| {
                            arr.iter()
                                .filter_map(|v| v.as_str().map(String::from))
                                .collect()
                        })
                        .unwrap_or_default(),
                    salience: parsed["salience"].as_f64().unwrap_or(0.7) as f32,
                    confidence: parsed["confidence"].as_f64().unwrap_or(0.7) as f32,
                });
            }
        }

        // Fallback: delegate to consolidate
        self.consolidate(episodes).await
    }

    async fn reconstruct(
        &self,
        memory: &MemoryRecord,
        query: &str,
        _current_context: &ContextFrame,
    ) -> BwResult<String> {
        let prompt = format!(
            "Reconstruct this memory in the context of the query.\n\
             Memory gist: {}\n\
             Key concepts: {}\n\
             Query: {}\n\n\
             Provide a natural, detailed reconstruction of this memory \
             as it relates to the query. 2-3 sentences max.",
            memory.gist,
            memory.concepts.join(", "),
            query,
        );

        if let Some(response) = self.try_generate(&prompt).await {
            return Ok(response);
        }

        // Fallback: return gist as-is
        Ok(memory.gist.clone())
    }
}
