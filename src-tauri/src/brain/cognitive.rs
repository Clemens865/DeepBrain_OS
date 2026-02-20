//! Cognitive processing engine for DeepBrain (Tauri port)

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;

use parking_lot::RwLock;
use serde::{Deserialize, Serialize};

use crate::brain::memory::NativeMemory;
use crate::brain::types::{CognitiveConfig, CognitiveStats, Thought, ThoughtType};
use crate::brain::utils::{generate_id, now_millis};
use crate::deepbrain::nervous_bridge::NervousBridge;
use crate::deepbrain::sona_bridge::SonaBridge;

/// Goal tracking
#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct Goal {
    pub id: String,
    pub description: String,
    pub priority: f64,
    pub progress: f64,
    pub status: GoalStatus,
    pub created_at: i64,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub(crate) enum GoalStatus {
    Pending,
    Active,
    Completed,
    Failed,
}

/// Belief with confidence
#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct Belief {
    pub id: String,
    pub content: String,
    pub confidence: f64,
    pub source: String,
    pub timestamp: i64,
}

/// The main cognitive engine
pub struct CognitiveEngine {
    /// Native memory system
    pub memory: Arc<NativeMemory>,
    /// SONA adaptive learning engine (replaces Q-learning NativeLearner)
    pub sona: Arc<SonaBridge>,
    /// Bio-inspired nervous system (Hopfield, routing, predictive coding)
    pub nervous: Arc<NervousBridge>,
    /// Thought stream
    thoughts: RwLock<Vec<Thought>>,
    /// Goals
    goals: RwLock<Vec<Goal>>,
    /// Beliefs
    beliefs: RwLock<Vec<Belief>>,
    /// Configuration
    config: RwLock<CognitiveConfig>,
    /// Running state
    running: AtomicBool,
    /// Cycle counter
    cycle_count: AtomicU64,
    /// Start time
    start_time: i64,
}

impl CognitiveEngine {
    /// Create a new cognitive engine
    pub fn new(config: Option<CognitiveConfig>) -> Self {
        let cfg = config.unwrap_or_default();
        let dim = cfg.dimensions as usize;

        Self {
            memory: Arc::new(NativeMemory::new(cfg.dimensions)),
            sona: Arc::new(SonaBridge::with_dim(dim)),
            nervous: Arc::new(NervousBridge::new()),
            thoughts: RwLock::new(Vec::with_capacity(1000)),
            goals: RwLock::new(Vec::new()),
            beliefs: RwLock::new(Vec::new()),
            config: RwLock::new(cfg),
            running: AtomicBool::new(false),
            cycle_count: AtomicU64::new(0),
            start_time: now_millis(),
        }
    }

    /// Create a cognitive engine with all external dependencies injected.
    pub fn with_all(
        config: Option<CognitiveConfig>,
        store: std::sync::Arc<crate::deepbrain::vector_store::DeepBrainVectorStore>,
        sona: Arc<SonaBridge>,
        nervous: Arc<NervousBridge>,
    ) -> Self {
        let cfg = config.unwrap_or_default();

        Self {
            memory: Arc::new(NativeMemory::with_vector_store(cfg.dimensions, store)),
            sona,
            nervous,
            thoughts: RwLock::new(Vec::with_capacity(1000)),
            goals: RwLock::new(Vec::new()),
            beliefs: RwLock::new(Vec::new()),
            config: RwLock::new(cfg),
            running: AtomicBool::new(false),
            cycle_count: AtomicU64::new(0),
            start_time: now_millis(),
        }
    }

    /// Store a memory with a text embedding (uses f32 vectors directly)
    pub fn remember_with_embedding(
        &self,
        content: String,
        vector: Vec<f32>,
        memory_type: String,
        importance: Option<f64>,
    ) -> Result<String, String> {
        let imp = importance.unwrap_or(0.5);

        // Store pattern in Hopfield for associative recall
        self.nervous.hopfield_store(&vector);

        // Update predictive model with this new input
        self.nervous.update_prediction(&vector);

        self.memory
            .store_f32(content, vector, memory_type, imp)
    }

    /// Store a memory (legacy f64 interface)
    pub fn remember(
        &self,
        content: String,
        vector: Vec<f64>,
        memory_type: String,
        importance: Option<f64>,
    ) -> Result<String, String> {
        let imp = importance.unwrap_or(0.5);
        self.memory.store(content, vector, memory_type, imp)
    }

    /// Recall memories by similarity (f32 interface)
    pub fn recall_f32(
        &self,
        query_vector: &[f32],
        k: Option<u32>,
        memory_types: Option<Vec<String>>,
    ) -> Result<Vec<RecallResult>, String> {
        let results =
            self.memory
                .search_f32(query_vector, k.unwrap_or(10), memory_types, Some(0.2))?;

        Ok(results
            .into_iter()
            .map(|r| RecallResult {
                id: r.id,
                content: r.content,
                similarity: r.similarity,
                memory_type: r.memory_type,
            })
            .collect())
    }

    /// Recall memories by similarity (legacy f64 interface)
    pub fn recall(
        &self,
        query_vector: Vec<f64>,
        k: Option<u32>,
        memory_types: Option<Vec<String>>,
    ) -> Result<Vec<RecallResult>, String> {
        let results = self
            .memory
            .search(query_vector, k.unwrap_or(10), memory_types, Some(0.2))?;

        Ok(results
            .into_iter()
            .map(|r| RecallResult {
                id: r.id,
                content: r.content,
                similarity: r.similarity,
                memory_type: r.memory_type,
            })
            .collect())
    }

    /// Learn from an experience by recording a SONA trajectory.
    ///
    /// The state/next_state f64 vectors are converted to f32 and recorded as a
    /// trajectory. The reward is mapped to a quality signal in [0.0, 1.0].
    pub fn learn(
        &self,
        state: Vec<f64>,
        _action: u32,
        reward: f64,
        next_state: Vec<f64>,
        _done: bool,
    ) -> Result<LearnResult, String> {
        let state_f32: Vec<f32> = state.iter().map(|&x| x as f32).collect();
        let next_f32: Vec<f32> = next_state.iter().map(|&x| x as f32).collect();

        // Map reward to quality in [0, 1]: shift from [-1, 1] to [0, 1]
        let quality = ((reward + 1.0) / 2.0).clamp(0.0, 1.0) as f32;

        self.sona.record_trajectory(&state_f32, &next_f32, quality);

        let thought = self.generate_thought(
            if reward > 0.0 {
                ThoughtType::Evaluation
            } else {
                ThoughtType::Reflection
            },
            format!(
                "SONA trajectory recorded: reward={:.2}, quality={:.2}",
                reward, quality
            ),
            reward.abs().min(1.0),
        );

        Ok(LearnResult {
            success: reward > 0.0,
            reward,
            td_error: 0.0, // No TD error in SONA's continuous learning
            thought_id: thought.id,
            insights: vec![],
        })
    }

    /// Select an action for a given state using SONA pattern matching.
    ///
    /// Returns the index of the best-matching pattern (mod 100) or 0 if none found.
    pub fn act(&self, state: Vec<f64>) -> u32 {
        let state_f32: Vec<f32> = state.iter().map(|&x| x as f32).collect();
        let patterns = self.sona.find_patterns(&state_f32, 1);
        if let Some(p) = patterns.first() {
            // Convert quality to a pseudo-action index
            ((p.avg_quality * 100.0) as u32) % 100
        } else {
            0
        }
    }

    /// Think - process input and generate response (with pre-computed embedding)
    pub fn think_with_embedding(
        &self,
        input: &str,
        embedding: &[f32],
    ) -> Result<ThinkResult, String> {
        // Predictive coding: check if this query is novel enough to process
        let is_novel = self.nervous.should_process(embedding);

        // Compute novelty score via dentate gyrus sparse encoding
        let novelty = self.nervous.novelty_score(embedding);

        // Recall from vector store
        let memories = self.recall_f32(embedding, Some(5), None)?;

        // Augment with Hopfield associative retrieval (pattern completion)
        if let Some((hopfield_pattern, sim)) = self.nervous.hopfield_retrieve(embedding) {
            // If Hopfield finds a strong match not in HNSW results, it suggests
            // an associative connection that pure cosine similarity might miss.
            if sim > 0.5 {
                // Record the associative retrieval as a SONA trajectory
                self.sona.record_query(embedding, sim.min(1.0));
            }
            let _ = hopfield_pattern; // Pattern used for SONA quality signal
        }

        // Update predictive model
        self.nervous.update_prediction(embedding);

        let thought = {
            let t = self.generate_thought(
                ThoughtType::Inference,
                format!(
                    "Processing{} (novelty={:.2}): {}",
                    if is_novel { " (novel)" } else { "" },
                    novelty,
                    &input[..input.len().min(80)]
                ),
                if is_novel { 0.8 } else { 0.6 },
            );
            // Patch the thought's novelty field with the dentate gyrus score
            let mut thoughts = self.thoughts.write();
            if let Some(last) = thoughts.last_mut() {
                last.novelty = novelty as f64;
            }
            t
        };

        let response = if memories.is_empty() {
            "No relevant information found in memory.".to_string()
        } else {
            format!(
                "Based on {} relevant memories: {}",
                memories.len(),
                memories
                    .first()
                    .map(|m| m.content.clone())
                    .unwrap_or_default()
            )
        };

        let confidence = memories.first().map(|m| m.similarity).unwrap_or(0.1);

        Ok(ThinkResult {
            response,
            confidence,
            thought_id: thought.id,
            memory_count: memories.len() as u32,
        })
    }

    /// Think - process input and generate response (legacy f64 interface)
    pub fn think(&self, input: String, input_vector: Vec<f64>) -> Result<ThinkResult, String> {
        let memories = self.recall(input_vector.clone(), Some(5), None)?;

        let thought = self.generate_thought(
            ThoughtType::Inference,
            format!("Processing: {}", &input[..input.len().min(100)]),
            0.7,
        );

        let response = if memories.is_empty() {
            "No relevant information found in memory.".to_string()
        } else {
            format!(
                "Based on {} relevant memories: {}",
                memories.len(),
                memories
                    .first()
                    .map(|m| m.content.clone())
                    .unwrap_or_default()
            )
        };

        let confidence = memories.first().map(|m| m.similarity).unwrap_or(0.1);

        Ok(ThinkResult {
            response,
            confidence,
            thought_id: thought.id,
            memory_count: memories.len() as u32,
        })
    }

    /// Add a goal
    pub fn add_goal(&self, description: String, priority: f64) -> String {
        let goal = Goal {
            id: generate_id(),
            description,
            priority,
            progress: 0.0,
            status: GoalStatus::Pending,
            created_at: now_millis(),
        };

        let id = goal.id.clone();
        self.goals.write().push(goal);
        id
    }

    /// Update goal progress
    pub fn update_goal(&self, goal_id: &str, progress: f64) -> bool {
        let mut goals = self.goals.write();
        if let Some(goal) = goals.iter_mut().find(|g| g.id == goal_id) {
            goal.progress = progress.min(1.0);
            if goal.progress >= 1.0 {
                goal.status = GoalStatus::Completed;
            } else if goal.progress > 0.0 {
                goal.status = GoalStatus::Active;
            }
            true
        } else {
            false
        }
    }

    /// Add a belief
    pub fn add_belief(&self, content: String, confidence: f64, source: String) -> String {
        let belief = Belief {
            id: generate_id(),
            content,
            confidence,
            source,
            timestamp: now_millis(),
        };

        let id = belief.id.clone();
        self.beliefs.write().push(belief);
        id
    }

    /// Generate a thought
    fn generate_thought(
        &self,
        thought_type: ThoughtType,
        content: String,
        confidence: f64,
    ) -> Thought {
        let thought = Thought {
            id: generate_id(),
            content,
            thought_type: format!("{:?}", thought_type),
            confidence,
            novelty: 0.5,
            utility: confidence,
            timestamp: now_millis(),
        };

        let mut thoughts = self.thoughts.write();
        thoughts.push(thought.clone());

        if thoughts.len() > 1000 {
            thoughts.drain(0..500);
        }

        thought
    }

    /// Self-improve - analyze and adapt
    pub fn evolve(&self) -> EvolutionResult {
        let mut adaptations = Vec::new();
        let mut improvements = Vec::new();

        let sona_stats = self.sona.stats();
        let nervous_stats = self.nervous.stats();

        // If too many trajectories are being dropped, force a learning cycle
        if sona_stats.trajectories_dropped > 0 {
            let msg = self.sona.force_learn();
            adaptations.push(format!("Forced SONA learning cycle: {}", msg));
        }

        let memory_stats = self.memory.stats();

        if memory_stats.avg_decay > 0.5 {
            let result = self.memory.consolidate();
            adaptations.push(format!(
                "Consolidated memory: pruned {} entries",
                result.pruned
            ));
        }

        // Report Hopfield capacity utilization
        if nervous_stats.hopfield_capacity > 0 {
            let utilization =
                nervous_stats.hopfield_patterns as f64 / nervous_stats.hopfield_capacity as f64;
            if utilization > 0.8 {
                adaptations.push(format!(
                    "Hopfield near capacity: {}/{} patterns ({:.0}%)",
                    nervous_stats.hopfield_patterns,
                    nervous_stats.hopfield_capacity,
                    utilization * 100.0
                ));
            }
        }

        // Report neural synchronization improvements
        if nervous_stats.router_sync > 0.7 {
            improvements.push(format!(
                "Neural sync: {:.2} — cognitive modules well-coordinated",
                nervous_stats.router_sync
            ));
        }

        let thought = self.generate_thought(
            ThoughtType::Reflection,
            format!(
                "Self-analysis: {} memories, {} SONA patterns, {} Hopfield patterns, sync={:.2}",
                memory_stats.total_memories,
                sona_stats.patterns_stored,
                nervous_stats.hopfield_patterns,
                nervous_stats.router_sync,
            ),
            0.8,
        );

        self.cycle_count.fetch_add(1, Ordering::Relaxed);

        EvolutionResult {
            adaptations,
            improvements,
            thought_id: thought.id,
        }
    }

    /// Introspect - get internal state
    pub fn introspect(&self) -> IntrospectionResult {
        let memory_stats = self.memory.stats();
        let sona_stats = self.sona.stats();
        let nervous_stats = self.nervous.stats();
        let thoughts = self.thoughts.read();
        let goals = self.goals.read();

        let active_goals = goals
            .iter()
            .filter(|g| g.status == GoalStatus::Active || g.status == GoalStatus::Pending)
            .count() as u32;

        // Derive trend from SONA pattern growth + neural sync
        let trend = if sona_stats.patterns_stored > 10 && nervous_stats.router_sync > 0.5 {
            "improving"
        } else if sona_stats.trajectories_dropped > 0 {
            "declining"
        } else {
            "stable"
        };

        IntrospectionResult {
            status: "healthy".to_string(),
            uptime_ms: now_millis() - self.start_time,
            total_memories: memory_stats.total_memories,
            total_thoughts: thoughts.len() as u32,
            total_experiences: sona_stats.trajectories_buffered as u32,
            active_goals,
            avg_reward: nervous_stats.router_sync as f64, // Use neural sync as health signal
            learning_trend: trend.to_string(),
            exploration_rate: 0.0, // No epsilon-greedy in SONA
        }
    }

    /// Get statistics
    pub fn stats(&self) -> CognitiveStats {
        let memory_stats = self.memory.stats();
        let sona_stats = self.sona.stats();

        CognitiveStats {
            total_memories: memory_stats.total_memories,
            total_thoughts: self.thoughts.read().len() as u32,
            total_experiences: sona_stats.trajectories_buffered as u32,
            avg_importance: memory_stats.avg_importance,
            avg_reward: 0.0, // SONA doesn't track global reward
            learning_trend: if sona_stats.patterns_stored > 0 { 1.0 } else { 0.0 },
        }
    }

    /// Get recent thoughts
    pub fn get_thoughts(&self, limit: Option<u32>) -> Vec<Thought> {
        let thoughts = self.thoughts.read();
        let n = limit.unwrap_or(10) as usize;
        thoughts.iter().rev().take(n).cloned().collect()
    }

    /// Run a cognitive cycle (ticks SONA engine + nervous system + periodic memory consolidation)
    pub fn cycle(&self) -> CycleResult {
        self.cycle_count.fetch_add(1, Ordering::Relaxed);

        // Step the oscillatory router (dt = 60s default cycle interval)
        self.nervous.router_step(60.0);

        // Tick SONA — runs background learning if the interval has elapsed
        let sona_msg = self.sona.tick();
        let mut insights = Vec::new();
        if let Some(msg) = sona_msg {
            insights.push(msg);
        }

        // Report nervous system sync level
        let sync = self.nervous.router_sync();
        if sync > 0.8 {
            insights.push(format!("Neural sync high: {:.2} — modules coherent", sync));
        }

        // Flush instant-loop MicroLoRA gradients
        self.sona.flush();

        let consolidated = if self.cycle_count.load(Ordering::Relaxed) % 100 == 0 {
            Some(self.memory.consolidate())
        } else {
            None
        };

        CycleResult {
            cycle_number: self.cycle_count.load(Ordering::Relaxed),
            training_insights: insights,
            memories_pruned: consolidated.map(|c| c.pruned).unwrap_or(0),
        }
    }

    /// Set running state
    pub fn set_running(&self, running: bool) {
        self.running.store(running, Ordering::Relaxed);
    }

    /// Check if running
    pub fn is_running(&self) -> bool {
        self.running.load(Ordering::Relaxed)
    }
}

/// Result of memory recall
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RecallResult {
    pub id: String,
    pub content: String,
    pub similarity: f64,
    pub memory_type: String,
}

/// Result of learning
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LearnResult {
    pub success: bool,
    pub reward: f64,
    pub td_error: f64,
    pub thought_id: String,
    pub insights: Vec<String>,
}

/// Result of thinking
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThinkResult {
    pub response: String,
    pub confidence: f64,
    pub thought_id: String,
    pub memory_count: u32,
}

/// Result of evolution/self-improvement
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvolutionResult {
    pub adaptations: Vec<String>,
    pub improvements: Vec<String>,
    pub thought_id: String,
}

/// Result of introspection
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IntrospectionResult {
    pub status: String,
    pub uptime_ms: i64,
    pub total_memories: u32,
    pub total_thoughts: u32,
    pub total_experiences: u32,
    pub active_goals: u32,
    pub avg_reward: f64,
    pub learning_trend: String,
    pub exploration_rate: f64,
}

/// Result of a cognitive cycle
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CycleResult {
    pub cycle_number: u64,
    pub training_insights: Vec<String>,
    pub memories_pruned: u32,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cognitive_engine() {
        let engine = CognitiveEngine::new(None);

        let id = engine
            .remember(
                "Test memory".to_string(),
                vec![0.1; 384],
                "semantic".to_string(),
                Some(0.8),
            )
            .unwrap();
        assert!(!id.is_empty());

        let goal_id = engine.add_goal("Test goal".to_string(), 0.9);
        assert!(!goal_id.is_empty());

        let state = engine.introspect();
        assert_eq!(state.status, "healthy");
        assert!(state.total_memories >= 1);
    }
}
