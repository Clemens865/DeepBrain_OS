//! High-performance learning algorithms for DeepBrain (Tauri port)

use std::sync::atomic::{AtomicU64, Ordering};

use dashmap::DashMap;
use parking_lot::RwLock;
use rand::prelude::*;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

use crate::brain::types::{Experience, LearningOutcome, LearningType};
use crate::brain::utils::{generate_id, now_millis, softmax};

/// Experience buffer entry
#[derive(Debug, Clone)]
struct ExperienceEntry {
    state: Vec<f64>,
    action: u32,
    reward: f64,
    next_state: Vec<f64>,
    done: bool,
    priority: f64,
    #[allow(dead_code)]
    timestamp: i64,
}

/// Q-table entry
#[derive(Debug, Clone, Default)]
struct QEntry {
    values: Vec<f64>,
    visits: u32,
}

/// Strategy performance tracking
#[derive(Debug, Clone)]
struct Strategy {
    id: String,
    name: String,
    learning_type: LearningType,
    success_rate: f64,
    usage_count: u64,
    avg_reward: f64,
    parameters: Vec<f64>,
}

/// High-performance native learning system
pub struct NativeLearner {
    /// Experience replay buffer
    experience_buffer: RwLock<Vec<ExperienceEntry>>,
    /// Q-table for value-based methods
    q_table: DashMap<u64, QEntry, ahash::RandomState>,
    /// Learning strategies
    strategies: RwLock<Vec<Strategy>>,
    /// Configuration
    config: RwLock<LearnerConfig>,
    /// Performance tracking
    recent_rewards: RwLock<Vec<f64>>,
    /// Statistics
    total_experiences: AtomicU64,
    total_updates: AtomicU64,
    #[allow(dead_code)]
    state_dimension: usize,
    action_count: usize,
}

#[derive(Debug, Clone)]
pub(crate) struct LearnerConfig {
    pub learning_rate: f64,
    pub discount_factor: f64,
    pub exploration_rate: f64,
    pub batch_size: usize,
    pub buffer_size: usize,
    #[allow(dead_code)]
    pub target_update_freq: u32,
    pub curiosity_weight: f64,
}

impl Default for LearnerConfig {
    fn default() -> Self {
        Self {
            learning_rate: 0.01,
            discount_factor: 0.99,
            exploration_rate: 0.1,
            batch_size: 32,
            buffer_size: 10_000,
            target_update_freq: 100,
            curiosity_weight: 0.5,
        }
    }
}

impl NativeLearner {
    /// Create a new native learner
    pub fn new(state_dim: u32, action_count: u32) -> Self {
        let mut learner = Self {
            experience_buffer: RwLock::new(Vec::with_capacity(10_000)),
            q_table: DashMap::with_hasher(ahash::RandomState::new()),
            strategies: RwLock::new(Vec::new()),
            config: RwLock::new(LearnerConfig::default()),
            recent_rewards: RwLock::new(Vec::with_capacity(100)),
            total_experiences: AtomicU64::new(0),
            total_updates: AtomicU64::new(0),
            state_dimension: state_dim as usize,
            action_count: action_count as usize,
        };

        learner.initialize_strategies();
        learner
    }

    fn initialize_strategies(&mut self) {
        let strategies = vec![
            Strategy {
                id: generate_id(),
                name: "Q-Learning".to_string(),
                learning_type: LearningType::QLearning,
                success_rate: 0.5,
                usage_count: 0,
                avg_reward: 0.0,
                parameters: vec![0.1, 0.99, 0.1],
            },
            Strategy {
                id: generate_id(),
                name: "SARSA".to_string(),
                learning_type: LearningType::SARSA,
                success_rate: 0.5,
                usage_count: 0,
                avg_reward: 0.0,
                parameters: vec![0.1, 0.99, 0.1],
            },
            Strategy {
                id: generate_id(),
                name: "Curiosity-Driven".to_string(),
                learning_type: LearningType::CuriosityDriven,
                success_rate: 0.5,
                usage_count: 0,
                avg_reward: 0.0,
                parameters: vec![0.5, 0.3],
            },
        ];

        *self.strategies.write() = strategies;
    }

    /// Learn from a new experience
    pub fn learn(&self, experience: Experience) -> Result<LearningOutcome, String> {
        let state: Vec<f64> = experience.state.clone();
        let next_state: Vec<f64> = experience.next_state.clone();

        let curiosity_bonus = self.calculate_curiosity(&state);
        let config = self.config.read();
        let total_reward = experience.reward + curiosity_bonus * config.curiosity_weight;

        {
            let mut buffer = self.experience_buffer.write();
            buffer.push(ExperienceEntry {
                state,
                action: experience.action,
                reward: total_reward,
                next_state,
                done: experience.done,
                priority: total_reward.abs(),
                timestamp: now_millis(),
            });

            if buffer.len() > config.buffer_size {
                buffer.remove(0);
            }
        }

        {
            let mut rewards = self.recent_rewards.write();
            rewards.push(total_reward);
            if rewards.len() > 100 {
                rewards.remove(0);
            }
        }

        self.total_experiences.fetch_add(1, Ordering::Relaxed);

        let insights = if self.experience_buffer.read().len() >= config.batch_size {
            self.train_batch()?
        } else {
            Vec::new()
        };

        let td_error = self.compute_td_error(&experience);
        let meta_insights = self.meta_learn(total_reward);

        let mut all_insights = insights;
        all_insights.extend(meta_insights);

        Ok(LearningOutcome {
            success: total_reward > 0.0,
            reward: total_reward,
            td_error,
            insights: all_insights,
        })
    }

    /// Train on a batch of experiences (parallel)
    pub fn train_batch(&self) -> Result<Vec<String>, String> {
        let config = self.config.read();
        let buffer = self.experience_buffer.read();

        if buffer.len() < config.batch_size {
            return Ok(Vec::new());
        }

        let batch = self.sample_prioritized_batch(&buffer, config.batch_size);

        let td_errors: Vec<f64> = batch
            .par_iter()
            .map(|exp| {
                let state_hash = self.hash_state(&exp.state);

                let mut q_entry = self.q_table.entry(state_hash).or_insert_with(|| QEntry {
                    values: vec![0.0; self.action_count],
                    visits: 0,
                });

                let next_state_hash = self.hash_state(&exp.next_state);
                let next_max_q = if exp.done {
                    0.0
                } else {
                    self.q_table
                        .get(&next_state_hash)
                        .map(|e| {
                            e.values
                                .iter()
                                .cloned()
                                .fold(f64::NEG_INFINITY, f64::max)
                        })
                        .unwrap_or(0.0)
                };

                let td_target = exp.reward + config.discount_factor * next_max_q;
                let current_q = q_entry.values[exp.action as usize];
                let td_error = td_target - current_q;

                q_entry.values[exp.action as usize] += config.learning_rate * td_error;
                q_entry.visits += 1;

                td_error
            })
            .collect();

        self.total_updates
            .fetch_add(batch.len() as u64, Ordering::Relaxed);

        let avg_td_error: f64 = td_errors.iter().sum::<f64>() / td_errors.len() as f64;
        let mut insights = Vec::new();

        if avg_td_error.abs() < 0.01 {
            insights.push("Learning converging - TD error near zero".to_string());
        } else if avg_td_error.abs() > 1.0 {
            insights.push("High TD error - significant learning occurring".to_string());
        }

        Ok(insights)
    }

    /// Select action using epsilon-greedy policy
    pub fn select_action(&self, state: Vec<f64>) -> u32 {
        let config = self.config.read();
        let mut rng = thread_rng();

        if rng.gen::<f64>() < config.exploration_rate {
            return rng.gen_range(0..self.action_count as u32);
        }

        let state_hash = self.hash_state(&state);

        self.q_table
            .get(&state_hash)
            .map(|entry| {
                entry
                    .values
                    .iter()
                    .enumerate()
                    .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
                    .map(|(i, _)| i as u32)
                    .unwrap_or(0)
            })
            .unwrap_or_else(|| rng.gen_range(0..self.action_count as u32))
    }

    /// Select action using softmax policy
    pub fn select_action_softmax(&self, state: Vec<f64>, temperature: f64) -> u32 {
        let state_hash = self.hash_state(&state);
        let mut rng = thread_rng();

        let q_values = self
            .q_table
            .get(&state_hash)
            .map(|e| e.values.clone())
            .unwrap_or_else(|| vec![0.0; self.action_count]);

        let mut probs: Vec<f64> = q_values.iter().map(|&q| q / temperature).collect();
        softmax(&mut probs);

        let r: f64 = rng.gen();
        let mut cumsum = 0.0;
        for (i, &p) in probs.iter().enumerate() {
            cumsum += p;
            if r <= cumsum {
                return i as u32;
            }
        }

        (self.action_count - 1) as u32
    }

    fn calculate_curiosity(&self, state: &[f64]) -> f64 {
        let state_hash = self.hash_state(state);
        let visits = self
            .q_table
            .get(&state_hash)
            .map(|e| e.visits)
            .unwrap_or(0);

        if visits == 0 {
            1.0
        } else {
            1.0 / (1.0 + (visits as f64).sqrt())
        }
    }

    fn meta_learn(&self, _reward: f64) -> Vec<String> {
        let mut insights = Vec::new();
        let rewards = self.recent_rewards.read();

        if rewards.len() < 50 {
            return insights;
        }

        let trend = self.calculate_trend(&rewards);
        let mut config = self.config.write();

        if trend < -0.05 {
            let old_rate = config.exploration_rate;
            config.exploration_rate = (config.exploration_rate * 1.1).min(0.3);
            if (config.exploration_rate - old_rate).abs() > 0.001 {
                insights.push(format!(
                    "Increased exploration rate: {:.2} -> {:.2}",
                    old_rate, config.exploration_rate
                ));
            }
        } else if trend > 0.05 && config.exploration_rate > 0.05 {
            let old_rate = config.exploration_rate;
            config.exploration_rate = (config.exploration_rate * 0.95).max(0.05);
            if (config.exploration_rate - old_rate).abs() > 0.001 {
                insights.push(format!(
                    "Decreased exploration rate: {:.2} -> {:.2}",
                    old_rate, config.exploration_rate
                ));
            }
        }

        insights
    }

    fn compute_td_error(&self, exp: &Experience) -> f64 {
        let config = self.config.read();
        let state = exp.state.clone();
        let next_state = exp.next_state.clone();

        let state_hash = self.hash_state(&state);
        let next_hash = self.hash_state(&next_state);

        let current_q = self
            .q_table
            .get(&state_hash)
            .map(|e| e.values.get(exp.action as usize).copied().unwrap_or(0.0))
            .unwrap_or(0.0);

        let next_max_q = if exp.done {
            0.0
        } else {
            self.q_table
                .get(&next_hash)
                .map(|e| {
                    e.values
                        .iter()
                        .cloned()
                        .fold(f64::NEG_INFINITY, f64::max)
                })
                .unwrap_or(0.0)
        };

        let td_target = exp.reward + config.discount_factor * next_max_q;
        td_target - current_q
    }

    fn sample_prioritized_batch(
        &self,
        buffer: &[ExperienceEntry],
        size: usize,
    ) -> Vec<ExperienceEntry> {
        let mut rng = thread_rng();
        let total_priority: f64 = buffer.iter().map(|e| e.priority.abs() + 0.01).sum();

        let mut batch = Vec::with_capacity(size);
        let mut selected = std::collections::HashSet::new();

        while batch.len() < size && selected.len() < buffer.len() {
            let r: f64 = rng.gen::<f64>() * total_priority;
            let mut cumsum = 0.0;

            for (i, exp) in buffer.iter().enumerate() {
                cumsum += exp.priority.abs() + 0.01;
                if r <= cumsum && !selected.contains(&i) {
                    batch.push(exp.clone());
                    selected.insert(i);
                    break;
                }
            }
        }

        batch
    }

    fn calculate_trend(&self, values: &[f64]) -> f64 {
        if values.len() < 2 {
            return 0.0;
        }

        let n = values.len() as f64;
        let mut sum_x = 0.0;
        let mut sum_y = 0.0;
        let mut sum_xy = 0.0;
        let mut sum_xx = 0.0;

        for (i, &y) in values.iter().enumerate() {
            let x = i as f64;
            sum_x += x;
            sum_y += y;
            sum_xy += x * y;
            sum_xx += x * x;
        }

        let denom = n * sum_xx - sum_x * sum_x;
        if denom.abs() < 1e-10 {
            return 0.0;
        }

        (n * sum_xy - sum_x * sum_y) / denom
    }

    fn hash_state(&self, state: &[f64]) -> u64 {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let mut hasher = DefaultHasher::new();
        for &x in state {
            let quantized = (x * 100.0).round() as i32;
            quantized.hash(&mut hasher);
        }
        hasher.finish()
    }

    /// Get statistics
    pub fn stats(&self) -> LearnerStats {
        let rewards = self.recent_rewards.read();
        let avg_reward = if rewards.is_empty() {
            0.0
        } else {
            rewards.iter().sum::<f64>() / rewards.len() as f64
        };

        let config = self.config.read();

        LearnerStats {
            total_experiences: self.total_experiences.load(Ordering::Relaxed) as f64,
            total_updates: self.total_updates.load(Ordering::Relaxed) as f64,
            q_table_size: self.q_table.len() as u32,
            avg_reward,
            exploration_rate: config.exploration_rate,
            learning_rate: config.learning_rate,
            trend: self.calculate_trend(&rewards),
        }
    }

    pub fn set_learning_rate(&self, rate: f64) {
        self.config.write().learning_rate = rate;
    }

    pub fn set_exploration_rate(&self, rate: f64) {
        self.config.write().exploration_rate = rate;
    }

    pub fn explore(&self) {
        self.config.write().exploration_rate = 0.5;
    }

    pub fn exploit(&self) {
        self.config.write().exploration_rate = 0.1;
    }

    /// Export Q-table for persistence
    pub fn export_q_table(&self) -> Vec<(u64, Vec<f64>, u32)> {
        self.q_table
            .iter()
            .map(|e| (*e.key(), e.value().values.clone(), e.value().visits))
            .collect()
    }

    /// Import Q-table from persistence
    pub fn import_q_table(&self, entries: Vec<(u64, Vec<f64>, u32)>) {
        for (key, values, visits) in entries {
            self.q_table.insert(key, QEntry { values, visits });
        }
    }

    /// Export experience buffer for persistence
    pub fn export_experiences(&self) -> Vec<Experience> {
        self.experience_buffer
            .read()
            .iter()
            .map(|e| Experience {
                state: e.state.clone(),
                action: e.action,
                reward: e.reward,
                next_state: e.next_state.clone(),
                done: e.done,
            })
            .collect()
    }
}

/// Learner statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LearnerStats {
    pub total_experiences: f64,
    pub total_updates: f64,
    pub q_table_size: u32,
    pub avg_reward: f64,
    pub exploration_rate: f64,
    pub learning_rate: f64,
    pub trend: f64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_learn_and_select() {
        let learner = NativeLearner::new(4, 3);

        let exp = Experience {
            state: vec![1.0, 0.0, 0.0, 0.0],
            action: 1,
            reward: 1.0,
            next_state: vec![0.0, 1.0, 0.0, 0.0],
            done: false,
        };

        let outcome = learner.learn(exp).unwrap();
        assert!(outcome.success);

        let action = learner.select_action(vec![1.0, 0.0, 0.0, 0.0]);
        assert!(action < 3);
    }
}
