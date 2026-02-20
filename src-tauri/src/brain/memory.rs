//! High-performance memory system for DeepBrain (Tauri port)
//!
//! Features:
//! - Lock-free concurrent access via DashMap
//! - SIMD-accelerated similarity search
//! - Automatic memory consolidation
//! - Importance-based retention

use std::sync::atomic::{AtomicU64, Ordering};

use dashmap::DashMap;
use parking_lot::RwLock;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use smallvec::SmallVec;

use std::sync::Arc;

use crate::brain::types::{DistanceMetric, MemoryEntry, MemoryType, parse_memory_type};
use crate::brain::utils::{
    cosine_similarity, dot_product, euclidean_distance, generate_id, normalize_vector, now_millis,
};
use crate::deepbrain::vector_store::{DeepBrainVectorStore, VectorFilter, VectorMetadata};

/// Internal memory storage with vector
#[derive(Debug, Clone)]
pub struct MemoryNode {
    pub id: String,
    pub content: String,
    pub vector: Vec<f32>,
    pub memory_type: MemoryType,
    pub importance: f64,
    pub decay: f64,
    pub access_count: u32,
    pub timestamp: i64,
    pub connections: SmallVec<[String; 8]>,
}

/// High-performance native memory system
pub struct NativeMemory {
    /// Memory storage using lock-free DashMap
    memories: DashMap<String, MemoryNode, ahash::RandomState>,
    /// Type indices for fast filtering
    type_indices: DashMap<String, Vec<String>, ahash::RandomState>,
    /// Vector dimension
    dimensions: usize,
    /// RVF-backed vector store for crash-safe, indexed search.
    /// None = brute-force fallback (tests or when store not yet initialized).
    vector_store: Option<Arc<DeepBrainVectorStore>>,
    /// Configuration
    config: RwLock<MemoryConfig>,
    /// Statistics
    total_accesses: AtomicU64,
    total_stores: AtomicU64,
}

#[derive(Debug, Clone)]
struct MemoryConfig {
    max_memories: usize,
    decay_rate: f64,
    consolidation_threshold: f64,
    importance_threshold: f64,
    metric: DistanceMetric,
}

impl Default for MemoryConfig {
    fn default() -> Self {
        Self {
            max_memories: 100_000,
            decay_rate: 0.01,
            consolidation_threshold: 0.85,
            importance_threshold: 0.3,
            metric: DistanceMetric::Cosine,
        }
    }
}

impl NativeMemory {
    /// Create a new native memory system (brute-force fallback, no RVF store).
    pub fn new(dimensions: u32) -> Self {
        Self {
            memories: DashMap::with_hasher(ahash::RandomState::new()),
            type_indices: DashMap::with_hasher(ahash::RandomState::new()),
            dimensions: dimensions as usize,
            vector_store: None,
            config: RwLock::new(MemoryConfig::default()),
            total_accesses: AtomicU64::new(0),
            total_stores: AtomicU64::new(0),
        }
    }

    /// Create a new native memory system backed by a DeepBrain RVF vector store.
    pub fn with_vector_store(dimensions: u32, store: Arc<DeepBrainVectorStore>) -> Self {
        Self {
            memories: DashMap::with_hasher(ahash::RandomState::new()),
            type_indices: DashMap::with_hasher(ahash::RandomState::new()),
            dimensions: dimensions as usize,
            vector_store: Some(store),
            config: RwLock::new(MemoryConfig::default()),
            total_accesses: AtomicU64::new(0),
            total_stores: AtomicU64::new(0),
        }
    }

    /// Store a memory with vector embedding
    pub fn store(
        &self,
        content: String,
        vector: Vec<f64>,
        memory_type: String,
        importance: f64,
    ) -> Result<String, String> {
        let mut vec_f32: Vec<f32> = vector.iter().map(|&x| x as f32).collect();

        if vec_f32.len() != self.dimensions {
            return Err(format!(
                "Vector dimension mismatch: expected {}, got {}",
                self.dimensions,
                vec_f32.len()
            ));
        }

        normalize_vector(&mut vec_f32);

        let id = generate_id();
        let mem_type = parse_memory_type(&memory_type);

        let node = MemoryNode {
            id: id.clone(),
            content,
            vector: vec_f32,
            memory_type: mem_type,
            importance,
            decay: 0.0,
            access_count: 0,
            timestamp: now_millis(),
            connections: SmallVec::new(),
        };

        self.memories.insert(id.clone(), node);

        self.type_indices
            .entry(memory_type)
            .or_insert_with(Vec::new)
            .push(id.clone());

        self.total_stores.fetch_add(1, Ordering::Relaxed);
        self.enforce_limits();

        Ok(id)
    }

    /// Store a memory with a pre-computed f32 vector (no conversion needed)
    pub fn store_f32(
        &self,
        content: String,
        mut vector: Vec<f32>,
        memory_type: String,
        importance: f64,
    ) -> Result<String, String> {
        if vector.len() != self.dimensions {
            return Err(format!(
                "Vector dimension mismatch: expected {}, got {}",
                self.dimensions,
                vector.len()
            ));
        }

        normalize_vector(&mut vector);

        let id = generate_id();
        let mem_type = parse_memory_type(&memory_type);

        // Insert into RVF vector store if available.
        if let Some(ref store) = self.vector_store {
            let meta = VectorMetadata {
                source: "memory".to_string(),
                memory_type: memory_type.clone(),
                importance,
                timestamp: now_millis(),
                ..Default::default()
            };
            if let Err(e) = store.store_vector(&id, &vector, meta) {
                tracing::warn!("Failed to store vector in RVF: {}", e);
            }
        }

        let node = MemoryNode {
            id: id.clone(),
            content,
            vector,
            memory_type: mem_type,
            importance,
            decay: 0.0,
            access_count: 0,
            timestamp: now_millis(),
            connections: SmallVec::new(),
        };

        self.memories.insert(id.clone(), node);

        self.type_indices
            .entry(memory_type)
            .or_insert_with(Vec::new)
            .push(id.clone());

        self.total_stores.fetch_add(1, Ordering::Relaxed);
        self.enforce_limits();

        Ok(id)
    }

    /// Store multiple memories in batch (parallel)
    pub fn store_batch(&self, entries: Vec<BatchEntry>) -> Result<Vec<String>, String> {
        let ids: Vec<String> = entries
            .into_par_iter()
            .filter_map(|entry| {
                let mut vec_f32: Vec<f32> = entry.vector.iter().map(|&x| x as f32).collect();
                if vec_f32.len() != self.dimensions {
                    return None;
                }
                normalize_vector(&mut vec_f32);

                let id = generate_id();
                let mem_type = parse_memory_type(&entry.memory_type);

                let node = MemoryNode {
                    id: id.clone(),
                    content: entry.content,
                    vector: vec_f32,
                    memory_type: mem_type,
                    importance: entry.importance,
                    decay: 0.0,
                    access_count: 0,
                    timestamp: now_millis(),
                    connections: SmallVec::new(),
                };

                self.memories.insert(id.clone(), node);
                Some(id)
            })
            .collect();

        self.total_stores
            .fetch_add(ids.len() as u64, Ordering::Relaxed);
        Ok(ids)
    }

    /// Search for similar memories
    pub fn search(
        &self,
        query_vector: Vec<f64>,
        k: u32,
        memory_types: Option<Vec<String>>,
        min_similarity: Option<f64>,
    ) -> Result<Vec<SearchResult>, String> {
        let query: Vec<f32> = query_vector.iter().map(|&x| x as f32).collect();

        if query.len() != self.dimensions {
            return Err("Query dimension mismatch".to_string());
        }

        let min_sim = min_similarity.unwrap_or(0.0) as f32;
        let type_filter: Option<Vec<MemoryType>> = memory_types
            .map(|types| types.iter().map(|t| parse_memory_type(t)).collect());

        let config = self.config.read();

        let mut results: Vec<(String, f32, MemoryNode)> = self
            .memories
            .iter()
            .filter_map(|entry| {
                let node = entry.value();

                if let Some(ref types) = type_filter {
                    if !types.contains(&node.memory_type) {
                        return None;
                    }
                }

                let similarity = match config.metric {
                    DistanceMetric::Cosine => cosine_similarity(&query, &node.vector),
                    DistanceMetric::Euclidean => {
                        1.0 / (1.0 + euclidean_distance(&query, &node.vector))
                    }
                    DistanceMetric::DotProduct => dot_product(&query, &node.vector),
                    DistanceMetric::Manhattan => {
                        let dist: f32 = query
                            .iter()
                            .zip(node.vector.iter())
                            .map(|(a, b)| (a - b).abs())
                            .sum();
                        1.0 / (1.0 + dist)
                    }
                };

                let adjusted_sim = similarity * (1.0 - node.decay as f32);

                if adjusted_sim >= min_sim {
                    Some((node.id.clone(), adjusted_sim, node.clone()))
                } else {
                    None
                }
            })
            .collect();

        results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        let top_k: Vec<SearchResult> = results
            .into_iter()
            .take(k as usize)
            .map(|(id, similarity, node)| {
                if let Some(mut entry) = self.memories.get_mut(&id) {
                    entry.access_count += 1;
                }
                self.total_accesses.fetch_add(1, Ordering::Relaxed);

                SearchResult {
                    id,
                    content: node.content,
                    similarity: similarity as f64,
                    memory_type: format!("{:?}", node.memory_type),
                    importance: node.importance,
                }
            })
            .collect();

        Ok(top_k)
    }

    /// Search with f32 query (no conversion needed).
    /// Uses HNSW index for O(log n) search when available, falls back to brute force.
    pub fn search_f32(
        &self,
        query: &[f32],
        k: u32,
        memory_types: Option<Vec<String>>,
        min_similarity: Option<f64>,
    ) -> Result<Vec<SearchResult>, String> {
        if query.len() != self.dimensions {
            return Err("Query dimension mismatch".to_string());
        }

        let min_sim = min_similarity.unwrap_or(0.0) as f32;
        let type_filter: Option<Vec<MemoryType>> = memory_types
            .map(|types| types.iter().map(|t| parse_memory_type(t)).collect());

        // Try RVF vector store first (fast O(log n) HNSW)
        if let Some(ref store) = self.vector_store {
            if self.memories.len() > 10 {
                return self.search_f32_rvf(store, query, k, type_filter.as_ref(), min_sim);
            }
        }

        // Fallback: brute-force scan (used for small memory counts or before HNSW is built)
        let config = self.config.read();

        let mut results: Vec<(String, f32, MemoryNode)> = self
            .memories
            .iter()
            .filter_map(|entry| {
                let node = entry.value();

                if let Some(ref types) = type_filter {
                    if !types.contains(&node.memory_type) {
                        return None;
                    }
                }

                let similarity = match config.metric {
                    DistanceMetric::Cosine => cosine_similarity(query, &node.vector),
                    DistanceMetric::Euclidean => {
                        1.0 / (1.0 + euclidean_distance(query, &node.vector))
                    }
                    DistanceMetric::DotProduct => dot_product(query, &node.vector),
                    DistanceMetric::Manhattan => {
                        let dist: f32 = query
                            .iter()
                            .zip(node.vector.iter())
                            .map(|(a, b)| (a - b).abs())
                            .sum();
                        1.0 / (1.0 + dist)
                    }
                };

                let adjusted_sim = similarity * (1.0 - node.decay as f32);

                if adjusted_sim >= min_sim {
                    Some((node.id.clone(), adjusted_sim, node.clone()))
                } else {
                    None
                }
            })
            .collect();

        results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        let top_k: Vec<SearchResult> = results
            .into_iter()
            .take(k as usize)
            .map(|(id, similarity, node)| {
                if let Some(mut entry) = self.memories.get_mut(&id) {
                    entry.access_count += 1;
                }
                self.total_accesses.fetch_add(1, Ordering::Relaxed);

                SearchResult {
                    id,
                    content: node.content,
                    similarity: similarity as f64,
                    memory_type: format!("{:?}", node.memory_type),
                    importance: node.importance,
                }
            })
            .collect();

        Ok(top_k)
    }

    /// RVF vector store accelerated search with post-filtering for decay/type.
    fn search_f32_rvf(
        &self,
        store: &DeepBrainVectorStore,
        query: &[f32],
        k: u32,
        type_filter: Option<&Vec<MemoryType>>,
        min_sim: f32,
    ) -> Result<Vec<SearchResult>, String> {
        // Build filter for source=memory. Type filtering is done post-search
        // against the DashMap since RVF metadata filtering is coarser.
        let filter = Some(VectorFilter::Source("memory".to_string()));

        // Over-fetch to account for type filtering and decay adjustments.
        let fetch_k = if type_filter.is_some() {
            (k as usize) * 4
        } else {
            (k as usize) * 2
        };

        let rvf_results = store
            .search(query, fetch_k, filter)
            .map_err(|e| format!("RVF search failed: {}", e))?;

        let mut results: Vec<SearchResult> = rvf_results
            .into_iter()
            .filter_map(|vr| {
                let node = self.memories.get(&vr.id)?;
                let node = node.value();

                // Type filter
                if let Some(types) = type_filter {
                    if !types.contains(&node.memory_type) {
                        return None;
                    }
                }

                // Apply decay adjustment to RVF similarity score.
                let adjusted_sim = vr.similarity as f32 * (1.0 - node.decay as f32);

                if adjusted_sim < min_sim {
                    return None;
                }

                Some(SearchResult {
                    id: vr.id,
                    content: node.content.clone(),
                    similarity: adjusted_sim as f64,
                    memory_type: format!("{:?}", node.memory_type),
                    importance: node.importance,
                })
            })
            .collect();

        results.sort_by(|a, b| {
            b.similarity
                .partial_cmp(&a.similarity)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        results.truncate(k as usize);

        // Update access counts.
        for r in &results {
            if let Some(mut entry) = self.memories.get_mut(&r.id) {
                entry.access_count += 1;
            }
            self.total_accesses.fetch_add(1, Ordering::Relaxed);
        }

        Ok(results)
    }

    /// Connect two memories
    pub fn connect(&self, id1: &str, id2: &str) -> bool {
        if let Some(mut node1) = self.memories.get_mut(id1) {
            if !node1.connections.contains(&id2.to_string()) {
                node1.connections.push(id2.to_string());
            }
        } else {
            return false;
        }

        if let Some(mut node2) = self.memories.get_mut(id2) {
            if !node2.connections.contains(&id1.to_string()) {
                node2.connections.push(id1.to_string());
            }
        }

        true
    }

    /// Consolidate memories - merge similar, prune weak
    pub fn consolidate(&self) -> ConsolidationResult {
        let config = self.config.read();
        let merged = 0u32;
        let mut pruned = 0u32;

        let to_prune: Vec<String> = self
            .memories
            .iter()
            .filter_map(|entry| {
                let node = entry.value();
                if node.decay > 0.9 && node.importance < config.importance_threshold {
                    Some(node.id.clone())
                } else {
                    None
                }
            })
            .collect();

        for id in to_prune {
            // Delete from RVF vector store.
            if let Some(ref store) = self.vector_store {
                if let Err(e) = store.delete(&id) {
                    tracing::warn!("Failed to delete vector from RVF during consolidation: {}", e);
                }
            }
            self.memories.remove(&id);
            pruned += 1;
        }

        self.memories.iter_mut().for_each(|mut entry| {
            entry.value_mut().decay += config.decay_rate;
        });

        ConsolidationResult {
            merged,
            pruned,
            total_remaining: self.memories.len() as u32,
        }
    }

    /// Update a memory's content, type, and/or importance
    pub fn update(&self, id: &str, content: Option<String>, memory_type: Option<String>, importance: Option<f64>) -> bool {
        if let Some(mut entry) = self.memories.get_mut(id) {
            if let Some(c) = content {
                entry.content = c;
            }
            if let Some(mt) = memory_type {
                entry.memory_type = parse_memory_type(&mt);
            }
            if let Some(imp) = importance {
                entry.importance = imp;
            }
            true
        } else {
            false
        }
    }

    /// Delete a memory
    pub fn delete(&self, id: &str) -> bool {
        if let Some(ref store) = self.vector_store {
            if let Err(e) = store.delete(id) {
                tracing::warn!("Failed to delete vector from RVF: {}", e);
            }
        }
        self.memories.remove(id).is_some()
    }

    /// Get memory count
    pub fn len(&self) -> u32 {
        self.memories.len() as u32
    }

    /// Check if memory is empty
    pub fn is_empty(&self) -> bool {
        self.memories.is_empty()
    }

    /// Get a specific memory
    pub fn get(&self, id: &str) -> Option<MemoryEntry> {
        self.memories.get(id).map(|node| MemoryEntry {
            id: node.id.clone(),
            content: node.content.clone(),
            memory_type: format!("{:?}", node.memory_type),
            importance: node.importance,
            decay: node.decay,
            access_count: node.access_count,
            timestamp: node.timestamp,
            connections: node.connections.to_vec(),
        })
    }

    /// Get all memory nodes (for persistence)
    pub fn all_nodes(&self) -> Vec<MemoryNode> {
        self.memories.iter().map(|e| e.value().clone()).collect()
    }

    /// Restore a memory node (for persistence loading).
    ///
    /// When a vector_store is present, this inserts the vector into RVF.
    /// The DashMap is always populated for fast metadata lookups.
    pub fn restore_node(&self, node: MemoryNode) {
        let type_str = format!("{:?}", node.memory_type);
        let id = node.id.clone();

        // Insert into RVF vector store if available.
        if let Some(ref store) = self.vector_store {
            let meta = VectorMetadata {
                source: "memory".to_string(),
                memory_type: type_str.clone(),
                importance: node.importance,
                timestamp: node.timestamp,
                ..Default::default()
            };
            if let Err(e) = store.store_vector(&id, &node.vector, meta) {
                tracing::warn!("Failed to restore vector into RVF: {}", e);
            }
        }

        self.memories.insert(id.clone(), node);
        self.type_indices
            .entry(type_str)
            .or_insert_with(Vec::new)
            .push(id);
    }

    /// Get statistics
    pub fn stats(&self) -> MemoryStats {
        let mut total_importance = 0.0;
        let mut total_decay = 0.0;
        let mut total_connections = 0u32;
        let count = self.memories.len();

        for entry in self.memories.iter() {
            total_importance += entry.importance;
            total_decay += entry.decay;
            total_connections += entry.connections.len() as u32;
        }

        let count_f = if count > 0 { count as f64 } else { 1.0 };

        MemoryStats {
            total_memories: count as u32,
            avg_importance: total_importance / count_f,
            avg_decay: total_decay / count_f,
            total_connections: total_connections / 2,
            total_stores: self.total_stores.load(Ordering::Relaxed) as f64,
            total_accesses: self.total_accesses.load(Ordering::Relaxed) as f64,
        }
    }

    /// Set distance metric
    pub fn set_metric(&self, metric: &str) {
        let mut config = self.config.write();
        config.metric = match metric.to_lowercase().as_str() {
            "euclidean" => DistanceMetric::Euclidean,
            "dotproduct" | "dot" => DistanceMetric::DotProduct,
            "manhattan" => DistanceMetric::Manhattan,
            _ => DistanceMetric::Cosine,
        };
    }

    /// Enforce memory limits
    fn enforce_limits(&self) {
        let config = self.config.read();
        let count = self.memories.len();

        if count > config.max_memories {
            let mut to_remove: Vec<(String, f64)> = self
                .memories
                .iter()
                .map(|e| (e.id.clone(), e.importance * (1.0 - e.decay)))
                .collect();

            to_remove
                .sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

            let remove_count = count - config.max_memories + config.max_memories / 10;
            for (id, _) in to_remove.into_iter().take(remove_count) {
                self.memories.remove(&id);
            }
        }
    }
}

/// Batch entry for bulk insert
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BatchEntry {
    pub content: String,
    pub vector: Vec<f64>,
    pub memory_type: String,
    pub importance: f64,
}

/// Search result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchResult {
    pub id: String,
    pub content: String,
    pub similarity: f64,
    pub memory_type: String,
    pub importance: f64,
}

/// Consolidation result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConsolidationResult {
    pub merged: u32,
    pub pruned: u32,
    pub total_remaining: u32,
}

/// Memory statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryStats {
    pub total_memories: u32,
    pub avg_importance: f64,
    pub avg_decay: f64,
    pub total_connections: u32,
    pub total_stores: f64,
    pub total_accesses: f64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_store_and_search() {
        let memory = NativeMemory::new(4);

        let id = memory
            .store(
                "Test memory".to_string(),
                vec![1.0, 0.0, 0.0, 0.0],
                "semantic".to_string(),
                0.8,
            )
            .unwrap();

        assert!(!id.is_empty());
        assert_eq!(memory.len(), 1);

        let results = memory
            .search(vec![1.0, 0.0, 0.0, 0.0], 5, None, None)
            .unwrap();

        assert_eq!(results.len(), 1);
        assert!(results[0].similarity > 0.99);
    }

    #[test]
    fn test_store_f32_and_search() {
        let memory = NativeMemory::new(4);

        let id = memory
            .store_f32(
                "Test memory".to_string(),
                vec![1.0f32, 0.0, 0.0, 0.0],
                "semantic".to_string(),
                0.8,
            )
            .unwrap();

        assert!(!id.is_empty());

        let results = memory
            .search_f32(&[1.0, 0.0, 0.0, 0.0], 5, None, None)
            .unwrap();

        assert_eq!(results.len(), 1);
        assert!(results[0].similarity > 0.99);
    }
}
