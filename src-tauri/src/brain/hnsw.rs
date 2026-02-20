//! HNSW (Hierarchical Navigable Small World) index wrapper for DeepBrain
//!
//! Provides O(log n) approximate nearest neighbor search as a secondary index
//! alongside SQLite. Maps string-based IDs to the numeric IDs required by hnsw_rs.

use std::collections::HashMap;
use std::sync::atomic::{AtomicUsize, Ordering};

use hnsw_rs::hnsw::Hnsw;
use hnsw_rs::prelude::*;
use parking_lot::RwLock;

/// HNSW search index wrapper with string ID mapping
pub struct HnswSearchIndex {
    /// The HNSW index (None before first build)
    index: RwLock<Option<Hnsw<'static, f32, DistCosine>>>,
    /// String ID -> numeric ID
    id_to_num: RwLock<HashMap<String, usize>>,
    /// Numeric ID -> String ID
    num_to_id: RwLock<Vec<String>>,
    /// Next numeric ID counter
    next_id: AtomicUsize,
    /// Set of invalidated numeric IDs (pseudo-deletion)
    stale: RwLock<std::collections::HashSet<usize>>,
    /// Vector dimensions
    dimensions: usize,
    /// HNSW parameters
    max_connection: usize,
    ef_construction: usize,
    ef_search: usize,
}

/// Search result from HNSW
#[derive(Debug, Clone)]
pub struct HnswResult {
    pub id: String,
    pub distance: f32,
}

impl HnswSearchIndex {
    /// Create a new empty HNSW index
    pub fn new(dimensions: usize) -> Self {
        Self {
            index: RwLock::new(None),
            id_to_num: RwLock::new(HashMap::new()),
            num_to_id: RwLock::new(Vec::new()),
            next_id: AtomicUsize::new(0),
            stale: RwLock::new(std::collections::HashSet::new()),
            dimensions,
            max_connection: 16,
            ef_construction: 200,
            ef_search: 100,
        }
    }

    /// Build index from a batch of (id, vector) entries.
    /// Replaces any existing index.
    pub fn build_from_entries(&self, entries: Vec<(String, Vec<f32>)>) {
        let count = entries.len();
        if count == 0 {
            *self.index.write() = None;
            return;
        }

        // Reset ID mappings
        let mut id_map = self.id_to_num.write();
        let mut num_map = self.num_to_id.write();
        id_map.clear();
        num_map.clear();
        self.next_id.store(0, Ordering::SeqCst);
        self.stale.write().clear();

        // Estimate max layers: log2(n) / ln(M)
        let max_layer = ((count as f64).log2() / (self.max_connection as f64).ln())
            .ceil()
            .max(4.0) as usize;

        let hnsw = Hnsw::<f32, DistCosine>::new(
            self.max_connection,
            count,
            max_layer,
            self.ef_construction,
            DistCosine {},
        );

        // Prepare data for parallel insert
        let mut data_for_insert: Vec<(&Vec<f32>, usize)> = Vec::with_capacity(count);
        let mut vecs: Vec<Vec<f32>> = Vec::with_capacity(count);

        for (str_id, vec) in &entries {
            let num_id = self.next_id.fetch_add(1, Ordering::SeqCst);
            id_map.insert(str_id.clone(), num_id);
            num_map.push(str_id.clone());
            vecs.push(vec.clone());
            // We'll set up the references after collecting all vecs
            let _ = num_id;
        }

        // Now create the reference tuples
        for (i, vec) in vecs.iter().enumerate() {
            data_for_insert.push((vec, i));
        }

        // Use parallel insert for large batches
        if count > 1000 {
            hnsw.parallel_insert(&data_for_insert);
        } else {
            for (vec, id) in &data_for_insert {
                hnsw.insert((*vec, *id));
            }
        }

        *self.index.write() = Some(hnsw);
    }

    /// Insert a single vector into the index.
    /// If the index hasn't been built yet, creates a new one.
    pub fn insert(&self, id: &str, vector: &[f32]) {
        if vector.len() != self.dimensions {
            return;
        }

        let num_id = {
            let mut id_map = self.id_to_num.write();
            let mut num_map = self.num_to_id.write();

            if let Some(&existing) = id_map.get(id) {
                // If re-inserting same ID, mark old as stale first
                self.stale.write().insert(existing);
                // Assign new numeric ID
                let new_id = self.next_id.fetch_add(1, Ordering::SeqCst);
                id_map.insert(id.to_string(), new_id);
                // Extend num_to_id if needed
                while num_map.len() <= new_id {
                    num_map.push(String::new());
                }
                num_map[new_id] = id.to_string();
                new_id
            } else {
                let new_id = self.next_id.fetch_add(1, Ordering::SeqCst);
                id_map.insert(id.to_string(), new_id);
                while num_map.len() <= new_id {
                    num_map.push(String::new());
                }
                num_map[new_id] = id.to_string();
                new_id
            }
        };

        let mut index_lock = self.index.write();
        if let Some(ref hnsw) = *index_lock {
            hnsw.insert((vector, num_id));
        } else {
            // Create a new index for this first insertion
            let max_layer = 6; // Good default for up to ~1M entries
            let hnsw = Hnsw::<f32, DistCosine>::new(
                self.max_connection,
                10_000, // Initial capacity estimate
                max_layer,
                self.ef_construction,
                DistCosine {},
            );
            hnsw.insert((vector, num_id));
            *index_lock = Some(hnsw);
        }
    }

    /// Mark an ID as stale (pseudo-delete). It won't appear in search results.
    pub fn mark_stale(&self, id: &str) {
        if let Some(&num_id) = self.id_to_num.read().get(id) {
            self.stale.write().insert(num_id);
        }
    }

    /// Search for k nearest neighbors. Returns (id, distance) pairs sorted by distance.
    /// Cosine distance: 0.0 = identical, 2.0 = opposite.
    pub fn search(&self, query: &[f32], k: usize) -> Vec<HnswResult> {
        if query.len() != self.dimensions {
            return Vec::new();
        }

        let index_lock = self.index.read();
        let hnsw = match index_lock.as_ref() {
            Some(h) => h,
            None => return Vec::new(),
        };

        // Over-fetch to compensate for stale entries
        let stale_count = self.stale.read().len();
        let fetch_k = k + stale_count.min(k);
        let neighbours = hnsw.search(query, fetch_k, self.ef_search);

        let stale = self.stale.read();
        let num_to_id = self.num_to_id.read();

        let mut results: Vec<HnswResult> = neighbours
            .into_iter()
            .filter(|n| !stale.contains(&n.d_id))
            .filter_map(|n| {
                let str_id = num_to_id.get(n.d_id)?;
                if str_id.is_empty() {
                    return None;
                }
                Some(HnswResult {
                    id: str_id.clone(),
                    distance: n.distance,
                })
            })
            .take(k)
            .collect();

        // Already sorted by distance from hnsw_rs
        results.truncate(k);
        results
    }

    /// Get the number of vectors in the index (including stale)
    pub fn len(&self) -> usize {
        self.next_id.load(Ordering::SeqCst)
    }

    /// Get the number of active (non-stale) vectors
    pub fn active_len(&self) -> usize {
        let total = self.next_id.load(Ordering::SeqCst);
        let stale = self.stale.read().len();
        total.saturating_sub(stale)
    }

    /// Check if index needs rebuild (>20% stale entries)
    pub fn needs_rebuild(&self) -> bool {
        let total = self.len();
        if total == 0 {
            return false;
        }
        let stale = self.stale.read().len();
        stale * 5 > total // 20% threshold
    }

    /// Check if the index is populated
    pub fn is_populated(&self) -> bool {
        self.index.read().is_some()
    }

    /// Convert HNSW cosine distance to similarity score (1.0 - distance/2.0)
    /// DistCosine returns 1.0 - cos(a,b), so range is [0, 2]
    /// Similarity = 1.0 - distance
    pub fn distance_to_similarity(distance: f32) -> f64 {
        (1.0 - distance) as f64
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hnsw_basic() {
        let index = HnswSearchIndex::new(4);

        // Build from entries
        let entries = vec![
            ("a".to_string(), vec![1.0, 0.0, 0.0, 0.0]),
            ("b".to_string(), vec![0.0, 1.0, 0.0, 0.0]),
            ("c".to_string(), vec![0.9, 0.1, 0.0, 0.0]),
        ];
        index.build_from_entries(entries);

        assert_eq!(index.active_len(), 3);

        // Search for something close to "a"
        let results = index.search(&[1.0, 0.0, 0.0, 0.0], 2);
        assert!(!results.is_empty());
        assert_eq!(results[0].id, "a");
    }

    #[test]
    fn test_hnsw_insert_and_search() {
        let index = HnswSearchIndex::new(4);

        index.insert("x", &[1.0, 0.0, 0.0, 0.0]);
        index.insert("y", &[0.0, 1.0, 0.0, 0.0]);

        let results = index.search(&[0.9, 0.1, 0.0, 0.0], 1);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].id, "x");
    }

    #[test]
    fn test_hnsw_stale_entries() {
        let index = HnswSearchIndex::new(4);

        index.insert("a", &[1.0, 0.0, 0.0, 0.0]);
        index.insert("b", &[0.0, 1.0, 0.0, 0.0]);

        index.mark_stale("a");

        let results = index.search(&[1.0, 0.0, 0.0, 0.0], 2);
        // "a" should be filtered out
        for r in &results {
            assert_ne!(r.id, "a");
        }
    }
}
