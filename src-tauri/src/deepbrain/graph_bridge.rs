//! Knowledge Graph bridge — wraps `ruvector_graph::GraphDB` for DeepBrain memory graph.

use std::collections::{HashSet, VecDeque};
use std::path::Path;

use parking_lot::RwLock;
use ruvector_graph::{
    Edge, GraphDB, Hyperedge, NodeBuilder,
    types::PropertyValue,
};
use serde::{Deserialize, Serialize};

/// Statistics about the knowledge graph.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphStats {
    pub node_count: usize,
    pub edge_count: usize,
    pub hyperedge_count: usize,
}

/// Bridge between DeepBrain and the ruvector-graph knowledge graph.
pub struct GraphBridge {
    db: RwLock<GraphDB>,
}

impl GraphBridge {
    /// Open a persistent graph database at `data_dir/graph.db`.
    pub fn open(data_dir: &Path) -> Result<Self, String> {
        let db_path = data_dir.join("graph.db");

        // Ensure parent directory exists
        if let Some(parent) = db_path.parent() {
            std::fs::create_dir_all(parent)
                .map_err(|e| format!("Failed to create graph data dir: {}", e))?;
        }

        let db = GraphDB::with_storage(&db_path)
            .map_err(|e| format!("Failed to open graph database: {}", e))?;

        Ok(Self { db: RwLock::new(db) })
    }

    /// Create an in-memory graph (fallback if storage fails).
    pub fn in_memory() -> Self {
        Self {
            db: RwLock::new(GraphDB::new()),
        }
    }

    /// Add a memory node to the knowledge graph.
    ///
    /// Returns the node ID on success.
    pub fn add_memory_node(
        &self,
        id: &str,
        memory_type: &str,
        content: &str,
    ) -> Result<String, String> {
        let node = NodeBuilder::new()
            .id(id)
            .label("Memory")
            .label(memory_type)
            .property("content", PropertyValue::String(content.to_string()))
            .property("memory_type", PropertyValue::String(memory_type.to_string()))
            .property(
                "created_at",
                PropertyValue::Integer(
                    std::time::SystemTime::now()
                        .duration_since(std::time::UNIX_EPOCH)
                        .map(|d| d.as_millis() as i64)
                        .unwrap_or(0),
                ),
            )
            .build();

        let db = self.db.write();
        db.create_node(node)
            .map_err(|e| format!("Failed to create graph node: {}", e))
    }

    /// Create a directed edge between two memory nodes.
    ///
    /// Returns the edge ID on success.
    pub fn connect_memories(
        &self,
        from: &str,
        to: &str,
        relationship: &str,
        confidence: f32,
    ) -> Result<String, String> {
        let edge = Edge::create(from.to_string(), to.to_string(), relationship);
        // We'll store confidence as a property by building a full edge
        let edge_with_props = Edge::new(
            uuid::Uuid::new_v4().to_string(),
            from.to_string(),
            to.to_string(),
            relationship.to_string(),
            std::collections::HashMap::from([
                (
                    "confidence".to_string(),
                    PropertyValue::Float(confidence as f64),
                ),
            ]),
        );

        // Suppress unused variable warning
        let _ = edge;

        let db = self.db.write();
        db.create_edge(edge_with_props)
            .map_err(|e| format!("Failed to create graph edge: {}", e))
    }

    /// Auto-connect a node to related nodes if similarity exceeds threshold.
    ///
    /// Returns the number of edges created.
    pub fn auto_connect(&self, id: &str, related_ids: &[String], threshold: f32) -> u32 {
        let mut created = 0u32;
        for related_id in related_ids {
            if related_id == id {
                continue;
            }
            // Connect with "similar_to" relationship
            if self
                .connect_memories(id, related_id, "similar_to", threshold)
                .is_ok()
            {
                created += 1;
            }
        }
        created
    }

    /// Create a hyperedge grouping multiple memory nodes into a cluster.
    ///
    /// Requires at least 2 nodes. Returns the hyperedge ID if created.
    pub fn create_cluster_hyperedge(
        &self,
        ids: &[String],
        desc: &str,
        confidence: f32,
    ) -> Result<Option<String>, String> {
        if ids.len() < 2 {
            return Ok(None);
        }

        let mut hyperedge = Hyperedge::new(ids.to_vec(), "cluster");
        hyperedge.set_description(desc);
        hyperedge.set_confidence(confidence);

        let db = self.db.write();
        let id = db
            .create_hyperedge(hyperedge)
            .map_err(|e| format!("Failed to create hyperedge: {}", e))?;

        Ok(Some(id))
    }

    /// BFS to find all nodes reachable within `hops` edges from `node_id`.
    ///
    /// Returns unique neighbor IDs (excluding the start node).
    pub fn k_hop_neighbors(&self, node_id: &str, hops: u32) -> Vec<String> {
        if hops == 0 {
            return vec![];
        }

        let db = self.db.read();
        let mut visited = HashSet::new();
        let mut queue = VecDeque::new();

        visited.insert(node_id.to_string());
        queue.push_back((node_id.to_string(), 0u32));

        let mut neighbors = Vec::new();

        while let Some((current, depth)) = queue.pop_front() {
            if depth >= hops {
                continue;
            }

            // Outgoing edges
            let outgoing = db.get_outgoing_edges(&current);
            for edge in &outgoing {
                if visited.insert(edge.to.clone()) {
                    neighbors.push(edge.to.clone());
                    queue.push_back((edge.to.clone(), depth + 1));
                }
            }

            // Incoming edges
            let incoming = db.get_incoming_edges(&current);
            for edge in &incoming {
                if visited.insert(edge.from.clone()) {
                    neighbors.push(edge.from.clone());
                    queue.push_back((edge.from.clone(), depth + 1));
                }
            }
        }

        neighbors
    }

    /// Return graph statistics.
    pub fn stats(&self) -> GraphStats {
        let db = self.db.read();
        GraphStats {
            node_count: db.node_count(),
            edge_count: db.edge_count(),
            hyperedge_count: db.hyperedge_count(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_add_and_connect_memories() {
        let bridge = GraphBridge::in_memory();

        let id1 = bridge
            .add_memory_node("mem1", "semantic", "Rust is a systems language")
            .unwrap();
        let id2 = bridge
            .add_memory_node("mem2", "semantic", "Rust has a borrow checker")
            .unwrap();

        assert_eq!(id1, "mem1");
        assert_eq!(id2, "mem2");

        let edge_id = bridge
            .connect_memories("mem1", "mem2", "related_to", 0.8)
            .unwrap();
        assert!(!edge_id.is_empty());

        let stats = bridge.stats();
        assert_eq!(stats.node_count, 2);
        assert_eq!(stats.edge_count, 1);
    }

    #[test]
    fn test_k_hop_neighbors() {
        let bridge = GraphBridge::in_memory();

        bridge.add_memory_node("a", "semantic", "Node A").unwrap();
        bridge.add_memory_node("b", "semantic", "Node B").unwrap();
        bridge.add_memory_node("c", "semantic", "Node C").unwrap();

        bridge.connect_memories("a", "b", "related_to", 0.9).unwrap();
        bridge.connect_memories("b", "c", "related_to", 0.8).unwrap();

        // 1-hop from A should find B
        let hop1 = bridge.k_hop_neighbors("a", 1);
        assert!(hop1.contains(&"b".to_string()));
        assert!(!hop1.contains(&"c".to_string()));

        // 2-hop from A should find B and C
        let hop2 = bridge.k_hop_neighbors("a", 2);
        assert!(hop2.contains(&"b".to_string()));
        assert!(hop2.contains(&"c".to_string()));
    }

    #[test]
    fn test_auto_connect() {
        let bridge = GraphBridge::in_memory();

        bridge.add_memory_node("x", "semantic", "X").unwrap();
        bridge.add_memory_node("y", "semantic", "Y").unwrap();
        bridge.add_memory_node("z", "semantic", "Z").unwrap();

        let created = bridge.auto_connect(
            "x",
            &["y".to_string(), "z".to_string(), "x".to_string()],
            0.6,
        );
        assert_eq!(created, 2);
        assert_eq!(bridge.stats().edge_count, 2);
    }

    #[test]
    fn test_cluster_hyperedge() {
        let bridge = GraphBridge::in_memory();

        bridge.add_memory_node("h1", "semantic", "H1").unwrap();
        bridge.add_memory_node("h2", "semantic", "H2").unwrap();
        bridge.add_memory_node("h3", "semantic", "H3").unwrap();

        let result = bridge
            .create_cluster_hyperedge(
                &["h1".to_string(), "h2".to_string(), "h3".to_string()],
                "test cluster",
                0.7,
            )
            .unwrap();

        assert!(result.is_some());
        assert_eq!(bridge.stats().hyperedge_count, 1);
    }
}
