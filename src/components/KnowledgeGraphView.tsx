import { useEffect, useRef, useState, useCallback } from "react";
import { useAppStore, type GraphStats } from "../store/appStore";
import { invoke } from "../services/backend";

interface GraphNode {
  id: string;
  label: string;
  memoryType: string;
  x: number;
  y: number;
  vx: number;
  vy: number;
}

interface GraphEdge {
  from: string;
  to: string;
}

const TYPE_COLORS: Record<string, string> = {
  semantic: "#6366f1",
  episodic: "#f59e0b",
  working: "#10b981",
  procedural: "#ef4444",
  meta: "#8b5cf6",
  causal: "#ec4899",
  goal: "#14b8a6",
  emotional: "#f97316",
};

function getColor(memoryType: string): string {
  return TYPE_COLORS[memoryType] ?? "#64748b";
}

export default function KnowledgeGraphView() {
  const { graphStats, loadDashboardData } = useAppStore();
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const animRef = useRef<number>(0);
  const nodesRef = useRef<GraphNode[]>([]);
  const edgesRef = useRef<GraphEdge[]>([]);
  const [hoveredNode, setHoveredNode] = useState<GraphNode | null>(null);
  const [stats, setStats] = useState<GraphStats | null>(null);
  const [loading, setLoading] = useState(true);
  const dragRef = useRef<{ nodeId: string; offsetX: number; offsetY: number } | null>(null);

  // Load graph data
  const loadGraphData = useCallback(async () => {
    setLoading(true);
    try {
      // Get stats
      const s = await invoke<GraphStats>("graph_stats").catch(() => null);
      setStats(s);

      // If graph has nodes, load memories and discover their neighbors
      if (s && s.node_count > 0) {
        // Load recent memories to get node IDs
        const memories = await invoke<{ id: string; content: string; memory_type: string }[]>(
          "list_memories",
          { memoryType: null, limit: 50, offset: 0 }
        ).catch(() => []);

        const nodes: GraphNode[] = [];
        const edges: GraphEdge[] = [];
        const nodeSet = new Set<string>();
        const width = containerRef.current?.clientWidth ?? 800;
        const height = containerRef.current?.clientHeight ?? 600;

        // Create nodes from memories
        for (const m of memories) {
          if (!nodeSet.has(m.id)) {
            nodeSet.add(m.id);
            nodes.push({
              id: m.id,
              label: m.content.slice(0, 40),
              memoryType: m.memory_type,
              x: width / 2 + (Math.random() - 0.5) * width * 0.6,
              y: height / 2 + (Math.random() - 0.5) * height * 0.6,
              vx: 0,
              vy: 0,
            });
          }
        }

        // Discover edges via graph neighbors
        for (const node of nodes) {
          const neighbors = await invoke<string[]>("graph_neighbors", {
            nodeId: node.id,
            hops: 1,
          }).catch(() => []);

          for (const nId of neighbors) {
            if (nodeSet.has(nId)) {
              // Only add edge if both nodes exist in our set
              const edgeKey = [node.id, nId].sort().join("-");
              if (!edges.some((e) => [e.from, e.to].sort().join("-") === edgeKey)) {
                edges.push({ from: node.id, to: nId });
              }
            }
          }
        }

        nodesRef.current = nodes;
        edgesRef.current = edges;
      } else {
        nodesRef.current = [];
        edgesRef.current = [];
      }
    } catch (error) {
      console.error("Failed to load graph data:", error);
    }
    setLoading(false);
  }, []);

  useEffect(() => {
    loadDashboardData();
    loadGraphData();
  }, [loadDashboardData, loadGraphData]);

  // Force-directed simulation + render loop
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const simulate = () => {
      const nodes = nodesRef.current;
      const edges = edgesRef.current;
      const width = canvas.width;
      const height = canvas.height;

      if (nodes.length === 0) {
        // Draw empty state
        ctx.clearRect(0, 0, width, height);
        ctx.fillStyle = "rgba(255,255,255,0.15)";
        ctx.font = "14px sans-serif";
        ctx.textAlign = "center";
        ctx.fillText(
          loading ? "Loading graph..." : "No graph data yet. Memories will appear here as connections form.",
          width / 2,
          height / 2
        );
        animRef.current = requestAnimationFrame(simulate);
        return;
      }

      // Force simulation step
      const repulsion = 2000;
      const attraction = 0.005;
      const damping = 0.9;
      const centerPull = 0.001;

      // Repulsion between all pairs
      for (let i = 0; i < nodes.length; i++) {
        for (let j = i + 1; j < nodes.length; j++) {
          const dx = nodes[j].x - nodes[i].x;
          const dy = nodes[j].y - nodes[i].y;
          const dist = Math.sqrt(dx * dx + dy * dy) || 1;
          const force = repulsion / (dist * dist);
          const fx = (dx / dist) * force;
          const fy = (dy / dist) * force;
          nodes[i].vx -= fx;
          nodes[i].vy -= fy;
          nodes[j].vx += fx;
          nodes[j].vy += fy;
        }
      }

      // Attraction along edges
      for (const edge of edges) {
        const a = nodes.find((n) => n.id === edge.from);
        const b = nodes.find((n) => n.id === edge.to);
        if (!a || !b) continue;
        const dx = b.x - a.x;
        const dy = b.y - a.y;
        const fx = dx * attraction;
        const fy = dy * attraction;
        a.vx += fx;
        a.vy += fy;
        b.vx -= fx;
        b.vy -= fy;
      }

      // Center pull + update positions
      for (const node of nodes) {
        // Skip dragged node
        if (dragRef.current?.nodeId === node.id) {
          node.vx = 0;
          node.vy = 0;
          continue;
        }
        node.vx += (width / 2 - node.x) * centerPull;
        node.vy += (height / 2 - node.y) * centerPull;
        node.vx *= damping;
        node.vy *= damping;
        node.x += node.vx;
        node.y += node.vy;
        // Clamp to bounds
        node.x = Math.max(20, Math.min(width - 20, node.x));
        node.y = Math.max(20, Math.min(height - 20, node.y));
      }

      // Render
      ctx.clearRect(0, 0, width, height);

      // Draw edges
      ctx.strokeStyle = "rgba(255,255,255,0.08)";
      ctx.lineWidth = 1;
      for (const edge of edges) {
        const a = nodes.find((n) => n.id === edge.from);
        const b = nodes.find((n) => n.id === edge.to);
        if (!a || !b) continue;
        ctx.beginPath();
        ctx.moveTo(a.x, a.y);
        ctx.lineTo(b.x, b.y);
        ctx.stroke();
      }

      // Draw nodes
      for (const node of nodes) {
        const isHovered = hoveredNode?.id === node.id;
        const radius = isHovered ? 7 : 5;
        const color = getColor(node.memoryType);

        ctx.beginPath();
        ctx.arc(node.x, node.y, radius, 0, Math.PI * 2);
        ctx.fillStyle = color;
        ctx.fill();

        if (isHovered) {
          ctx.strokeStyle = "rgba(255,255,255,0.5)";
          ctx.lineWidth = 1.5;
          ctx.stroke();
        }
      }

      // Draw hovered node tooltip
      if (hoveredNode) {
        const node = nodes.find((n) => n.id === hoveredNode.id);
        if (node) {
          ctx.fillStyle = "rgba(15,23,42,0.9)";
          ctx.strokeStyle = "rgba(255,255,255,0.15)";
          ctx.lineWidth = 1;
          const text = node.label;
          const typeText = node.memoryType;
          const textWidth = Math.max(ctx.measureText(text).width, ctx.measureText(typeText).width);
          const boxWidth = textWidth + 16;
          const boxX = node.x - boxWidth / 2;
          const boxY = node.y - 40;

          ctx.beginPath();
          ctx.roundRect(boxX, boxY, boxWidth, 30, 4);
          ctx.fill();
          ctx.stroke();

          ctx.fillStyle = "rgba(255,255,255,0.7)";
          ctx.font = "11px sans-serif";
          ctx.textAlign = "center";
          ctx.fillText(text, node.x, boxY + 12);
          ctx.fillStyle = getColor(node.memoryType);
          ctx.font = "9px sans-serif";
          ctx.fillText(typeText, node.x, boxY + 24);
        }
      }

      animRef.current = requestAnimationFrame(simulate);
    };

    simulate();

    return () => {
      cancelAnimationFrame(animRef.current);
    };
  }, [loading, hoveredNode]);

  // Resize canvas to container
  useEffect(() => {
    const resize = () => {
      const canvas = canvasRef.current;
      const container = containerRef.current;
      if (!canvas || !container) return;
      canvas.width = container.clientWidth;
      canvas.height = container.clientHeight;
    };
    resize();
    window.addEventListener("resize", resize);
    return () => window.removeEventListener("resize", resize);
  }, []);

  // Mouse interaction handlers
  const findNodeAt = useCallback((x: number, y: number): GraphNode | null => {
    for (const node of nodesRef.current) {
      const dx = x - node.x;
      const dy = y - node.y;
      if (dx * dx + dy * dy < 100) return node; // radius ~10px
    }
    return null;
  }, []);

  const handleMouseMove = useCallback(
    (e: React.MouseEvent<HTMLCanvasElement>) => {
      const canvas = canvasRef.current;
      if (!canvas) return;
      const rect = canvas.getBoundingClientRect();
      const x = e.clientX - rect.left;
      const y = e.clientY - rect.top;

      if (dragRef.current) {
        const node = nodesRef.current.find((n) => n.id === dragRef.current!.nodeId);
        if (node) {
          node.x = x;
          node.y = y;
        }
        return;
      }

      const node = findNodeAt(x, y);
      setHoveredNode(node);
      canvas.style.cursor = node ? "pointer" : "default";
    },
    [findNodeAt]
  );

  const handleMouseDown = useCallback(
    (e: React.MouseEvent<HTMLCanvasElement>) => {
      const canvas = canvasRef.current;
      if (!canvas) return;
      const rect = canvas.getBoundingClientRect();
      const x = e.clientX - rect.left;
      const y = e.clientY - rect.top;
      const node = findNodeAt(x, y);
      if (node) {
        dragRef.current = { nodeId: node.id, offsetX: x - node.x, offsetY: y - node.y };
      }
    },
    [findNodeAt]
  );

  const handleMouseUp = useCallback(() => {
    dragRef.current = null;
  }, []);

  return (
    <div className="h-full flex flex-col">
      {/* Header bar */}
      <div className="flex-shrink-0 p-3 border-b border-brain-border bg-brain-surface/50">
        <div className="flex items-center justify-between">
          <h2 className="text-white text-sm font-medium">Knowledge Graph</h2>
          <button
            onClick={loadGraphData}
            className="px-3 py-1 bg-brain-accent/20 text-brain-accent text-[11px] rounded-lg hover:bg-brain-accent/30 transition-colors"
          >
            Refresh
          </button>
        </div>
        {/* Stats row */}
        {(stats || graphStats) && (
          <div className="flex gap-4 mt-2">
            {[
              { label: "Nodes", value: (stats || graphStats)!.node_count },
              { label: "Edges", value: (stats || graphStats)!.edge_count },
              { label: "Clusters", value: (stats || graphStats)!.hyperedge_count },
            ].map((s) => (
              <div key={s.label} className="text-center">
                <div className="text-brain-text/40 text-[10px] uppercase tracking-wider">{s.label}</div>
                <div className="text-white text-sm font-semibold">{s.value}</div>
              </div>
            ))}
          </div>
        )}
        {/* Legend */}
        <div className="flex flex-wrap gap-3 mt-2">
          {Object.entries(TYPE_COLORS).map(([type, color]) => (
            <div key={type} className="flex items-center gap-1">
              <div className="w-2 h-2 rounded-full" style={{ backgroundColor: color }} />
              <span className="text-brain-text/40 text-[9px] capitalize">{type}</span>
            </div>
          ))}
        </div>
      </div>

      {/* Canvas area */}
      <div ref={containerRef} className="flex-1 min-h-0 relative bg-brain-bg">
        <canvas
          ref={canvasRef}
          onMouseMove={handleMouseMove}
          onMouseDown={handleMouseDown}
          onMouseUp={handleMouseUp}
          onMouseLeave={handleMouseUp}
          className="absolute inset-0"
        />
      </div>
    </div>
  );
}
