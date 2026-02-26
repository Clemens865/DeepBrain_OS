import { useEffect, useRef, useState, useCallback } from "react";
import { useAppStore, type GraphStats } from "../store/appStore";
import { invoke } from "../services/backend";

/* ── Types ─────────────────────────────────────────────────────── */

interface GraphNode {
  id: string;
  label: string;
  content: string;
  memoryType: string;
  importance: number;
  x: number;
  y: number;
  vx: number;
  vy: number;
  degree: number;
}

interface GraphEdge {
  from: string;
  to: string;
}

interface Camera {
  x: number;
  y: number;
  scale: number;
}

interface SelectedInfo {
  node: GraphNode;
  connections: GraphNode[];
}

/* ── Constants ─────────────────────────────────────────────────── */

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

const MEMORY_TYPES = Object.keys(TYPE_COLORS);
const DEFAULT_COLOR = "#64748b";

/* ── Helpers ───────────────────────────────────────────────────── */

function getColor(t: string): string {
  return TYPE_COLORS[t] ?? DEFAULT_COLOR;
}

function hexToRgba(hex: string, a: number): string {
  const r = parseInt(hex.slice(1, 3), 16);
  const g = parseInt(hex.slice(3, 5), 16);
  const b = parseInt(hex.slice(5, 7), 16);
  return `rgba(${r},${g},${b},${a})`;
}

function nodeRadius(degree: number): number {
  return 4 + Math.min(degree, 10) * 1.2;
}

/* ── Component ─────────────────────────────────────────────────── */

export default function KnowledgeGraphView() {
  const { graphStats, loadDashboardData } = useAppStore();
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const animRef = useRef<number>(0);

  // Graph data refs (mutated directly, no re-renders)
  const nodesRef = useRef<GraphNode[]>([]);
  const edgesRef = useRef<GraphEdge[]>([]);
  const nodeMapRef = useRef<Map<string, GraphNode>>(new Map());
  const neighborMapRef = useRef<Map<string, Set<string>>>(new Map());

  // Interaction refs
  const hoveredRef = useRef<string | null>(null);
  const selectedRef = useRef<string | null>(null);
  const dragRef = useRef<{ nodeId: string } | null>(null);
  const panRef = useRef<{
    startX: number;
    startY: number;
    camX: number;
    camY: number;
  } | null>(null);
  const didMoveRef = useRef(false);

  // Camera & simulation
  const cameraRef = useRef<Camera>({ x: 0, y: 0, scale: 1 });
  const cameraTargetRef = useRef<{ x: number; y: number } | null>(null);
  const alphaRef = useRef(1.0);

  // React state (for UI that needs re-render)
  const [stats, setStats] = useState<GraphStats | null>(null);
  const [loading, setLoading] = useState(true);
  const [selected, setSelected] = useState<SelectedInfo | null>(null);

  /* ── Build neighbor lookup ───────────────────────────────────── */

  const buildNeighborMap = useCallback((edges: GraphEdge[]) => {
    const map = new Map<string, Set<string>>();
    for (const e of edges) {
      if (!map.has(e.from)) map.set(e.from, new Set());
      if (!map.has(e.to)) map.set(e.to, new Set());
      map.get(e.from)!.add(e.to);
      map.get(e.to)!.add(e.from);
    }
    neighborMapRef.current = map;
  }, []);

  /* ── Select a node (updates ref + React state + camera target) ── */

  const selectNode = useCallback(
    (nodeId: string | null) => {
      selectedRef.current = nodeId;
      if (!nodeId) {
        setSelected(null);
        return;
      }
      const node = nodeMapRef.current.get(nodeId);
      if (!node) {
        setSelected(null);
        return;
      }
      // Build connections list
      const neighborIds = neighborMapRef.current.get(nodeId) ?? new Set();
      const connections: GraphNode[] = [];
      for (const nId of neighborIds) {
        const n = nodeMapRef.current.get(nId);
        if (n) connections.push(n);
      }
      connections.sort((a, b) => b.degree - a.degree);
      setSelected({ node, connections });
      // Pan camera to center on node
      cameraTargetRef.current = { x: node.x, y: node.y };
    },
    []
  );

  /* ── Load graph data (per-type for diversity) ───────────────── */

  const loadGraphData = useCallback(async () => {
    setLoading(true);
    selectNode(null);
    try {
      const s = await invoke<GraphStats>("graph_stats").catch(() => null);
      setStats(s);

      if (s && s.node_count > 0) {
        // Load memories per type for color diversity
        const perType = 25;
        const allMemories: {
          id: string;
          content: string;
          memory_type: string;
          importance: number;
          connection_count: number;
        }[] = [];

        // Fire all type queries in parallel
        const results = await Promise.all(
          MEMORY_TYPES.map((t) =>
            invoke<
              {
                id: string;
                content: string;
                memory_type: string;
                importance: number;
                connection_count: number;
              }[]
            >("list_memories", {
              memoryType: t,
              limit: perType,
              offset: 0,
            }).catch(() => [])
          )
        );
        for (const batch of results) {
          allMemories.push(...batch);
        }

        // Deduplicate
        const nodes: GraphNode[] = [];
        const nodeSet = new Set<string>();
        const nodeMap = new Map<string, GraphNode>();
        const edgeSet = new Set<string>();
        const edges: GraphEdge[] = [];

        // Place nodes in a tight circle
        const angleStep =
          (2 * Math.PI) / Math.max(allMemories.length, 1);
        const initRadius = 60 + allMemories.length * 0.5;

        let idx = 0;
        for (const m of allMemories) {
          if (nodeSet.has(m.id)) continue;
          nodeSet.add(m.id);
          const angle = idx * angleStep;
          const node: GraphNode = {
            id: m.id,
            label: m.content.slice(0, 60),
            content: m.content,
            memoryType: m.memory_type,
            importance: m.importance,
            x:
              Math.cos(angle) * initRadius +
              (Math.random() - 0.5) * 20,
            y:
              Math.sin(angle) * initRadius +
              (Math.random() - 0.5) * 20,
            vx: 0,
            vy: 0,
            degree: m.connection_count ?? 0,
          };
          nodes.push(node);
          nodeMap.set(m.id, node);
          idx++;
        }

        // Discover edges
        for (const node of nodes) {
          const neighbors = await invoke<string[]>("graph_neighbors", {
            nodeId: node.id,
            hops: 1,
          }).catch(() => []);

          for (const nId of neighbors) {
            if (!nodeSet.has(nId)) continue;
            const key = [node.id, nId].sort().join("|");
            if (edgeSet.has(key)) continue;
            edgeSet.add(key);
            edges.push({ from: node.id, to: nId });
          }
        }

        // Recalculate visible degree (edges within loaded set)
        for (const n of nodes) n.degree = 0;
        for (const e of edges) {
          const a = nodeMap.get(e.from);
          const b = nodeMap.get(e.to);
          if (a) a.degree++;
          if (b) b.degree++;
        }

        nodesRef.current = nodes;
        edgesRef.current = edges;
        nodeMapRef.current = nodeMap;
        buildNeighborMap(edges);
        alphaRef.current = 1.0;
        cameraRef.current = { x: 0, y: 0, scale: 1 };
        cameraTargetRef.current = null;
      } else {
        nodesRef.current = [];
        edgesRef.current = [];
        nodeMapRef.current = new Map();
        neighborMapRef.current = new Map();
      }
    } catch (error) {
      console.error("Failed to load graph data:", error);
    }
    setLoading(false);
  }, [buildNeighborMap, selectNode]);

  useEffect(() => {
    loadDashboardData();
    loadGraphData();
  }, [loadDashboardData, loadGraphData]);

  /* ── Resize canvas (HiDPI-aware) ────────────────────────────── */

  useEffect(() => {
    const resize = () => {
      const canvas = canvasRef.current;
      const container = containerRef.current;
      if (!canvas || !container) return;
      const dpr = window.devicePixelRatio || 1;
      const w = container.clientWidth;
      const h = container.clientHeight;
      canvas.width = w * dpr;
      canvas.height = h * dpr;
      canvas.style.width = w + "px";
      canvas.style.height = h + "px";
    };
    resize();
    window.addEventListener("resize", resize);
    return () => window.removeEventListener("resize", resize);
  }, []);

  /* ── Wheel zoom (non-passive for preventDefault) ────────────── */

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const onWheel = (e: WheelEvent) => {
      e.preventDefault();
      const rect = canvas.getBoundingClientRect();
      const dpr = window.devicePixelRatio || 1;
      const w = canvas.width / dpr;
      const h = canvas.height / dpr;
      const sx = e.clientX - rect.left;
      const sy = e.clientY - rect.top;
      const cam = cameraRef.current;

      const wx = (sx - w / 2) / cam.scale + cam.x;
      const wy = (sy - h / 2) / cam.scale + cam.y;

      const factor = e.deltaY > 0 ? 0.92 : 1.08;
      cam.scale = Math.max(0.15, Math.min(6, cam.scale * factor));

      cam.x = wx - (sx - w / 2) / cam.scale;
      cam.y = wy - (sy - h / 2) / cam.scale;
      cameraTargetRef.current = null; // cancel any auto-pan
    };

    canvas.addEventListener("wheel", onWheel, { passive: false });
    return () => canvas.removeEventListener("wheel", onWheel);
  }, []);

  /* ── Main simulation + render loop ──────────────────────────── */

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const tick = () => {
      const nodes = nodesRef.current;
      const edges = edgesRef.current;
      const nodeMap = nodeMapRef.current;
      const neighbors = neighborMapRef.current;
      const cam = cameraRef.current;
      const camTarget = cameraTargetRef.current;
      const dpr = window.devicePixelRatio || 1;
      const w = canvas.width / dpr;
      const h = canvas.height / dpr;
      const alpha = alphaRef.current;
      const focusId = hoveredRef.current || selectedRef.current;
      const focusNeighbors = focusId
        ? neighbors.get(focusId) ?? new Set<string>()
        : null;

      // Smooth camera pan toward target
      if (camTarget) {
        const dx = camTarget.x - cam.x;
        const dy = camTarget.y - cam.y;
        if (Math.abs(dx) < 0.5 && Math.abs(dy) < 0.5) {
          cam.x = camTarget.x;
          cam.y = camTarget.y;
          cameraTargetRef.current = null;
        } else {
          cam.x += dx * 0.12;
          cam.y += dy * 0.12;
        }
      }

      // HiDPI transform
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      ctx.clearRect(0, 0, w, h);

      /* ── Background ──────────────────────────────────────────── */
      const bgGrad = ctx.createRadialGradient(
        w / 2,
        h / 2,
        0,
        w / 2,
        h / 2,
        Math.max(w, h) * 0.7
      );
      bgGrad.addColorStop(0, "#1e1f24");
      bgGrad.addColorStop(1, "#141518");
      ctx.fillStyle = bgGrad;
      ctx.fillRect(0, 0, w, h);

      // Subtle dot grid
      const gs = 40;
      const gOffX = (((-cam.x * cam.scale * 0.3) % gs) + gs) % gs;
      const gOffY = (((-cam.y * cam.scale * 0.3) % gs) + gs) % gs;
      ctx.fillStyle = "rgba(255,255,255,0.02)";
      for (let gx = gOffX; gx < w; gx += gs) {
        for (let gy = gOffY; gy < h; gy += gs) {
          ctx.fillRect(gx - 0.5, gy - 0.5, 1, 1);
        }
      }

      /* ── Empty state ─────────────────────────────────────────── */
      if (nodes.length === 0) {
        ctx.fillStyle = "rgba(255,255,255,0.12)";
        ctx.font = "14px Inter, system-ui, sans-serif";
        ctx.textAlign = "center";
        ctx.fillText(
          loading
            ? "Loading graph..."
            : "No graph data yet. Run bootstrap to populate memories.",
          w / 2,
          h / 2
        );
        animRef.current = requestAnimationFrame(tick);
        return;
      }

      /* ── Physics step ────────────────────────────────────────── */
      if (alpha > 0.003) {
        const REPULSION = 500;
        const SPRING_K = 0.04;
        const SPRING_LEN = 90;
        const GRAVITY = 0.06;
        const DAMPING = 0.5;
        const MAX_VEL = 12;

        for (let i = 0; i < nodes.length; i++) {
          for (let j = i + 1; j < nodes.length; j++) {
            const dx = nodes[j].x - nodes[i].x;
            const dy = nodes[j].y - nodes[i].y;
            const d2 = dx * dx + dy * dy;
            const dist = Math.sqrt(d2) || 0.1;
            const f = (REPULSION * alpha) / d2;
            const fx = (dx / dist) * f;
            const fy = (dy / dist) * f;
            nodes[i].vx -= fx;
            nodes[i].vy -= fy;
            nodes[j].vx += fx;
            nodes[j].vy += fy;
          }
        }

        for (const e of edges) {
          const a = nodeMap.get(e.from);
          const b = nodeMap.get(e.to);
          if (!a || !b) continue;
          const dx = b.x - a.x;
          const dy = b.y - a.y;
          const dist = Math.sqrt(dx * dx + dy * dy) || 0.1;
          const disp = dist - SPRING_LEN;
          const f = disp * SPRING_K * alpha;
          const fx = (dx / dist) * f;
          const fy = (dy / dist) * f;
          a.vx += fx;
          a.vy += fy;
          b.vx -= fx;
          b.vy -= fy;
        }

        for (const n of nodes) {
          if (dragRef.current?.nodeId === n.id) {
            n.vx = 0;
            n.vy = 0;
            continue;
          }
          n.vx -= n.x * GRAVITY * alpha;
          n.vy -= n.y * GRAVITY * alpha;
          n.vx *= DAMPING;
          n.vy *= DAMPING;
          const vel = Math.sqrt(n.vx * n.vx + n.vy * n.vy);
          if (vel > MAX_VEL) {
            n.vx = (n.vx / vel) * MAX_VEL;
            n.vy = (n.vy / vel) * MAX_VEL;
          }
          n.x += n.vx;
          n.y += n.vy;
        }

        alphaRef.current *= 0.995;
      }

      /* ── Camera transform ────────────────────────────────────── */
      ctx.save();
      ctx.translate(w / 2, h / 2);
      ctx.scale(cam.scale, cam.scale);
      ctx.translate(-cam.x, -cam.y);

      const isNodeLit = (id: string) =>
        !focusId || id === focusId || focusNeighbors!.has(id);
      const isEdgeLit = (from: string, to: string) => {
        if (!focusId) return true;
        return (
          (from === focusId && focusNeighbors!.has(to)) ||
          (to === focusId && focusNeighbors!.has(from))
        );
      };

      /* ── Draw edges ──────────────────────────────────────────── */
      for (const e of edges) {
        const a = nodeMap.get(e.from);
        const b = nodeMap.get(e.to);
        if (!a || !b) continue;
        const lit = isEdgeLit(e.from, e.to);
        const eAlpha = lit ? (focusId ? 0.45 : 0.12) : 0.025;
        const color = getColor(a.memoryType);

        ctx.strokeStyle = hexToRgba(color, eAlpha);
        ctx.lineWidth = (lit && focusId ? 1.8 : 0.7) / cam.scale;
        ctx.beginPath();
        ctx.moveTo(a.x, a.y);
        ctx.lineTo(b.x, b.y);
        ctx.stroke();
      }

      /* ── Draw nodes ──────────────────────────────────────────── */
      for (const n of nodes) {
        const color = getColor(n.memoryType);
        const r = nodeRadius(n.degree);
        const lit = isNodeLit(n.id);
        const isFocused = n.id === focusId;
        const isSelected = n.id === selectedRef.current;
        const nAlpha = lit ? 1.0 : 0.08;

        // Glow halo
        if (lit) {
          const glowR = r * (isFocused ? 4.5 : 2.5);
          const grad = ctx.createRadialGradient(
            n.x,
            n.y,
            r * 0.3,
            n.x,
            n.y,
            glowR
          );
          grad.addColorStop(
            0,
            hexToRgba(color, isFocused ? 0.5 : 0.2)
          );
          grad.addColorStop(1, hexToRgba(color, 0));
          ctx.fillStyle = grad;
          ctx.beginPath();
          ctx.arc(n.x, n.y, glowR, 0, Math.PI * 2);
          ctx.fill();
        }

        // Core circle
        ctx.beginPath();
        ctx.arc(n.x, n.y, r, 0, Math.PI * 2);
        ctx.fillStyle = hexToRgba(color, nAlpha);
        ctx.fill();

        // Selected ring (persistent white ring)
        if (isSelected) {
          ctx.strokeStyle = "rgba(255,255,255,0.9)";
          ctx.lineWidth = 2.5 / cam.scale;
          ctx.stroke();
        } else if (isFocused) {
          ctx.strokeStyle = "rgba(255,255,255,0.5)";
          ctx.lineWidth = 1.5 / cam.scale;
          ctx.stroke();
        }

        // Type label for selected node's connections (when zoomed enough)
        if (
          isSelected &&
          cam.scale > 0.6
        ) {
          ctx.font = `${10 / cam.scale}px Inter, system-ui, sans-serif`;
          ctx.fillStyle = "rgba(255,255,255,0.7)";
          ctx.textAlign = "center";
          ctx.fillText(
            n.label.slice(0, 30) + (n.label.length > 30 ? "..." : ""),
            n.x,
            n.y + r + 14 / cam.scale
          );
        }
      }

      ctx.restore(); // back to screen space

      /* ── Tooltip (screen space, only when hovering non-selected) ── */
      const tipId = hoveredRef.current;
      if (tipId && tipId !== selectedRef.current) {
        const n = nodeMap.get(tipId);
        if (n) {
          const sx = (n.x - cam.x) * cam.scale + w / 2;
          const sy = (n.y - cam.y) * cam.scale + h / 2;
          const color = getColor(n.memoryType);
          const label = n.label;
          const typeText = n.memoryType;
          const degreeText = `${n.degree} connection${n.degree !== 1 ? "s" : ""}`;

          ctx.font = "600 11px Inter, system-ui, sans-serif";
          const labelW = ctx.measureText(label).width;
          ctx.font = "9px Inter, system-ui, sans-serif";
          const typeW = ctx.measureText(
            typeText + "  " + degreeText
          ).width;
          const boxW = Math.max(labelW, typeW) + 24;
          const boxH = 40;
          const boxX = Math.max(
            4,
            Math.min(w - boxW - 4, sx - boxW / 2)
          );
          const boxY =
            sy - nodeRadius(n.degree) * cam.scale - boxH - 10;

          ctx.fillStyle = "rgba(0,0,0,0.4)";
          ctx.beginPath();
          ctx.roundRect(boxX + 2, boxY + 2, boxW, boxH, 8);
          ctx.fill();

          ctx.fillStyle = "rgba(18,20,26,0.95)";
          ctx.strokeStyle = hexToRgba(color, 0.35);
          ctx.lineWidth = 1;
          ctx.beginPath();
          ctx.roundRect(boxX, boxY, boxW, boxH, 8);
          ctx.fill();
          ctx.stroke();

          ctx.fillStyle = color;
          ctx.beginPath();
          ctx.roundRect(boxX, boxY, 3, boxH, [8, 0, 0, 8]);
          ctx.fill();

          ctx.font = "600 11px Inter, system-ui, sans-serif";
          ctx.fillStyle = "rgba(255,255,255,0.9)";
          ctx.textAlign = "left";
          ctx.fillText(label, boxX + 12, boxY + 16);

          ctx.font = "9px Inter, system-ui, sans-serif";
          ctx.fillStyle = color;
          ctx.fillText(typeText, boxX + 12, boxY + 30);
          ctx.fillStyle = "rgba(255,255,255,0.35)";
          ctx.fillText(
            degreeText,
            boxX + 12 + ctx.measureText(typeText + "  ").width,
            boxY + 30
          );
        }
      }

      animRef.current = requestAnimationFrame(tick);
    };

    tick();
    return () => cancelAnimationFrame(animRef.current);
  }, [loading]);

  /* ── Coordinate helpers ──────────────────────────────────────── */

  const screenToWorld = useCallback(
    (sx: number, sy: number): { x: number; y: number } => {
      const canvas = canvasRef.current;
      if (!canvas) return { x: sx, y: sy };
      const dpr = window.devicePixelRatio || 1;
      const w = canvas.width / dpr;
      const h = canvas.height / dpr;
      const cam = cameraRef.current;
      return {
        x: (sx - w / 2) / cam.scale + cam.x,
        y: (sy - h / 2) / cam.scale + cam.y,
      };
    },
    []
  );

  const findNodeAt = useCallback(
    (sx: number, sy: number): GraphNode | null => {
      const { x: wx, y: wy } = screenToWorld(sx, sy);
      for (const n of nodesRef.current) {
        const dx = wx - n.x;
        const dy = wy - n.y;
        const hitR = nodeRadius(n.degree) + 5;
        if (dx * dx + dy * dy < hitR * hitR) return n;
      }
      return null;
    },
    [screenToWorld]
  );

  /* ── Mouse handlers ──────────────────────────────────────────── */

  const handleMouseMove = useCallback(
    (e: React.MouseEvent<HTMLCanvasElement>) => {
      const canvas = canvasRef.current;
      if (!canvas) return;
      const rect = canvas.getBoundingClientRect();
      const sx = e.clientX - rect.left;
      const sy = e.clientY - rect.top;

      if (panRef.current) {
        didMoveRef.current = true;
        const cam = cameraRef.current;
        cam.x =
          panRef.current.camX -
          (sx - panRef.current.startX) / cam.scale;
        cam.y =
          panRef.current.camY -
          (sy - panRef.current.startY) / cam.scale;
        cameraTargetRef.current = null;
        return;
      }

      if (dragRef.current) {
        didMoveRef.current = true;
        const { x: wx, y: wy } = screenToWorld(sx, sy);
        const node = nodeMapRef.current.get(dragRef.current.nodeId);
        if (node) {
          node.x = wx;
          node.y = wy;
        }
        return;
      }

      const node = findNodeAt(sx, sy);
      hoveredRef.current = node?.id ?? null;
      canvas.style.cursor = node ? "pointer" : "grab";
    },
    [findNodeAt, screenToWorld]
  );

  const handleMouseDown = useCallback(
    (e: React.MouseEvent<HTMLCanvasElement>) => {
      const canvas = canvasRef.current;
      if (!canvas) return;
      const rect = canvas.getBoundingClientRect();
      const sx = e.clientX - rect.left;
      const sy = e.clientY - rect.top;
      didMoveRef.current = false;

      const node = findNodeAt(sx, sy);
      if (node) {
        dragRef.current = { nodeId: node.id };
        alphaRef.current = Math.max(alphaRef.current, 0.15);
        canvas.style.cursor = "grabbing";
      } else {
        panRef.current = {
          startX: sx,
          startY: sy,
          camX: cameraRef.current.x,
          camY: cameraRef.current.y,
        };
        canvas.style.cursor = "grabbing";
      }
    },
    [findNodeAt]
  );

  const handleMouseUp = useCallback(
    (e: React.MouseEvent<HTMLCanvasElement>) => {
      const wasDrag = didMoveRef.current;
      dragRef.current = null;
      panRef.current = null;

      const canvas = canvasRef.current;
      if (!canvas) return;
      canvas.style.cursor = "grab";

      // Click (no movement) — select/deselect
      if (!wasDrag) {
        const rect = canvas.getBoundingClientRect();
        const sx = e.clientX - rect.left;
        const sy = e.clientY - rect.top;
        const node = findNodeAt(sx, sy);
        if (node) {
          // Toggle if same node, otherwise select new
          selectNode(
            selectedRef.current === node.id ? null : node.id
          );
        } else {
          selectNode(null);
        }
      }
    },
    [findNodeAt, selectNode]
  );

  const handleMouseLeave = useCallback(() => {
    dragRef.current = null;
    panRef.current = null;
    hoveredRef.current = null;
  }, []);

  /* ── Navigate to a connection (from detail panel) ───────────── */

  const navigateToNode = useCallback(
    (nodeId: string) => {
      selectNode(nodeId);
    },
    [selectNode]
  );

  /* ── Render ──────────────────────────────────────────────────── */

  return (
    <div className="h-full flex flex-col">
      {/* Header bar */}
      <div className="flex-shrink-0 p-3 border-b border-brain-border bg-brain-surface/50">
        <div className="flex items-center justify-between">
          <h2 className="text-white text-sm font-medium">
            Knowledge Graph
          </h2>
          <div className="flex items-center gap-2">
            <button
              onClick={() => {
                cameraRef.current = { x: 0, y: 0, scale: 1 };
                cameraTargetRef.current = null;
              }}
              className="px-2 py-1 bg-brain-surface text-brain-text/60 text-[10px] rounded-md hover:bg-brain-border hover:text-brain-text transition-colors"
              title="Reset zoom and pan"
            >
              Reset View
            </button>
            <button
              onClick={() => {
                alphaRef.current = 0.8;
              }}
              className="px-2 py-1 bg-brain-surface text-brain-text/60 text-[10px] rounded-md hover:bg-brain-border hover:text-brain-text transition-colors"
              title="Re-run layout simulation"
            >
              Re-layout
            </button>
            <button
              onClick={loadGraphData}
              className="px-3 py-1 bg-brain-accent/20 text-brain-accent text-[11px] rounded-lg hover:bg-brain-accent/30 transition-colors"
            >
              Refresh
            </button>
          </div>
        </div>

        {/* Stats row */}
        {(stats || graphStats) && (
          <div className="flex gap-4 mt-2">
            {[
              {
                label: "Nodes",
                value: (stats || graphStats)!.node_count,
              },
              {
                label: "Edges",
                value: (stats || graphStats)!.edge_count,
              },
              {
                label: "Clusters",
                value: (stats || graphStats)!.hyperedge_count,
              },
            ].map((s) => (
              <div key={s.label} className="text-center">
                <div className="text-brain-text/40 text-[10px] uppercase tracking-wider">
                  {s.label}
                </div>
                <div className="text-white text-sm font-semibold">
                  {s.value}
                </div>
              </div>
            ))}
          </div>
        )}

        {/* Legend */}
        <div className="flex flex-wrap gap-3 mt-2">
          {Object.entries(TYPE_COLORS).map(([type, color]) => (
            <div key={type} className="flex items-center gap-1">
              <div
                className="w-2 h-2 rounded-full"
                style={{ backgroundColor: color }}
              />
              <span className="text-brain-text/40 text-[9px] capitalize">
                {type}
              </span>
            </div>
          ))}
        </div>

        <div className="text-brain-text/25 text-[9px] mt-1">
          Scroll to zoom · Drag to pan · Click a node to explore
          its connections
        </div>
      </div>

      {/* Canvas + detail panel area */}
      <div ref={containerRef} className="flex-1 min-h-0 relative">
        <canvas
          ref={canvasRef}
          onMouseMove={handleMouseMove}
          onMouseDown={handleMouseDown}
          onMouseUp={handleMouseUp}
          onMouseLeave={handleMouseLeave}
          className="absolute inset-0 cursor-grab"
        />

        {/* Detail panel — slides in from right when a node is selected */}
        {selected && (
          <div className="absolute top-0 right-0 bottom-0 w-[300px] bg-brain-surface/95 backdrop-blur-sm border-l border-brain-border flex flex-col overflow-hidden">
            {/* Panel header */}
            <div className="flex-shrink-0 p-3 border-b border-brain-border">
              <div className="flex items-center justify-between mb-2">
                <div className="flex items-center gap-2">
                  <div
                    className="w-3 h-3 rounded-full flex-shrink-0"
                    style={{
                      backgroundColor: getColor(
                        selected.node.memoryType
                      ),
                    }}
                  />
                  <span className="text-[10px] uppercase tracking-wider text-brain-text/50">
                    {selected.node.memoryType}
                  </span>
                </div>
                <button
                  onClick={() => selectNode(null)}
                  className="text-brain-text/40 hover:text-white text-sm transition-colors px-1"
                  title="Close panel"
                >
                  ✕
                </button>
              </div>
              <div className="flex gap-3 text-[10px]">
                <span className="text-brain-text/40">
                  {selected.node.degree} connection
                  {selected.node.degree !== 1 ? "s" : ""}
                </span>
                <span className="text-brain-text/40">
                  importance:{" "}
                  {selected.node.importance.toFixed(2)}
                </span>
              </div>
            </div>

            {/* Content */}
            <div className="flex-shrink-0 max-h-[40%] overflow-y-auto p-3 border-b border-brain-border">
              <div className="text-[10px] uppercase tracking-wider text-brain-text/30 mb-1">
                Content
              </div>
              <p className="text-brain-text text-[12px] leading-relaxed whitespace-pre-wrap break-words">
                {selected.node.content}
              </p>
            </div>

            {/* Connections list */}
            <div className="flex-1 min-h-0 overflow-y-auto">
              <div className="p-3">
                <div className="text-[10px] uppercase tracking-wider text-brain-text/30 mb-2">
                  Connections ({selected.connections.length})
                </div>
                {selected.connections.length === 0 ? (
                  <p className="text-brain-text/30 text-[11px]">
                    No visible connections
                  </p>
                ) : (
                  <div className="space-y-1">
                    {selected.connections.map((conn) => (
                      <button
                        key={conn.id}
                        onClick={() => navigateToNode(conn.id)}
                        className="w-full text-left p-2 rounded-lg hover:bg-brain-bg/60 transition-colors group"
                      >
                        <div className="flex items-center gap-2 mb-0.5">
                          <div
                            className="w-2 h-2 rounded-full flex-shrink-0"
                            style={{
                              backgroundColor: getColor(
                                conn.memoryType
                              ),
                            }}
                          />
                          <span className="text-[9px] uppercase tracking-wider text-brain-text/40">
                            {conn.memoryType}
                          </span>
                          <span className="text-[9px] text-brain-text/25 ml-auto">
                            {conn.degree} conn.
                          </span>
                        </div>
                        <p className="text-brain-text/70 text-[11px] leading-snug line-clamp-2 group-hover:text-brain-text transition-colors">
                          {conn.label}
                        </p>
                      </button>
                    ))}
                  </div>
                )}
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
