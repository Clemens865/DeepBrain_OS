import { useEffect, useState, useCallback } from "react";
import { useAppStore } from "../store/appStore";

function formatBytes(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  if (bytes < 1024 * 1024 * 1024) return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
  return `${(bytes / (1024 * 1024 * 1024)).toFixed(2)} GB`;
}

function formatUs(us: number): string {
  if (us < 1000) return `${us}us`;
  return `${(us / 1000).toFixed(1)}ms`;
}

function formatUptime(ms: number): string {
  if (ms < 60000) return `${Math.floor(ms / 1000)}s`;
  if (ms < 3600000) return `${Math.floor(ms / 60000)}m`;
  return `${Math.floor(ms / 3600000)}h ${Math.floor((ms % 3600000) / 60000)}m`;
}

function Dot({ ok }: { ok: boolean }) {
  return (
    <div className={`w-2 h-2 rounded-full flex-shrink-0 ${ok ? "bg-brain-success" : "bg-brain-error"}`} />
  );
}

function Stat({ label, value }: { label: string; value: string | number }) {
  return (
    <div>
      <div className="text-brain-text/40 text-[10px] uppercase tracking-wider">{label}</div>
      <div className="text-white text-sm font-semibold mt-0.5">{value}</div>
    </div>
  );
}

function Panel({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <div className="bg-brain-surface rounded-xl p-4 border border-brain-border">
      <h3 className="text-white text-xs font-medium mb-3">{title}</h3>
      {children}
    </div>
  );
}

export default function DashboardView() {
  const {
    status,
    storageMetrics,
    sonaStats,
    nervousStats,
    llmStatus,
    graphStats,
    compressionStats,
    verifyResult,
    verifying,
    loadStatus,
    loadDashboardData,
    verifyAll,
    loadModel,
    unloadModel,
    bootstrapSona,
    bootstrapKnowledge,
    bootstrapRunning,
    bootstrapProgress,
    bootstrapResult,
  } = useAppStore();

  const [modelInput, setModelInput] = useState("");
  const [bootstrapping, setBootstrapping] = useState(false);
  const [selectedSources, setSelectedSources] = useState<Set<string>>(
    new Set(["file_index", "claude_code", "claude_memory", "whatsapp", "browser", "deepnote"])
  );

  const handleBootstrap = useCallback(async () => {
    setBootstrapping(true);
    await bootstrapSona();
    setBootstrapping(false);
  }, [bootstrapSona]);

  const toggleSource = useCallback((source: string) => {
    setSelectedSources(prev => {
      const next = new Set(prev);
      if (next.has(source)) next.delete(source);
      else next.add(source);
      return next;
    });
  }, []);

  const handleBootstrapKnowledge = useCallback(async () => {
    if (selectedSources.size === 0) return;
    await bootstrapKnowledge(Array.from(selectedSources));
  }, [bootstrapKnowledge, selectedSources]);

  // Listen for bootstrap progress events from Tauri
  useEffect(() => {
    let unlisten: (() => void) | null = null;
    (async () => {
      try {
        const { listen } = await import("@tauri-apps/api/event");
        const fn = await listen<{
          source: string;
          phase: string;
          current: number;
          total: number;
          memories_created: number;
        }>("bootstrap-progress", (event) => {
          useAppStore.setState({ bootstrapProgress: event.payload });
        });
        unlisten = fn;
      } catch {
        // Not in Tauri context
      }
    })();
    return () => { unlisten?.(); };
  }, []);

  // Load data on mount and auto-refresh every 10s
  useEffect(() => {
    loadStatus();
    loadDashboardData();
    const interval = setInterval(() => {
      loadStatus();
      loadDashboardData();
    }, 10000);
    return () => clearInterval(interval);
  }, [loadStatus, loadDashboardData]);

  return (
    <div className="p-4 space-y-4 overflow-y-auto h-full">
      {/* System Overview */}
      {status && (
        <div className="bg-brain-surface rounded-xl p-4 border border-brain-border">
          <div className="flex items-center gap-2 mb-3">
            <div className={`w-2.5 h-2.5 rounded-full ${status.status === "healthy" ? "bg-brain-success" : "bg-brain-warning"}`} />
            <span className="text-white text-sm font-medium">DeepBrain Dashboard</span>
            <span className="text-brain-text/30 text-[10px] ml-auto">
              uptime {formatUptime(status.uptime_ms)} &middot; {status.learning_trend}
            </span>
          </div>
          <div className="grid grid-cols-5 gap-3">
            <Stat label="Memories" value={status.memory_count} />
            <Stat label="Thoughts" value={status.thought_count} />
            <Stat label="Files" value={status.indexed_files} />
            <Stat label="Chunks" value={status.indexed_chunks} />
            <Stat label="Emails" value={status.indexed_emails} />
          </div>
        </div>
      )}

      {/* Knowledge Bootstrap */}
      <div className="bg-brain-surface rounded-xl p-4 border border-brain-border">
        <h3 className="text-white text-xs font-medium mb-3">Bootstrap Knowledge</h3>
        <div className="grid grid-cols-3 gap-2 mb-3">
          {[
            { id: "file_index", label: "Documents", desc: "md, pdf, txt, html..." },
            { id: "claude_code", label: "Claude Code", desc: "Conversations" },
            { id: "claude_memory", label: "Claude Memory", desc: "MEMORY.md files" },
            { id: "claude_plans", label: "Claude Plans", desc: "Implementation plans" },
            { id: "claude_desktop", label: "Claude Desktop", desc: "Agent sessions" },
            { id: "claude_history", label: "Claude History", desc: "Session prompts" },
            { id: "whatsapp", label: "WhatsApp", desc: "Chat messages" },
            { id: "browser", label: "Browser", desc: "Comet history" },
            { id: "deepnote", label: "Deepnote", desc: "AI notebooks" },
          ].map(({ id, label, desc }) => (
            <button
              key={id}
              onClick={() => toggleSource(id)}
              disabled={bootstrapRunning}
              className={`flex flex-col items-start px-3 py-2 rounded-lg border text-left transition-colors ${
                selectedSources.has(id)
                  ? "border-brain-accent bg-brain-accent/10 text-brain-accent"
                  : "border-brain-border bg-brain-bg/50 text-brain-text/40"
              } disabled:opacity-50`}
            >
              <span className="text-[11px] font-medium">{label}</span>
              <span className="text-[9px] opacity-60">{desc}</span>
            </button>
          ))}
        </div>

        {bootstrapProgress && (
          <div className="mb-3">
            <div className="flex justify-between text-[10px] mb-1">
              <span className="text-brain-text/60">
                {bootstrapProgress.source}: {bootstrapProgress.phase}
              </span>
              <span className="text-brain-text/40">
                {bootstrapProgress.current}/{bootstrapProgress.total}
              </span>
            </div>
            <div className="h-1.5 bg-brain-bg rounded-full overflow-hidden">
              <div
                className="h-full bg-brain-accent rounded-full transition-all duration-300"
                style={{
                  width: `${bootstrapProgress.total > 0 ? (bootstrapProgress.current / bootstrapProgress.total) * 100 : 0}%`,
                }}
              />
            </div>
            <div className="text-brain-text/30 text-[9px] mt-1">
              {bootstrapProgress.memories_created} memories created so far
            </div>
          </div>
        )}

        {bootstrapResult && !bootstrapRunning && (
          <div className="mb-3 bg-brain-bg/50 rounded-lg p-3">
            <div className="text-brain-success text-[11px] font-medium mb-1">
              Bootstrap Complete
            </div>
            <div className="grid grid-cols-3 gap-2 text-[10px]">
              <div>
                <span className="text-brain-text/40">Created:</span>{" "}
                <span className="text-white">{bootstrapResult.total_memories_created}</span>
              </div>
              <div>
                <span className="text-brain-text/40">Skipped:</span>{" "}
                <span className="text-white">{bootstrapResult.total_skipped}</span>
              </div>
              <div>
                <span className="text-brain-text/40">Duration:</span>{" "}
                <span className="text-white">{bootstrapResult.duration_secs.toFixed(1)}s</span>
              </div>
            </div>
            {bootstrapResult.sources.length > 0 && (
              <div className="mt-2 space-y-1">
                {bootstrapResult.sources.map((s) => (
                  <div key={s.source} className="flex items-center gap-2 text-[10px]">
                    <Dot ok={s.errors === 0} />
                    <span className="text-brain-text/60">{s.source}</span>
                    <span className="text-brain-text/30 ml-auto">
                      +{s.memories_created} / ~{s.skipped_existing}
                    </span>
                  </div>
                ))}
              </div>
            )}
          </div>
        )}

        <button
          onClick={handleBootstrapKnowledge}
          disabled={bootstrapRunning || selectedSources.size === 0}
          className="w-full px-3 py-2 bg-brain-accent/20 text-brain-accent text-[11px] rounded-lg hover:bg-brain-accent/30 transition-colors disabled:opacity-50"
        >
          {bootstrapRunning
            ? "Bootstrapping..."
            : `Bootstrap ${selectedSources.size === 9 ? "All Sources" : `${selectedSources.size} Source${selectedSources.size !== 1 ? "s" : ""}`}`}
        </button>
      </div>

      {/* Subsystem Grid */}
      <div className="grid grid-cols-2 gap-3">
        {/* Storage Panel */}
        <Panel title="Vector Storage">
          {storageMetrics ? (
            <div className="space-y-2">
              <div className="grid grid-cols-2 gap-2">
                <Stat label="Vectors" value={storageMetrics.total_vectors} />
                <Stat label="File Size" value={formatBytes(storageMetrics.file_size_bytes)} />
              </div>
              <div className="border-t border-brain-border/50 pt-2 mt-2">
                <div className="text-brain-text/30 text-[10px]">
                  Query latency: p50 {formatUs(storageMetrics.query_latency_p50_us)} / p99 {formatUs(storageMetrics.query_latency_p99_us)}
                </div>
              </div>
            </div>
          ) : (
            <div className="text-brain-text/30 text-xs">Loading...</div>
          )}
        </Panel>

        {/* SONA Panel */}
        <Panel title="SONA Learning">
          {sonaStats ? (
            <div className="space-y-2">
              <div className="grid grid-cols-2 gap-2">
                <Stat label="Patterns" value={sonaStats.patterns_stored} />
                <Stat label="Buffered" value={sonaStats.trajectories_buffered} />
                <Stat label="Dropped" value={sonaStats.trajectories_dropped} />
                <Stat label="EWC Tasks" value={sonaStats.ewc_tasks} />
              </div>
              <div className="flex gap-3 border-t border-brain-border/50 pt-2 mt-2">
                <div className="flex items-center gap-1.5">
                  <Dot ok={sonaStats.instant_enabled} />
                  <span className="text-brain-text/40 text-[10px]">Instant Loop</span>
                </div>
                <div className="flex items-center gap-1.5">
                  <Dot ok={sonaStats.background_enabled} />
                  <span className="text-brain-text/40 text-[10px]">Background Loop</span>
                </div>
              </div>
              {sonaStats.patterns_stored === 0 && (
                <button
                  onClick={handleBootstrap}
                  disabled={bootstrapping}
                  className="w-full mt-2 px-3 py-1.5 bg-brain-accent/20 text-brain-accent text-[11px] rounded-lg hover:bg-brain-accent/30 transition-colors disabled:opacity-50"
                >
                  {bootstrapping ? "Bootstrapping..." : "Bootstrap Learning from File Index"}
                </button>
              )}
            </div>
          ) : (
            <div className="text-brain-text/30 text-xs">Loading...</div>
          )}
        </Panel>

        {/* Nervous System Panel */}
        <Panel title="Nervous System">
          {nervousStats ? (
            <div className="space-y-2">
              <div className="grid grid-cols-2 gap-2">
                <Stat label="Hopfield" value={`${nervousStats.hopfield_patterns} patterns`} />
                <Stat label="Capacity" value={nervousStats.hopfield_capacity.toLocaleString()} />
              </div>
              {/* Sync gauge */}
              <div>
                <div className="flex justify-between text-[10px] mb-1">
                  <span className="text-brain-text/40">Router Sync</span>
                  <span className="text-brain-text/60">{(nervousStats.router_sync * 100).toFixed(0)}%</span>
                </div>
                <div className="h-1.5 bg-brain-bg rounded-full overflow-hidden">
                  <div
                    className="h-full bg-brain-accent rounded-full transition-all duration-500"
                    style={{ width: `${Math.min(nervousStats.router_sync * 100, 100)}%` }}
                  />
                </div>
              </div>
              <div className="border-t border-brain-border/50 pt-2 mt-2 text-brain-text/30 text-[10px] space-y-0.5">
                <div>Beta: {nervousStats.hopfield_beta} &middot; Modules: {nervousStats.router_modules}</div>
                <div>Predictive threshold: {nervousStats.predictive_threshold}</div>
                <div>Dentate: {nervousStats.dentate_input_dim}d &rarr; {nervousStats.dentate_output_dim}d (k={nervousStats.dentate_k})</div>
              </div>
            </div>
          ) : (
            <div className="text-brain-text/30 text-xs">Loading...</div>
          )}
        </Panel>

        {/* LLM Panel */}
        <Panel title="Local LLM (ruvllm)">
          <div className="space-y-2">
            <div className="flex items-center gap-2">
              <Dot ok={llmStatus?.model_loaded ?? false} />
              <span className="text-brain-text/60 text-xs">
                {llmStatus?.model_loaded ? "Model loaded" : "No model loaded"}
              </span>
            </div>
            {llmStatus?.model_id && (
              <div className="text-brain-text/40 text-[10px] truncate">
                {llmStatus.model_id}
              </div>
            )}
            {llmStatus?.model_info && (
              <div className="text-brain-text/30 text-[10px]">
                {llmStatus.model_info}
              </div>
            )}
            <div className="flex gap-2 mt-2">
              <input
                type="text"
                value={modelInput}
                onChange={(e) => setModelInput(e.target.value)}
                placeholder="Model ID or GGUF path"
                className="flex-1 bg-brain-bg text-white text-[11px] px-2 py-1.5 rounded-lg border border-brain-border outline-none focus:border-brain-accent/50 min-w-0"
              />
              {llmStatus?.model_loaded ? (
                <button
                  onClick={() => unloadModel()}
                  className="px-2.5 py-1.5 bg-brain-error/20 text-brain-error text-[11px] rounded-lg hover:bg-brain-error/30 transition-colors flex-shrink-0"
                >
                  Unload
                </button>
              ) : (
                <button
                  onClick={() => {
                    if (modelInput.trim()) loadModel(modelInput.trim());
                  }}
                  disabled={!modelInput.trim()}
                  className="px-2.5 py-1.5 bg-brain-accent/20 text-brain-accent text-[11px] rounded-lg hover:bg-brain-accent/30 transition-colors disabled:opacity-30 flex-shrink-0"
                >
                  Load
                </button>
              )}
            </div>
          </div>
        </Panel>

        {/* Indexer Panel */}
        {status && (
          <Panel title="File Indexer">
            <div className="grid grid-cols-2 gap-2">
              <Stat label="Files" value={status.indexed_files} />
              <Stat label="Chunks" value={status.indexed_chunks} />
              <Stat label="Emails" value={status.indexed_emails} />
              <Stat label="AI" value={status.ai_available ? "Connected" : "Offline"} />
            </div>
          </Panel>
        )}

        {/* Embeddings Panel */}
        {status && (
          <Panel title="Embeddings">
            <div className="flex items-center gap-2">
              <Dot ok={true} />
              <span className="text-brain-text/60 text-xs">{status.embedding_provider}</span>
            </div>
            <div className="mt-2">
              <Stat label="AI Provider" value={status.ai_provider} />
            </div>
          </Panel>
        )}

        {/* Knowledge Graph Panel */}
        <Panel title="Knowledge Graph">
          {graphStats ? (
            <div className="space-y-2">
              <div className="grid grid-cols-3 gap-2">
                <Stat label="Nodes" value={graphStats.node_count} />
                <Stat label="Edges" value={graphStats.edge_count} />
                <Stat label="Clusters" value={graphStats.hyperedge_count} />
              </div>
              <div className="border-t border-brain-border/50 pt-2 mt-2">
                <div className="flex items-center gap-1.5">
                  <Dot ok={graphStats.node_count > 0} />
                  <span className="text-brain-text/40 text-[10px]">
                    {graphStats.node_count > 0 ? "Graph active" : "No data yet"}
                  </span>
                </div>
              </div>
            </div>
          ) : (
            <div className="text-brain-text/30 text-xs">Loading...</div>
          )}
        </Panel>

        {/* GNN Re-ranker Panel */}
        <Panel title="GNN Re-ranker">
          <div className="space-y-2">
            <div className="flex items-center gap-2">
              <Dot ok={true} />
              <span className="text-brain-text/60 text-xs">Attention layer active</span>
            </div>
            <div className="text-brain-text/30 text-[10px] space-y-0.5">
              <div>Dimensions: 384 &middot; Heads: 4</div>
              <div>Blend: 0.7 vector + 0.3 GNN</div>
              <div>Weights persisted on flush</div>
            </div>
          </div>
        </Panel>

        {/* Tensor Compression Panel */}
        <Panel title="Tensor Compression">
          {compressionStats ? (
            <div className="space-y-2">
              <div className="grid grid-cols-2 gap-2">
                <Stat label="Hot" value={compressionStats.hot_count} />
                <Stat label="Warm" value={compressionStats.warm_count} />
                <Stat label="Cold" value={compressionStats.cold_count} />
                <Stat label="Savings" value={`${compressionStats.estimated_savings_pct}%`} />
              </div>
              <div className="border-t border-brain-border/50 pt-2 mt-2 text-brain-text/30 text-[10px]">
                {compressionStats.total_memories} memories scanned
              </div>
            </div>
          ) : (
            <div className="text-brain-text/30 text-xs">Loading...</div>
          )}
        </Panel>
      </div>

      {/* Health Verification */}
      <div className="bg-brain-surface rounded-xl p-4 border border-brain-border">
        <div className="flex items-center justify-between mb-3">
          <h3 className="text-white text-xs font-medium">Health Verification</h3>
          <button
            onClick={verifyAll}
            disabled={verifying}
            className="px-3 py-1.5 bg-brain-accent text-white text-[11px] rounded-lg hover:bg-brain-accent/80 transition-colors disabled:opacity-50"
          >
            {verifying ? "Verifying..." : "Run Verification"}
          </button>
        </div>
        {verifyResult ? (
          <div className="grid grid-cols-3 gap-2">
            {Object.entries(verifyResult).map(([name, data]) => (
              <div key={name} className="flex items-center gap-2 bg-brain-bg/50 rounded-lg px-3 py-2">
                <Dot ok={!!(data as { ok: boolean }).ok} />
                <span className="text-brain-text/60 text-[11px] capitalize">{name}</span>
              </div>
            ))}
          </div>
        ) : (
          <div className="text-brain-text/30 text-xs">
            Click "Run Verification" to check all subsystems
          </div>
        )}
      </div>
    </div>
  );
}
