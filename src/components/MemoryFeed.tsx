import { useEffect, useState } from "react";
import { useAppStore, type BrainwireStats } from "../store/appStore";
import { invoke } from "../services/backend";

export default function MemoryFeed() {
  const { status, loadStatus } = useAppStore();
  const [bwStats, setBwStats] = useState<BrainwireStats | null>(null);

  useEffect(() => {
    loadStatus();
    invoke<BrainwireStats>("brainwire_status").then(setBwStats).catch(() => {});
    const interval = setInterval(() => {
      loadStatus();
      invoke<BrainwireStats>("brainwire_status").then(setBwStats).catch(() => {});
    }, 10000);
    return () => clearInterval(interval);
  }, [loadStatus]);

  return (
    <div className="flex-1 px-4 py-3 overflow-y-auto">
      {/* Status card */}
      <div className="bg-brain-surface rounded-xl p-4 border border-brain-border mb-3">
        <div className="flex items-center gap-2 mb-3">
          <div
            className={`w-2 h-2 rounded-full ${
              status?.status === "healthy" ? "bg-brain-success" : "bg-brain-warning"
            }`}
          />
          <span className="text-white text-sm font-medium">DeepBrain</span>
          <span className="text-brain-text/50 text-xs ml-auto">
            {status?.learning_trend || "initializing"}
          </span>
        </div>

        <div className="grid grid-cols-4 gap-3">
          <StatCard label="Memories" value={status?.memory_count ?? 0} />
          <StatCard label="Thoughts" value={status?.thought_count ?? 0} />
          <StatCard label="Files" value={status?.indexed_files ?? 0} />
          <StatCard
            label="Uptime"
            value={formatUptime(status?.uptime_ms ?? 0)}
          />
        </div>
      </div>

      {/* Brainwire status row */}
      {bwStats && (
        <div className="bg-brain-surface rounded-xl p-3 border border-brain-border mb-3">
          <div className="flex items-center gap-2">
            <div className={`w-2 h-2 rounded-full ${bwStats.total_memories > 0 ? "bg-brain-success" : "bg-brain-text/20"}`} />
            <span className="text-white text-xs font-medium">Brainwire</span>
            <span className="text-brain-text/40 text-[10px] ml-auto">
              {bwStats.total_memories} memories &middot; {bwStats.concept_count} concepts &middot; {bwStats.working_memory_items} in focus
            </span>
          </div>
        </div>
      )}

      {/* Tips */}
      <div className="text-brain-text/40 text-xs space-y-1.5 px-1">
        <p>Type to search your memories and knowledge base</p>
        <p>Use Quick Actions below to store new memories</p>
      </div>
    </div>
  );
}

function StatCard({ label, value }: { label: string; value: number | string }) {
  return (
    <div className="text-center">
      <div className="text-white text-lg font-semibold">{value}</div>
      <div className="text-brain-text/50 text-[10px] uppercase tracking-wider">
        {label}
      </div>
    </div>
  );
}

function formatUptime(ms: number): string {
  if (ms < 60000) return `${Math.floor(ms / 1000)}s`;
  if (ms < 3600000) return `${Math.floor(ms / 60000)}m`;
  return `${Math.floor(ms / 3600000)}h`;
}
