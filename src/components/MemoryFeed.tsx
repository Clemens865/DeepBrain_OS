import { useEffect } from "react";
import { useAppStore } from "../store/appStore";

export default function MemoryFeed() {
  const { status, loadStatus } = useAppStore();

  useEffect(() => {
    loadStatus();
    const interval = setInterval(loadStatus, 10000);
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
