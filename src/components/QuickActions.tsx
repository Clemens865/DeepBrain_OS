import { useCallback, useEffect, useState } from "react";
import { useAppStore } from "../store/appStore";

interface QuickActionsProps {
  onSettings: () => void;
  onBrowse?: () => void;
}

export default function QuickActions({ onSettings, onBrowse }: QuickActionsProps) {
  const { remember, loadStatus, runWorkflow, clipboardHistory, loadClipboardHistory, status, enterBrowseMode } = useAppStore();
  const [showClipboard, setShowClipboard] = useState(false);
  const [toast, setToast] = useState<string | null>(null);
  const [toastType, setToastType] = useState<"info" | "success" | "error">("info");

  useEffect(() => {
    if (showClipboard) loadClipboardHistory();
  }, [showClipboard, loadClipboardHistory]);

  const showToast = useCallback((msg: string, type: "info" | "success" | "error" = "info") => {
    setToast(msg);
    setToastType(type);
    setTimeout(() => setToast(null), 4000);
  }, []);

  const handleDigest = useCallback(async () => {
    showToast("Running digest...", "info");
    try {
      const result = await runWorkflow("digest");
      showToast(result.message, result.success ? "success" : "error");
    } catch {
      showToast("Digest failed", "error");
    }
  }, [runWorkflow, showToast]);

  const handleStatus = useCallback(async () => {
    await loadStatus();
    const s = useAppStore.getState().status;
    if (s) {
      showToast(`${s.memory_count} memories, ${s.thought_count} thoughts, ${s.indexed_files} files — ${s.learning_trend}`, "success");
    } else {
      showToast("Status refreshed", "info");
    }
  }, [loadStatus, showToast]);

  const handleRememberClip = useCallback(
    async (content: string) => {
      await remember(content, "working", 0.6);
      loadStatus();
    },
    [remember, loadStatus],
  );

  if (showClipboard) {
    return (
      <div className="px-4 py-2 border-t border-brain-border flex-1 overflow-hidden flex flex-col">
        <div className="flex items-center justify-between mb-2">
          <span className="text-brain-text/50 text-xs font-medium">Recent Clipboard</span>
          <button
            onClick={() => setShowClipboard(false)}
            className="text-brain-text/30 hover:text-brain-text text-xs transition-colors"
          >
            Close
          </button>
        </div>
        <div className="flex-1 overflow-y-auto space-y-1">
          {clipboardHistory.length === 0 ? (
            <p className="text-brain-text/30 text-xs py-4 text-center">No clipboard entries yet. Copy some text!</p>
          ) : (
            clipboardHistory.map((entry, i) => (
              <div
                key={i}
                className="flex items-start gap-2 p-2 rounded-lg bg-brain-surface/50 hover:bg-brain-surface border border-transparent hover:border-brain-border transition-colors group"
              >
                <p className="flex-1 text-brain-text text-xs line-clamp-2">{entry.content}</p>
                <button
                  onClick={() => handleRememberClip(entry.content)}
                  className="opacity-0 group-hover:opacity-100 text-brain-accent text-[10px] px-2 py-0.5 rounded bg-brain-accent/10 hover:bg-brain-accent/20 transition-all whitespace-nowrap"
                >
                  Remember
                </button>
              </div>
            ))
          )}
        </div>
      </div>
    );
  }

  return (
    <div className="px-4 py-3 border-t border-brain-border">
      {toast && (
        <div
          className={`mb-2 px-3 py-2 rounded-lg text-xs animate-fade-in ${
            toastType === "success"
              ? "bg-brain-success/10 text-brain-success border border-brain-success/20"
              : toastType === "error"
              ? "bg-red-500/10 text-red-400 border border-red-500/20"
              : "bg-brain-accent/10 text-brain-accent border border-brain-accent/20"
          }`}
        >
          {toast}
        </div>
      )}
      <div className="flex gap-2">
        <ActionButton
          icon={<svg className="w-3.5 h-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M4 6a2 2 0 012-2h2a2 2 0 012 2v2a2 2 0 01-2 2H6a2 2 0 01-2-2V6zM14 6a2 2 0 012-2h2a2 2 0 012 2v2a2 2 0 01-2 2h-2a2 2 0 01-2-2V6zM4 16a2 2 0 012-2h2a2 2 0 012 2v2a2 2 0 01-2 2H6a2 2 0 01-2-2v-2zM14 16a2 2 0 012-2h2a2 2 0 012 2v2a2 2 0 01-2 2h-2a2 2 0 01-2-2v-2z" /></svg>}
          label="Browse"
          onClick={() => (onBrowse ? onBrowse() : enterBrowseMode())}
        />
        <ActionButton
          icon={<svg className="w-3.5 h-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5H7a2 2 0 00-2 2v12a2 2 0 002 2h10a2 2 0 002-2V7a2 2 0 00-2-2h-2M9 5a2 2 0 002 2h2a2 2 0 002-2M9 5a2 2 0 012-2h2a2 2 0 012 2" /></svg>}
          label="Clipboard"
          onClick={() => setShowClipboard(true)}
        />
        <ActionButton
          icon={<svg className="w-3.5 h-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 10V3L4 14h7v7l9-11h-7z" /></svg>}
          label="Digest"
          onClick={handleDigest}
        />
        <ActionButton
          icon={<svg className="w-3.5 h-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" /></svg>}
          label="Status"
          onClick={handleStatus}
        />
        <ActionButton
          icon={<svg className="w-3.5 h-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M10.325 4.317c.426-1.756 2.924-1.756 3.35 0a1.724 1.724 0 002.573 1.066c1.543-.94 3.31.826 2.37 2.37a1.724 1.724 0 001.066 2.573c1.756.426 1.756 2.924 0 3.35a1.724 1.724 0 00-1.066 2.573c.94 1.543-.826 3.31-2.37 2.37a1.724 1.724 0 00-2.573 1.066c-.426 1.756-2.924 1.756-3.35 0a1.724 1.724 0 00-2.573-1.066c-1.543.94-3.31-.826-2.37-2.37a1.724 1.724 0 00-1.066-2.573c-1.756-.426-1.756-2.924 0-3.35a1.724 1.724 0 001.066-2.573c-.94-1.543.826-3.31 2.37-2.37.996.608 2.296.07 2.572-1.065z" /><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 12a3 3 0 11-6 0 3 3 0 016 0z" /></svg>}
          label="Settings"
          onClick={onSettings}
        />
      </div>
    </div>
  );
}

function ActionButton({
  icon,
  label,
  onClick,
}: {
  icon: React.ReactNode;
  label: string;
  onClick: () => void;
}) {
  return (
    <button
      onClick={onClick}
      className="flex items-center gap-1.5 px-3 py-2 bg-brain-surface text-brain-text text-xs rounded-lg border border-brain-border hover:border-brain-accent/30 hover:text-white transition-colors"
    >
      {icon}
      {label}
    </button>
  );
}
