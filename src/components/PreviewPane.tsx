import { useState } from "react";
import { useAppStore } from "../store/appStore";

function timeAgo(ts: number): string {
  const now = Date.now();
  const msTs = ts > 1e12 ? ts : ts * 1000;
  const diff = now - msTs;
  if (diff < 60_000) return "just now";
  if (diff < 3_600_000) return `${Math.floor(diff / 60_000)}m ago`;
  if (diff < 86_400_000) return `${Math.floor(diff / 3_600_000)}h ago`;
  return `${Math.floor(diff / 86_400_000)}d ago`;
}

function formatDate(ts: number): string {
  const msTs = ts > 1e12 ? ts : ts * 1000;
  return new Date(msTs).toLocaleString();
}

export default function PreviewPane() {
  const { selectedItemDetail, deleteBrowseItem } = useAppStore();
  const [confirmDelete, setConfirmDelete] = useState(false);

  if (!selectedItemDetail) {
    return (
      <div className="w-72 flex-shrink-0 border-l border-brain-border flex items-center justify-center text-brain-text/20 text-xs p-4 text-center">
        Select an item to preview
      </div>
    );
  }

  const item = selectedItemDetail;

  const handleCopy = () => {
    let text = "";
    if (item.kind === "memory") text = item.data.content;
    else if (item.kind === "file") text = item.data.path;
    else if (item.kind === "thought") text = item.data.content;
    else if (item.kind === "clipboard") text = item.data.content;
    else if (item.kind === "email") text = `${item.data.subject}\nFrom: ${item.data.sender}\n${item.data.preview || ""}`;
    navigator.clipboard.writeText(text);
  };

  const handleDelete = async () => {
    if (!confirmDelete) {
      setConfirmDelete(true);
      setTimeout(() => setConfirmDelete(false), 3000);
      return;
    }
    if (item.kind === "memory") {
      await deleteBrowseItem(item.data.id);
    }
    setConfirmDelete(false);
  };

  return (
    <div className="w-72 flex-shrink-0 border-l border-brain-border flex flex-col bg-brain-bg/30">
      {/* Content */}
      <div className="flex-1 overflow-y-auto p-3 space-y-3">
        {item.kind === "memory" && <MemoryPreview data={item.data} />}
        {item.kind === "file" && <FilePreview data={item.data} />}
        {item.kind === "thought" && <ThoughtPreview data={item.data} />}
        {item.kind === "clipboard" && <ClipboardPreview data={item.data} />}
        {item.kind === "email" && <EmailPreview data={item.data} />}
      </div>

      {/* Actions */}
      <div className="flex gap-2 p-3 border-t border-brain-border">
        <button
          onClick={handleCopy}
          className="flex-1 flex items-center justify-center gap-1.5 px-2 py-1.5 text-xs rounded bg-brain-surface text-brain-text border border-brain-border hover:border-brain-accent/30 transition-colors"
        >
          <svg className="w-3 h-3" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8 16H6a2 2 0 01-2-2V6a2 2 0 012-2h8a2 2 0 012 2v2m-6 12h8a2 2 0 002-2v-8a2 2 0 00-2-2h-8a2 2 0 00-2 2v8a2 2 0 002 2z" />
          </svg>
          Copy
        </button>
        {item.kind === "memory" && (
          <button
            onClick={handleDelete}
            className={`flex-1 flex items-center justify-center gap-1.5 px-2 py-1.5 text-xs rounded border transition-colors ${
              confirmDelete
                ? "bg-red-500/20 text-red-400 border-red-500/30"
                : "bg-brain-surface text-brain-text/60 border-brain-border hover:border-red-500/30 hover:text-red-400"
            }`}
          >
            <svg className="w-3 h-3" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16" />
            </svg>
            {confirmDelete ? "Confirm?" : "Delete"}
          </button>
        )}
      </div>
    </div>
  );
}

function MemoryPreview({ data }: { data: any }) {
  return (
    <>
      <p className="text-brain-text text-xs leading-relaxed whitespace-pre-wrap">{data.content}</p>
      <div className="space-y-2">
        <MetaRow label="Type" value={data.memory_type} accent />
        <MetaRow label="Importance" value={`${(data.importance * 100).toFixed(0)}%`} />
        <ImportanceBar value={data.importance} />
        <MetaRow label="Access count" value={data.access_count} />
        <MetaRow label="Connections" value={data.connection_count} />
        <MetaRow label="Created" value={timeAgo(data.timestamp)} />
        <MetaRow label="Date" value={formatDate(data.timestamp)} />
      </div>
    </>
  );
}

function FilePreview({ data }: { data: any }) {
  return (
    <>
      <div className="flex items-center gap-2">
        <svg className="w-5 h-5 text-brain-accent" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M7 21h10a2 2 0 002-2V9.414a1 1 0 00-.293-.707l-5.414-5.414A1 1 0 0012.586 3H7a2 2 0 00-2 2v14a2 2 0 002 2z" />
        </svg>
        <span className="text-white text-sm font-medium truncate">{data.name}</span>
      </div>
      <div className="space-y-2">
        <MetaRow label="Path" value={data.path} mono />
        <MetaRow label="Extension" value={`.${data.ext}`} accent />
        <MetaRow label="Chunks" value={data.chunk_count} />
        <MetaRow label="Modified" value={timeAgo(data.modified)} />
        <MetaRow label="Date" value={formatDate(data.modified)} />
      </div>
    </>
  );
}

function ThoughtPreview({ data }: { data: any }) {
  return (
    <>
      <p className="text-brain-text text-xs leading-relaxed whitespace-pre-wrap">{data.content}</p>
      <div className="space-y-2">
        <MetaRow label="Type" value={data.thought_type} accent />
        <MetaRow label="Confidence" value={`${(data.confidence * 100).toFixed(0)}%`} />
        <MetaRow label="Novelty" value={`${(data.novelty * 100).toFixed(0)}%`} />
        <MetaRow label="Utility" value={`${(data.utility * 100).toFixed(0)}%`} />
        <MetaRow label="Created" value={timeAgo(data.timestamp)} />
      </div>
    </>
  );
}

function ClipboardPreview({ data }: { data: any }) {
  return (
    <>
      <p className="text-brain-text text-xs leading-relaxed whitespace-pre-wrap font-mono bg-brain-surface/50 rounded p-2">
        {data.content}
      </p>
      <div className="space-y-2">
        <MetaRow label="Captured" value={timeAgo(data.timestamp)} />
        <MetaRow label="Length" value={`${data.content.length} chars`} />
      </div>
    </>
  );
}

function EmailPreview({ data }: { data: any }) {
  return (
    <>
      <div className="flex items-center gap-2">
        <svg className="w-5 h-5 text-blue-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M3 8l7.89 5.26a2 2 0 002.22 0L21 8M5 19h14a2 2 0 002-2V7a2 2 0 00-2-2H5a2 2 0 00-2 2v10a2 2 0 002 2z" />
        </svg>
        <span className="text-white text-sm font-medium truncate">{data.subject}</span>
      </div>
      <div className="space-y-2 mt-2">
        <MetaRow label="From" value={data.sender} />
        <MetaRow label="Date" value={data.date} />
        {data.account && <MetaRow label="Account" value={data.account} />}
        <MetaRow label="Mailbox" value={data.mailbox} accent />
      </div>
      {data.preview && (
        <div className="mt-3">
          <div className="text-brain-text/40 text-[10px] uppercase tracking-wider mb-1">Preview</div>
          <p className="text-brain-text/70 text-xs leading-relaxed whitespace-pre-wrap bg-brain-surface/50 rounded p-2">
            {data.preview}
          </p>
        </div>
      )}
    </>
  );
}

function MetaRow({ label, value, accent, mono }: { label: string; value: any; accent?: boolean; mono?: boolean }) {
  return (
    <div className="flex items-start justify-between text-[11px]">
      <span className="text-brain-text/40">{label}</span>
      <span
        className={`text-right max-w-[60%] truncate ${
          accent ? "text-brain-accent" : mono ? "text-brain-text/60 font-mono text-[10px]" : "text-brain-text/70"
        }`}
        title={String(value)}
      >
        {value}
      </span>
    </div>
  );
}

function ImportanceBar({ value }: { value: number }) {
  return (
    <div className="w-full bg-brain-surface rounded-full h-1.5">
      <div
        className="bg-brain-accent rounded-full h-1.5 transition-all"
        style={{ width: `${Math.min(value * 100, 100)}%` }}
      />
    </div>
  );
}
