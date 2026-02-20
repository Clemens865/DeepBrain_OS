import { useCallback, useEffect, useRef, useMemo } from "react";
import { useAppStore, type BrowseItem, type SortField, type SortDirection } from "../store/appStore";

function getItemId(item: BrowseItem): string {
  switch (item.kind) {
    case "memory": return item.data.id;
    case "file": return item.data.path;
    case "thought": return item.data.id;
    case "clipboard": return String(item.data.timestamp);
    case "spotlight": return item.data.path;
    case "email": return String(item.data.message_id);
  }
}

function timeAgo(ts: number): string {
  const now = Date.now();
  // timestamps from Rust may be in seconds or milliseconds
  const msTs = ts > 1e12 ? ts : ts * 1000;
  const diff = now - msTs;
  if (diff < 60_000) return "just now";
  if (diff < 3_600_000) return `${Math.floor(diff / 60_000)}m ago`;
  if (diff < 86_400_000) return `${Math.floor(diff / 3_600_000)}h ago`;
  return `${Math.floor(diff / 86_400_000)}d ago`;
}

function sortItems(items: BrowseItem[], field: SortField, direction: SortDirection): BrowseItem[] {
  const sorted = [...items];
  const dir = direction === "asc" ? 1 : -1;

  sorted.sort((a, b) => {
    let va: number | string = 0;
    let vb: number | string = 0;

    switch (field) {
      case "timestamp":
        va = a.kind === "memory" ? a.data.timestamp : a.kind === "file" ? a.data.modified : a.kind === "thought" ? a.data.timestamp : a.kind === "clipboard" ? a.data.timestamp : a.kind === "spotlight" ? Number(a.data.modified || 0) : a.kind === "email" ? a.data.message_id : 0;
        vb = b.kind === "memory" ? b.data.timestamp : b.kind === "file" ? b.data.modified : b.kind === "thought" ? b.data.timestamp : b.kind === "clipboard" ? b.data.timestamp : b.kind === "spotlight" ? Number(b.data.modified || 0) : b.kind === "email" ? b.data.message_id : 0;
        break;
      case "importance":
        va = a.kind === "memory" ? a.data.importance : 0;
        vb = b.kind === "memory" ? b.data.importance : 0;
        break;
      case "access_count":
        va = a.kind === "memory" ? a.data.access_count : 0;
        vb = b.kind === "memory" ? b.data.access_count : 0;
        break;
      case "type":
        va = a.kind === "memory" ? a.data.memory_type : a.kind === "thought" ? a.data.thought_type : "";
        vb = b.kind === "memory" ? b.data.memory_type : b.kind === "thought" ? b.data.thought_type : "";
        if (typeof va === "string") return va.localeCompare(vb as string) * dir;
        break;
      case "name":
        va = a.kind === "file" ? a.data.name : "";
        vb = b.kind === "file" ? b.data.name : "";
        if (typeof va === "string") return va.localeCompare(vb as string) * dir;
        break;
    }
    return ((va as number) - (vb as number)) * dir;
  });

  return sorted;
}

export default function ContentList() {
  const {
    browseItems,
    browseLoading,
    selectedItemId,
    selectBrowseItem,
    browseFilterText,
    browseCategory,
    sortField,
    sortDirection,
  } = useAppStore();
  const listRef = useRef<HTMLDivElement>(null);

  // Filter then sort
  const filtered = useMemo(() => {
    let items = browseItems;
    if (browseFilterText) {
      const text = browseFilterText.toLowerCase();
      items = items.filter((item) => {
        if (item.kind === "memory") return item.data.content.toLowerCase().includes(text);
        if (item.kind === "file") return item.data.name.toLowerCase().includes(text) || item.data.path.toLowerCase().includes(text);
        if (item.kind === "thought") return item.data.content.toLowerCase().includes(text);
        if (item.kind === "clipboard") return item.data.content.toLowerCase().includes(text);
        if (item.kind === "spotlight") return item.data.name.toLowerCase().includes(text) || item.data.path.toLowerCase().includes(text);
        if (item.kind === "email") return item.data.subject.toLowerCase().includes(text) || item.data.sender.toLowerCase().includes(text) || item.data.preview.toLowerCase().includes(text);
        return true;
      });
    }
    return sortItems(items, sortField, sortDirection);
  }, [browseItems, browseFilterText, sortField, sortDirection]);

  // Keyboard navigation
  const handleKeyDown = useCallback(
    (e: KeyboardEvent) => {
      if (filtered.length === 0) return;
      const currentIdx = filtered.findIndex((i) => getItemId(i) === selectedItemId);

      if (e.key === "ArrowDown") {
        e.preventDefault();
        const next = Math.min(currentIdx + 1, filtered.length - 1);
        selectBrowseItem(getItemId(filtered[next]));
      } else if (e.key === "ArrowUp") {
        e.preventDefault();
        const prev = Math.max(currentIdx - 1, 0);
        selectBrowseItem(getItemId(filtered[prev]));
      }
    },
    [filtered, selectedItemId, selectBrowseItem],
  );

  useEffect(() => {
    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, [handleKeyDown]);

  if (browseCategory === "home" || browseCategory === "status") {
    return null; // These categories render their own full-view content in BrowseView
  }

  if (browseLoading) {
    return (
      <div className="flex-1 flex items-center justify-center text-brain-text/30 text-xs">
        Loading...
      </div>
    );
  }

  if (filtered.length === 0) {
    return (
      <div className="flex-1 flex items-center justify-center text-brain-text/30 text-xs p-4">
        {browseFilterText ? "No matching items" : "No items in this category"}
      </div>
    );
  }

  return (
    <div ref={listRef} className="flex-1 min-w-0 overflow-y-auto">
      {filtered.map((item) => {
        const id = getItemId(item);
        const isSelected = selectedItemId === id;
        return (
          <button
            key={id}
            onClick={() => selectBrowseItem(id)}
            className={`w-full text-left px-3 py-2 border-b border-brain-border/50 transition-colors ${
              isSelected
                ? "bg-brain-accent/15 border-l-2 border-l-brain-accent"
                : "hover:bg-brain-surface/50 border-l-2 border-l-transparent"
            }`}
          >
            {item.kind === "memory" && <MemoryRow data={item.data} />}
            {item.kind === "file" && <FileRow data={item.data} />}
            {item.kind === "thought" && <ThoughtRow data={item.data} />}
            {item.kind === "clipboard" && <ClipboardRow data={item.data} />}
            {item.kind === "spotlight" && <SpotlightRow data={item.data} />}
            {item.kind === "email" && <EmailRow data={item.data} />}
          </button>
        );
      })}
    </div>
  );
}

function MemoryRow({ data }: { data: any }) {
  const mem = data;
  return (
    <div className="space-y-1">
      <p className="text-brain-text text-xs line-clamp-2">{mem.content}</p>
      <div className="flex items-center gap-2 text-[10px] text-brain-text/40">
        <span className="px-1.5 py-0.5 rounded bg-brain-accent/10 text-brain-accent">{mem.memory_type}</span>
        <span>imp: {(mem.importance * 100).toFixed(0)}%</span>
        <span className="ml-auto">{timeAgo(mem.timestamp)}</span>
      </div>
    </div>
  );
}

function FileRow({ data }: { data: any }) {
  return (
    <div className="space-y-1">
      <div className="flex items-center gap-2">
        <svg className="w-3.5 h-3.5 text-brain-text/40 flex-shrink-0" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M7 21h10a2 2 0 002-2V9.414a1 1 0 00-.293-.707l-5.414-5.414A1 1 0 0012.586 3H7a2 2 0 00-2 2v14a2 2 0 002 2z" />
        </svg>
        <span className="text-brain-text text-xs truncate">{data.name}</span>
        <span className="text-[10px] px-1 py-0.5 rounded bg-brain-surface text-brain-text/40">.{data.ext}</span>
      </div>
      <div className="flex items-center gap-2 text-[10px] text-brain-text/30 pl-5">
        <span className="truncate">{data.path}</span>
        <span className="ml-auto flex-shrink-0">{data.chunk_count} chunks</span>
      </div>
    </div>
  );
}

function ThoughtRow({ data }: { data: any }) {
  return (
    <div className="space-y-1">
      <p className="text-brain-text text-xs line-clamp-2">{data.content}</p>
      <div className="flex items-center gap-2 text-[10px] text-brain-text/40">
        <span className="px-1.5 py-0.5 rounded bg-purple-500/10 text-purple-400">{data.thought_type}</span>
        <span>conf: {(data.confidence * 100).toFixed(0)}%</span>
        <span className="ml-auto">{timeAgo(data.timestamp)}</span>
      </div>
    </div>
  );
}

function ClipboardRow({ data }: { data: any }) {
  return (
    <div className="space-y-1">
      <p className="text-brain-text text-xs line-clamp-2 font-mono">{data.content}</p>
      <div className="text-[10px] text-brain-text/30">{timeAgo(data.timestamp)}</div>
    </div>
  );
}

const SPOTLIGHT_KIND_COLORS: Record<string, string> = {
  email: "bg-blue-500/10 text-blue-400",
  document: "bg-green-500/10 text-green-400",
  pdf: "bg-red-500/10 text-red-400",
  code: "bg-brain-accent/10 text-brain-accent",
  image: "bg-pink-500/10 text-pink-400",
  spreadsheet: "bg-emerald-500/10 text-emerald-400",
  presentation: "bg-orange-500/10 text-orange-400",
  audio: "bg-violet-500/10 text-violet-400",
  video: "bg-cyan-500/10 text-cyan-400",
  file: "bg-brain-surface text-brain-text/40",
};

function EmailRow({ data }: { data: any }) {
  return (
    <div className="space-y-1">
      <div className="flex items-center gap-2">
        <svg className="w-3.5 h-3.5 text-blue-400 flex-shrink-0" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M3 8l7.89 5.26a2 2 0 002.22 0L21 8M5 19h14a2 2 0 002-2V7a2 2 0 00-2-2H5a2 2 0 00-2 2v10a2 2 0 002 2z" />
        </svg>
        <span className="text-brain-text text-xs truncate">{data.subject}</span>
      </div>
      <div className="flex items-center gap-2 text-[10px] text-brain-text/40 pl-5">
        <span className="truncate">{data.sender}</span>
        <span className="ml-auto flex-shrink-0 text-brain-text/30">{data.date}</span>
      </div>
      {data.preview && (
        <p className="text-brain-text/30 text-[10px] pl-5 line-clamp-1">{data.preview}</p>
      )}
    </div>
  );
}

function SpotlightRow({ data }: { data: any }) {
  const kindColor = SPOTLIGHT_KIND_COLORS[data.kind] || SPOTLIGHT_KIND_COLORS.file;
  return (
    <div className="space-y-1">
      <div className="flex items-center gap-2">
        <svg className="w-3.5 h-3.5 text-brain-text/40 flex-shrink-0" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" />
        </svg>
        <span className="text-brain-text text-xs truncate">{data.name}</span>
        <span className={`text-[10px] px-1 py-0.5 rounded ${kindColor}`}>{data.kind}</span>
      </div>
      <div className="flex items-center gap-2 text-[10px] text-brain-text/30 pl-5">
        <span className="truncate">{data.path}</span>
        {data.modified && <span className="ml-auto flex-shrink-0">{timeAgo(Number(data.modified))}</span>}
      </div>
    </div>
  );
}
