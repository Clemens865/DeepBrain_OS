import { useEffect, useState, useCallback } from "react";
import { invoke } from "../services/backend";

interface TopicItem {
  slug: string;
  name: string;
  size_bytes: number;
  modified_ts: number;
  memory_count: number;
}

interface TopicStats {
  topics_written: number;
  memories_exported: number;
  consolidation_cycles: number;
  avg_salience: number;
}

interface TopicsResponse {
  topics: TopicItem[];
  stats: TopicStats | null;
  last_exported: number | null;
}

interface TopicContent {
  slug: string;
  content: string;
}

function formatBytes(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

function formatDate(ts: number): string {
  if (!ts) return "Never";
  return new Date(ts * 1000).toLocaleDateString(undefined, {
    month: "short",
    day: "numeric",
    hour: "2-digit",
    minute: "2-digit",
  });
}

function StatCard({ label, value }: { label: string; value: string | number }) {
  return (
    <div className="bg-brain-surface rounded-lg p-3 border border-brain-border">
      <div className="text-brain-text/40 text-[10px] uppercase tracking-wider">{label}</div>
      <div className="text-white text-sm font-semibold mt-1">{value}</div>
    </div>
  );
}

function renderMarkdown(md: string): React.ReactNode[] {
  const lines = md.split("\n");
  const nodes: React.ReactNode[] = [];
  let listItems: string[] = [];

  const flushList = () => {
    if (listItems.length > 0) {
      nodes.push(
        <ul key={`ul-${nodes.length}`} className="list-disc list-inside space-y-0.5 text-brain-text/70 text-xs ml-2 mb-2">
          {listItems.map((item, i) => (
            <li key={i}>{inlineFormat(item)}</li>
          ))}
        </ul>
      );
      listItems = [];
    }
  };

  const inlineFormat = (text: string): React.ReactNode => {
    // Bold + italic
    const parts: React.ReactNode[] = [];
    let remaining = text;
    let key = 0;

    while (remaining.length > 0) {
      const boldMatch = remaining.match(/\*\*(.+?)\*\*/);
      const italicMatch = remaining.match(/\*(.+?)\*/);

      const match = boldMatch && italicMatch
        ? (boldMatch.index! <= italicMatch.index! ? boldMatch : italicMatch)
        : boldMatch || italicMatch;

      if (!match || match.index === undefined) {
        parts.push(<span key={key++}>{remaining}</span>);
        break;
      }

      if (match.index > 0) {
        parts.push(<span key={key++}>{remaining.slice(0, match.index)}</span>);
      }

      if (match[0].startsWith("**")) {
        parts.push(<strong key={key++} className="text-white font-semibold">{match[1]}</strong>);
      } else {
        parts.push(<em key={key++} className="italic text-brain-text/80">{match[1]}</em>);
      }

      remaining = remaining.slice(match.index + match[0].length);
    }

    return parts.length === 1 ? parts[0] : <>{parts}</>;
  };

  for (let i = 0; i < lines.length; i++) {
    const line = lines[i];

    if (line.startsWith("# ")) {
      flushList();
      nodes.push(
        <h1 key={i} className="text-white text-lg font-bold mb-3 mt-1">{line.slice(2)}</h1>
      );
    } else if (line.startsWith("## ")) {
      flushList();
      nodes.push(
        <h2 key={i} className="text-white text-sm font-semibold mb-2 mt-4 border-b border-brain-border pb-1">{line.slice(3)}</h2>
      );
    } else if (line.startsWith("### ")) {
      flushList();
      nodes.push(
        <h3 key={i} className="text-brain-accent text-xs font-medium mb-1 mt-3">{line.slice(4)}</h3>
      );
    } else if (line.startsWith("- ") || line.startsWith("* ")) {
      listItems.push(line.slice(2));
    } else if (line.startsWith("---")) {
      flushList();
      nodes.push(<hr key={i} className="border-brain-border my-3" />);
    } else if (line.trim() === "") {
      flushList();
    } else {
      flushList();
      nodes.push(
        <p key={i} className="text-brain-text/70 text-xs mb-1.5 leading-relaxed">{inlineFormat(line)}</p>
      );
    }
  }

  flushList();
  return nodes;
}

export default function KnowledgeExportView() {
  const [exporting, setExporting] = useState(false);
  const [loading, setLoading] = useState(true);
  const [topics, setTopics] = useState<TopicItem[]>([]);
  const [stats, setStats] = useState<TopicStats | null>(null);
  const [lastExported, setLastExported] = useState<number | null>(null);
  const [selectedSlug, setSelectedSlug] = useState<string | null>(null);
  const [selectedContent, setSelectedContent] = useState<string | null>(null);
  const [contentLoading, setContentLoading] = useState(false);

  const loadTopics = useCallback(async () => {
    setLoading(true);
    try {
      const data = await invoke<TopicsResponse>("brainwire_knowledge_topics");
      setTopics(data.topics);
      setStats(data.stats);
      setLastExported(data.last_exported);
    } catch (e) {
      console.error("Failed to load topics:", e);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    loadTopics();
  }, [loadTopics]);

  const handleExport = async () => {
    setExporting(true);
    try {
      await invoke("brainwire_export_knowledge");
      await loadTopics();
    } catch (e) {
      console.error("Export failed:", e);
    } finally {
      setExporting(false);
    }
  };

  const handleSelectTopic = async (slug: string) => {
    setSelectedSlug(slug);
    setContentLoading(true);
    try {
      const data = await invoke<TopicContent>("brainwire_knowledge_topic", { slug });
      setSelectedContent(data.content);
    } catch (e) {
      console.error("Failed to load topic:", e);
      setSelectedContent(null);
    } finally {
      setContentLoading(false);
    }
  };

  return (
    <div className="flex flex-col h-full overflow-hidden">
      {/* Top bar */}
      <div className="flex items-center justify-between px-4 py-3 border-b border-brain-border">
        <div>
          <h2 className="text-white text-sm font-semibold">Knowledge Export</h2>
          <p className="text-brain-text/40 text-[10px] mt-0.5">
            {lastExported
              ? `Last exported: ${formatDate(lastExported)}`
              : "Not yet exported"}
          </p>
        </div>
        <button
          onClick={handleExport}
          disabled={exporting}
          className="px-3 py-1.5 bg-brain-accent text-white text-xs font-medium rounded-lg hover:bg-brain-accent/80 disabled:opacity-50 disabled:cursor-not-allowed transition-colors flex items-center gap-1.5"
        >
          {exporting ? (
            <>
              <span className="animate-spin text-[10px]">&#9696;</span>
              Exporting...
            </>
          ) : (
            <>
              <svg className="w-3.5 h-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-4l-4 4m0 0l-4-4m4 4V4" />
              </svg>
              Export Now
            </>
          )}
        </button>
      </div>

      {/* Stats row */}
      <div className="grid grid-cols-4 gap-2 px-4 py-3 border-b border-brain-border">
        <StatCard
          label="Topics"
          value={stats?.topics_written ?? topics.length}
        />
        <StatCard
          label="Memories"
          value={stats?.memories_exported ?? 0}
        />
        <StatCard
          label="Cycles"
          value={stats?.consolidation_cycles ?? 0}
        />
        <StatCard
          label="Avg Salience"
          value={stats?.avg_salience != null ? stats.avg_salience.toFixed(2) : "—"}
        />
      </div>

      {/* Split pane: topic list + content */}
      <div className="flex flex-1 min-h-0">
        {/* Topic list */}
        <div className="w-48 flex-shrink-0 border-r border-brain-border overflow-y-auto">
          {loading ? (
            <div className="p-4 text-brain-text/40 text-xs text-center">Loading...</div>
          ) : topics.length === 0 ? (
            <div className="p-4 text-brain-text/40 text-xs text-center">
              No topics yet. Click "Export Now" to generate.
            </div>
          ) : (
            topics.map((topic) => (
              <button
                key={topic.slug}
                onClick={() => handleSelectTopic(topic.slug)}
                className={`w-full text-left px-3 py-2 border-b border-brain-border/50 transition-colors ${
                  selectedSlug === topic.slug
                    ? "bg-brain-accent/15 text-brain-accent"
                    : "text-brain-text/70 hover:bg-brain-surface"
                }`}
              >
                <div className="text-xs font-medium truncate">{topic.name}</div>
                <div className="flex items-center gap-2 mt-0.5">
                  <span className="text-[10px] text-brain-text/30">
                    {topic.memory_count} {topic.memory_count === 1 ? "memory" : "memories"}
                  </span>
                  <span className="text-[10px] text-brain-text/20">
                    {formatBytes(topic.size_bytes)}
                  </span>
                </div>
              </button>
            ))
          )}
        </div>

        {/* Content pane */}
        <div className="flex-1 min-w-0 overflow-y-auto p-4">
          {contentLoading ? (
            <div className="text-brain-text/40 text-xs text-center mt-8">Loading topic...</div>
          ) : selectedContent ? (
            <div className="max-w-2xl">{renderMarkdown(selectedContent)}</div>
          ) : (
            <div className="text-brain-text/30 text-xs text-center mt-8">
              {topics.length > 0
                ? "Select a topic to view its content"
                : "Export knowledge to see topics here"}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
