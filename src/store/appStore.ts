import { create } from "zustand";
import { invoke } from "../services/backend";

interface Memory {
  id: string;
  content: string;
  similarity: number;
  memory_type: string;
}

interface ThinkResult {
  response: string;
  confidence: number;
  thought_id: string;
  memory_count: number;
  ai_enhanced: boolean;
}

interface FileResult {
  path: string;
  name: string;
  chunk: string;
  similarity: number;
  file_type: string;
}

interface WorkflowResult {
  action: string;
  success: boolean;
  message: string;
  data: unknown | null;
}

interface SystemStatus {
  status: string;
  memory_count: number;
  thought_count: number;
  uptime_ms: number;
  ai_provider: string;
  ai_available: boolean;
  embedding_provider: string;
  learning_trend: string;
  indexed_files: number;
  indexed_chunks: number;
  indexed_emails: number;
}

interface Settings {
  ai_provider: string;
  ollama_model: string;
  ruvllm_model: string | null;
  claude_api_key: string | null;
  hotkey: string;
  indexed_folders: string[];
  email_accounts: string[];
  theme: string;
  auto_start: boolean;
  privacy_mode: boolean;
  onboarded: boolean;
}

// ---- DeepBrain subsystem types ----

export interface StorageMetrics {
  total_vectors: number;
  file_size_bytes: number;
  query_latency_p50_us: number;
  query_latency_p99_us: number;
}

export interface SonaStats {
  trajectories_buffered: number;
  trajectories_dropped: number;
  patterns_stored: number;
  ewc_tasks: number;
  instant_enabled: boolean;
  background_enabled: boolean;
}

export interface NervousStats {
  hopfield_patterns: number;
  hopfield_capacity: number;
  hopfield_beta: number;
  router_sync: number;
  router_modules: number;
  predictive_threshold: number;
  dentate_input_dim: number;
  dentate_output_dim: number;
  dentate_k: number;
}

export interface LlmStatus {
  model_loaded: boolean;
  model_id: string | null;
  model_info: string | null;
}

export interface VerifyResult {
  [subsystem: string]: {
    ok: boolean;
    [key: string]: unknown;
  };
}

// ---- Knowledge Graph / GNN / Compression types ----

export interface GraphStats {
  node_count: number;
  edge_count: number;
  hyperedge_count: number;
}

export interface RankedResult {
  id: string;
  content: string;
  vector_score: number;
  gnn_score: number;
  blended_score: number;
}

export interface CompressionStats {
  total_memories: number;
  hot_count: number;
  warm_count: number;
  cold_count: number;
  estimated_savings_pct: number;
}

export interface BootstrapSourceResult {
  source: string;
  items_scanned: number;
  memories_created: number;
  skipped_existing: number;
  errors: number;
}

export interface KnowledgeBootstrapResult {
  total_memories_created: number;
  total_skipped: number;
  sources: BootstrapSourceResult[];
  duration_secs: number;
}

export interface BootstrapProgress {
  source: string;
  phase: string;
  current: number;
  total: number;
  memories_created: number;
}

export interface SpotlightResult {
  path: string;
  name: string;
  kind: string;
  modified: string | null;
  content?: string;
}

export interface MailSearchResult {
  subject: string;
  sender: string;
  date: string;
  preview: string;
  account: string;
  mailbox: string;
  message_id: number;
}

export interface EmailSemanticResult {
  subject: string;
  sender: string;
  date: string;
  mailbox: string;
  chunk: string;
  similarity: number;
}

export interface EmailIndexStats {
  indexed_count: number;
  chunk_count: number;
  is_indexing: boolean;
}

interface SearchResults {
  memories: Memory[];
  files: FileResult[];
  spotlight: SpotlightResult[];
  emails: MailSearchResult[];
  semanticEmails: EmailSemanticResult[];
  thinkResult: ThinkResult | null;
}

export interface ChatMessage {
  id: string;
  role: "user" | "assistant";
  content: string;
  timestamp: number;
  ai_enhanced?: boolean;
  confidence?: number;
  memory_count?: number;
}

interface ClipboardEntry {
  content: string;
  timestamp: number;
}

// ---- Browse types ----

export type BrowseCategory =
  | "home"
  | "all-memories"
  | "semantic"
  | "episodic"
  | "working"
  | "procedural"
  | "meta"
  | "causal"
  | "goal"
  | "emotional"
  | "all-files"
  | "folder"
  | "thoughts"
  | "clipboard"
  | "spotlight"
  | "email"
  | "status"
  | "knowledge-graph";

export interface MemoryBrowseItem {
  id: string;
  content: string;
  memory_type: string;
  importance: number;
  access_count: number;
  timestamp: number;
  connection_count: number;
}

export interface FileListItem {
  path: string;
  name: string;
  ext: string;
  modified: number;
  chunk_count: number;
}

export interface Thought {
  id: string;
  content: string;
  thought_type: string;
  confidence: number;
  novelty: number;
  utility: number;
  timestamp: number;
}

export type BrowseItem =
  | { kind: "memory"; data: MemoryBrowseItem }
  | { kind: "file"; data: FileListItem }
  | { kind: "thought"; data: Thought }
  | { kind: "clipboard"; data: ClipboardEntry }
  | { kind: "spotlight"; data: SpotlightResult }
  | { kind: "email"; data: MailSearchResult };

export interface FileChunkItem {
  chunk_index: number;
  content: string;
}

export type SortField = "timestamp" | "importance" | "access_count" | "type" | "name";
export type SortDirection = "asc" | "desc";

export interface Toast {
  id: number;
  message: string;
  type: "success" | "error" | "info";
}

interface NavEntry {
  category: BrowseCategory;
  filter: string | null;
}

interface AppState {
  query: string;
  results: SearchResults;
  isSearching: boolean;
  recentMemories: Memory[];
  status: SystemStatus | null;
  settings: Settings | null;
  clipboardHistory: ClipboardEntry[];
  mode: "search" | "remember" | "browse" | "chat";

  // Pin / Expand state
  pinned: boolean;
  expandedView: boolean;

  // Chat state
  chatMessages: ChatMessage[];
  chatLoading: boolean;

  // Browse state
  browseCategory: BrowseCategory;
  browseFilter: string | null;
  browseItems: BrowseItem[];
  selectedItemId: string | null;
  selectedItemDetail: BrowseItem | null;
  browseLoading: boolean;
  navHistory: NavEntry[];
  navIndex: number;
  showPreview: boolean;
  browseFilterText: string;
  sortField: SortField;
  sortDirection: SortDirection;
  toasts: Toast[];
  fileChunks: FileChunkItem[];
  fileChunksLoading: boolean;
  spotlightQuery: string;
  spotlightResults: SpotlightResult[];
  spotlightLoading: boolean;
  mailQuery: string;
  mailResults: MailSearchResult[];
  mailLoading: boolean;

  // Dashboard state
  storageMetrics: StorageMetrics | null;
  sonaStats: SonaStats | null;
  nervousStats: NervousStats | null;
  llmStatus: LlmStatus | null;
  graphStats: GraphStats | null;
  compressionStats: CompressionStats | null;
  verifyResult: VerifyResult | null;
  verifying: boolean;

  // Bootstrap state
  bootstrapRunning: boolean;
  bootstrapProgress: BootstrapProgress | null;
  bootstrapResult: KnowledgeBootstrapResult | null;

  setQuery: (query: string) => void;
  setMode: (mode: "search" | "remember" | "browse" | "chat") => void;
  search: (query: string) => Promise<void>;
  searchFiles: (query: string) => Promise<FileResult[]>;
  runWorkflow: (action: string, query?: string) => Promise<WorkflowResult>;
  remember: (content: string, type: string, importance?: number) => Promise<void>;
  think: (input: string) => Promise<ThinkResult>;
  loadStatus: () => Promise<void>;
  loadSettings: () => Promise<void>;
  updateSettings: (settings: Settings) => Promise<void>;
  loadClipboardHistory: () => Promise<void>;
  addIndexedFolder: (path: string) => Promise<void>;
  indexFiles: () => Promise<void>;
  clearResults: () => void;

  // Browse actions
  enterBrowseMode: () => void;
  exitBrowseMode: () => void;
  navigateTo: (category: BrowseCategory, filter?: string | null) => void;
  navigateBack: () => void;
  navigateForward: () => void;
  selectBrowseItem: (id: string | null) => void;
  loadBrowseItems: () => Promise<void>;
  deleteBrowseItem: (id: string) => Promise<void>;
  togglePreview: () => void;
  setBrowseFilterText: (text: string) => void;
  setSortField: (field: SortField) => void;
  setSortDirection: (dir: SortDirection) => void;
  addToast: (message: string, type?: "success" | "error" | "info") => void;
  removeToast: (id: number) => void;
  updateBrowseItem: (id: string, content?: string, memoryType?: string, importance?: number) => Promise<void>;
  loadFileChunks: (filePath: string) => Promise<void>;
  spotlightSearch: (query: string, kind?: string) => Promise<void>;
  mailSearch: (query: string, field?: string) => Promise<void>;

  // Dashboard actions
  loadDashboardData: () => Promise<void>;
  verifyAll: () => Promise<void>;
  loadModel: (modelId: string) => Promise<void>;
  unloadModel: () => Promise<void>;
  bootstrapSona: () => Promise<void>;
  bootstrapKnowledge: (sources: string[]) => Promise<void>;

  // Pin / Expand actions
  setPinned: (pinned: boolean) => void;
  setExpandedView: (expanded: boolean) => void;

  // Chat actions
  enterChatMode: () => void;
  exitChatMode: () => void;
  sendChatMessage: (content: string) => Promise<void>;
  clearChat: () => void;
}

export const useAppStore = create<AppState>((set, get) => ({
  query: "",
  results: { memories: [], files: [], spotlight: [], emails: [], semanticEmails: [], thinkResult: null },
  isSearching: false,
  recentMemories: [],
  status: null,
  settings: null,
  clipboardHistory: [],
  mode: "search",

  // Pin / Expand defaults
  pinned: false,
  expandedView: false,

  // Chat state defaults
  chatMessages: [],
  chatLoading: false,

  // Browse state defaults
  browseCategory: "home",
  browseFilter: null,
  browseItems: [],
  selectedItemId: null,
  selectedItemDetail: null,
  browseLoading: false,
  navHistory: [{ category: "home", filter: null }],
  navIndex: 0,
  showPreview: true,
  browseFilterText: "",
  sortField: "timestamp",
  sortDirection: "desc",
  toasts: [],
  fileChunks: [],
  fileChunksLoading: false,
  spotlightQuery: "",
  spotlightResults: [],
  spotlightLoading: false,
  mailQuery: "",
  mailResults: [],
  mailLoading: false,

  // Dashboard defaults
  storageMetrics: null,
  sonaStats: null,
  nervousStats: null,
  llmStatus: null,
  graphStats: null,
  compressionStats: null,
  verifyResult: null,
  verifying: false,

  // Bootstrap defaults
  bootstrapRunning: false,
  bootstrapProgress: null,
  bootstrapResult: null,

  setQuery: (query: string) => set({ query }),
  setMode: (mode: "search" | "remember" | "browse" | "chat") => set({ mode }),

  search: async (query: string) => {
    set({ query, isSearching: true });

    try {
      // Run recall, think, file search, spotlight, semantic emails, and live mail search in parallel
      // All calls have .catch() so a slow/failing Ollama never blocks the whole search
      const [memories, thinkResult, files, spotlight, semanticEmails, emails] = await Promise.all([
        invoke<Memory[]>("recall", { query, limit: 10 }).catch(() => [] as Memory[]),
        invoke<ThinkResult>("think", { input: query }).catch(() => null as ThinkResult | null),
        invoke<FileResult[]>("search_files", { query, limit: 10 }).catch(() => [] as FileResult[]),
        invoke<SpotlightResult[]>("spotlight_search", { query, limit: 10 }).catch(() => [] as SpotlightResult[]),
        invoke<EmailSemanticResult[]>("search_emails", { query, limit: 10 }).catch(() => [] as EmailSemanticResult[]),
        invoke<MailSearchResult[]>("mail_search", { query, field: "all", limit: 5 }).catch(() => [] as MailSearchResult[]),
      ]);

      set({
        results: { memories, files, spotlight, emails, semanticEmails, thinkResult },
        isSearching: false,
      });
    } catch (error) {
      console.error("Search failed:", error);
      set({ isSearching: false });
    }
  },

  searchFiles: async (query: string) => {
    return invoke<FileResult[]>("search_files", { query, limit: 20 });
  },

  runWorkflow: async (action: string, query?: string) => {
    return invoke<WorkflowResult>("run_workflow", { action, query: query ?? null });
  },

  remember: async (content: string, type: string, importance?: number) => {
    try {
      await invoke("remember", {
        content,
        memoryType: type,
        importance: importance ?? 0.7,
      });
      // Refresh status after storing
      get().loadStatus();
    } catch (error) {
      console.error("Remember failed:", error);
    }
  },

  think: async (input: string) => {
    const result = await invoke<ThinkResult>("think", { input });
    return result;
  },

  loadStatus: async () => {
    try {
      const status = await invoke<SystemStatus>("get_status");
      set({ status });
    } catch (error) {
      console.error("Failed to load status:", error);
    }
  },

  loadSettings: async () => {
    try {
      const settings = await invoke<Settings>("get_settings");
      set({ settings });
    } catch (error) {
      console.error("Failed to load settings:", error);
    }
  },

  updateSettings: async (settings: Settings) => {
    try {
      await invoke("update_settings", { settings });
      set({ settings });
    } catch (error) {
      console.error("Failed to update settings:", error);
    }
  },

  loadClipboardHistory: async () => {
    try {
      const history = await invoke<ClipboardEntry[]>("get_clipboard_history", { limit: 20 });
      set({ clipboardHistory: history });
    } catch (error) {
      console.error("Failed to load clipboard history:", error);
    }
  },

  addIndexedFolder: async (path: string) => {
    try {
      await invoke("add_indexed_folder", { path });
      get().loadSettings();
      get().loadStatus();
    } catch (error) {
      console.error("Failed to add folder:", error);
    }
  },

  indexFiles: async () => {
    try {
      await invoke("index_files");
      get().loadStatus();
    } catch (error) {
      console.error("Failed to index files:", error);
    }
  },

  clearResults: () => {
    set({ query: "", results: { memories: [], files: [], spotlight: [], emails: [], semanticEmails: [], thinkResult: null } });
  },

  // ---- Browse actions ----

  enterBrowseMode: () => {
    set({
      mode: "browse",
      browseCategory: "home",
      browseFilter: null,
      browseItems: [],
      selectedItemId: null,
      selectedItemDetail: null,
      navHistory: [{ category: "home", filter: null }],
      navIndex: 0,
      browseFilterText: "",
    });
  },

  exitBrowseMode: () => {
    set({
      mode: "search",
      browseItems: [],
      selectedItemId: null,
      selectedItemDetail: null,
      browseFilterText: "",
    });
  },

  navigateTo: (category: BrowseCategory, filter?: string | null) => {
    const { navHistory, navIndex } = get();
    // Trim forward history when navigating to new location
    const newHistory = navHistory.slice(0, navIndex + 1);
    newHistory.push({ category, filter: filter ?? null });

    set({
      browseCategory: category,
      browseFilter: filter ?? null,
      navHistory: newHistory,
      navIndex: newHistory.length - 1,
      selectedItemId: null,
      selectedItemDetail: null,
      browseFilterText: "",
    });

    // Load items for the new category
    setTimeout(() => get().loadBrowseItems(), 0);
  },

  navigateBack: () => {
    const { navIndex, navHistory } = get();
    if (navIndex <= 0) return;
    const newIndex = navIndex - 1;
    const entry = navHistory[newIndex];
    set({
      navIndex: newIndex,
      browseCategory: entry.category,
      browseFilter: entry.filter,
      selectedItemId: null,
      selectedItemDetail: null,
      browseFilterText: "",
    });
    setTimeout(() => get().loadBrowseItems(), 0);
  },

  navigateForward: () => {
    const { navIndex, navHistory } = get();
    if (navIndex >= navHistory.length - 1) return;
    const newIndex = navIndex + 1;
    const entry = navHistory[newIndex];
    set({
      navIndex: newIndex,
      browseCategory: entry.category,
      browseFilter: entry.filter,
      selectedItemId: null,
      selectedItemDetail: null,
      browseFilterText: "",
    });
    setTimeout(() => get().loadBrowseItems(), 0);
  },

  selectBrowseItem: (id: string | null) => {
    if (!id) {
      set({ selectedItemId: null, selectedItemDetail: null });
      return;
    }
    const { browseItems } = get();
    const item = browseItems.find((bi) => {
      if (bi.kind === "memory") return bi.data.id === id;
      if (bi.kind === "file") return bi.data.path === id;
      if (bi.kind === "thought") return bi.data.id === id;
      if (bi.kind === "clipboard") return String(bi.data.timestamp) === id;
      if (bi.kind === "spotlight") return bi.data.path === id;
      if (bi.kind === "email") return String(bi.data.message_id) === id;
      return false;
    });
    set({ selectedItemId: id, selectedItemDetail: item ?? null });
  },

  loadBrowseItems: async () => {
    const { browseCategory, browseFilter } = get();
    set({ browseLoading: true });

    try {
      const memoryTypes = [
        "all-memories", "semantic", "episodic", "working", "procedural",
        "meta", "causal", "goal", "emotional",
      ];

      if (memoryTypes.includes(browseCategory)) {
        const memoryType = browseCategory === "all-memories" ? null : browseCategory;
        const items = await invoke<MemoryBrowseItem[]>("list_memories", {
          memoryType,
          limit: 200,
          offset: 0,
        });
        set({
          browseItems: items.map((d) => ({ kind: "memory" as const, data: d })),
          browseLoading: false,
        });
      } else if (browseCategory === "all-files" || browseCategory === "folder") {
        const items = await invoke<FileListItem[]>("list_indexed_files", {
          folder: browseFilter,
          limit: 200,
          offset: 0,
        });
        set({
          browseItems: items.map((d) => ({ kind: "file" as const, data: d })),
          browseLoading: false,
        });
      } else if (browseCategory === "thoughts") {
        const items = await invoke<Thought[]>("get_thoughts", { limit: 200 });
        set({
          browseItems: items.map((d) => ({ kind: "thought" as const, data: d })),
          browseLoading: false,
        });
      } else if (browseCategory === "clipboard") {
        const items = await invoke<ClipboardEntry[]>("get_clipboard_history", { limit: 50 });
        set({
          browseItems: items.map((d) => ({ kind: "clipboard" as const, data: d })),
          clipboardHistory: items,
          browseLoading: false,
        });
      } else if (browseCategory === "spotlight") {
        // Spotlight requires a query — use spotlightQuery or browseFilterText
        const q = get().spotlightQuery || get().browseFilterText;
        if (q) {
          const items = await invoke<SpotlightResult[]>("spotlight_search", { query: q, limit: 50 });
          set({
            browseItems: items.map((d) => ({ kind: "spotlight" as const, data: d })),
            spotlightResults: items,
            browseLoading: false,
          });
        } else {
          set({ browseItems: [], browseLoading: false });
        }
      } else if (browseCategory === "email") {
        // If there's a search query, use semantic search; otherwise show cached recent emails.
        // Both paths use SQLite-cached data (no live AppleScript).
        const q = get().mailQuery || get().browseFilterText;
        if (q) {
          // Semantic search — returns {subject, sender, date, mailbox, chunk, similarity}
          const raw = await invoke<any[]>("search_emails", { query: q, limit: 50 });
          const items: MailSearchResult[] = raw.map((r) => ({
            subject: r.subject,
            sender: r.sender,
            date: r.date,
            preview: (r.chunk || "").slice(0, 200),
            account: "",
            mailbox: r.mailbox || "",
            message_id: 0,
          }));
          set({
            browseItems: items.map((d) => ({ kind: "email" as const, data: d })),
            mailResults: items,
            browseLoading: false,
          });
        } else {
          // Recent indexed emails from SQLite cache (instant, no AppleScript)
          const items = await invoke<MailSearchResult[]>("get_recent_emails", { limit: 30 }).catch(() => [] as MailSearchResult[]);
          set({
            browseItems: items.map((d) => ({ kind: "email" as const, data: d })),
            mailResults: items,
            browseLoading: false,
          });
        }
      } else {
        // "home" and "status" don't load items
        set({ browseItems: [], browseLoading: false });
      }
    } catch (error) {
      console.error("Failed to load browse items:", error);
      set({ browseLoading: false });
    }
  },

  deleteBrowseItem: async (id: string) => {
    try {
      await invoke("delete_memory", { id });
      // Remove from local list
      const { browseItems, selectedItemId } = get();
      const filtered = browseItems.filter(
        (bi) => !(bi.kind === "memory" && bi.data.id === id)
      );
      set({
        browseItems: filtered,
        selectedItemId: selectedItemId === id ? null : selectedItemId,
        selectedItemDetail: selectedItemId === id ? null : get().selectedItemDetail,
      });
      get().loadStatus();
      get().addToast("Memory deleted", "success");
    } catch (error) {
      console.error("Failed to delete item:", error);
      get().addToast("Failed to delete memory", "error");
    }
  },

  togglePreview: () => {
    set({ showPreview: !get().showPreview });
  },

  setBrowseFilterText: (text: string) => {
    set({ browseFilterText: text });
  },

  setSortField: (field: SortField) => {
    set({ sortField: field });
  },

  setSortDirection: (dir: SortDirection) => {
    set({ sortDirection: dir });
  },

  addToast: (message: string, type: "success" | "error" | "info" = "info") => {
    const id = Date.now();
    const toast: Toast = { id, message, type };
    set({ toasts: [...get().toasts, toast] });
    setTimeout(() => {
      set({ toasts: get().toasts.filter((t) => t.id !== id) });
    }, 3000);
  },

  removeToast: (id: number) => {
    set({ toasts: get().toasts.filter((t) => t.id !== id) });
  },

  updateBrowseItem: async (id: string, content?: string, memoryType?: string, importance?: number) => {
    try {
      const updated = await invoke<MemoryBrowseItem>("update_memory", {
        id,
        content: content ?? null,
        memoryType: memoryType ?? null,
        importance: importance ?? null,
      });
      // Update in local list
      const { browseItems, selectedItemId } = get();
      const newItems = browseItems.map((bi) =>
        bi.kind === "memory" && bi.data.id === id
          ? { ...bi, data: updated }
          : bi
      );
      set({ browseItems: newItems });
      // Update preview if this item is selected
      if (selectedItemId === id) {
        set({ selectedItemDetail: { kind: "memory", data: updated } });
      }
      get().addToast("Memory updated", "success");
    } catch (error) {
      console.error("Failed to update memory:", error);
      get().addToast("Failed to update memory", "error");
    }
  },

  loadFileChunks: async (filePath: string) => {
    set({ fileChunksLoading: true, fileChunks: [] });
    try {
      const chunks = await invoke<FileChunkItem[]>("get_file_chunks", { filePath, limit: 10 });
      set({ fileChunks: chunks, fileChunksLoading: false });
    } catch (error) {
      console.error("Failed to load file chunks:", error);
      set({ fileChunksLoading: false });
    }
  },

  spotlightSearch: async (query: string, kind?: string) => {
    set({ spotlightLoading: true, spotlightQuery: query });
    try {
      const items = await invoke<SpotlightResult[]>("spotlight_search", {
        query,
        kind: kind ?? null,
        limit: 50,
      });
      set({ spotlightResults: items, spotlightLoading: false });
      // If in spotlight browse category, also update browseItems
      if (get().browseCategory === "spotlight") {
        set({
          browseItems: items.map((d) => ({ kind: "spotlight" as const, data: d })),
        });
      }
    } catch (error) {
      console.error("Spotlight search failed:", error);
      set({ spotlightLoading: false });
    }
  },

  mailSearch: async (query: string, field?: string) => {
    set({ mailLoading: true, mailQuery: query });
    try {
      const items = await invoke<MailSearchResult[]>("mail_search", {
        query,
        field: field ?? "all",
        limit: 50,
      });
      set({ mailResults: items, mailLoading: false });
      if (get().browseCategory === "email") {
        set({
          browseItems: items.map((d) => ({ kind: "email" as const, data: d })),
        });
      }
    } catch (error) {
      console.error("Mail search failed:", error);
      set({ mailLoading: false });
    }
  },

  // ---- Dashboard actions ----

  loadDashboardData: async () => {
    try {
      const [storageMetrics, sonaStats, nervousStats, llmStatus, graphStats, compressionStats] = await Promise.all([
        invoke<StorageMetrics>("get_storage_metrics").catch(() => null),
        invoke<SonaStats>("get_sona_stats").catch(() => null),
        invoke<NervousStats>("get_nervous_stats").catch(() => null),
        invoke<LlmStatus>("local_model_status").catch(() => null),
        invoke<GraphStats>("graph_stats").catch(() => null),
        invoke<CompressionStats>("compression_scan").catch(() => null),
      ]);
      set({ storageMetrics, sonaStats, nervousStats, llmStatus, graphStats, compressionStats });
    } catch (error) {
      console.error("Failed to load dashboard data:", error);
    }
  },

  verifyAll: async () => {
    set({ verifying: true });
    try {
      const result = await invoke<VerifyResult>("verify_all");
      set({ verifyResult: result, verifying: false });
    } catch (error) {
      console.error("Verification failed:", error);
      set({ verifying: false });
    }
  },

  loadModel: async (modelId: string) => {
    try {
      const status = await invoke<LlmStatus>("load_local_model", { modelId });
      set({ llmStatus: status });
      get().addToast("Model loaded successfully", "success");
    } catch (error) {
      console.error("Failed to load model:", error);
      get().addToast(`Failed to load model: ${error}`, "error");
    }
  },

  unloadModel: async () => {
    try {
      await invoke("unload_local_model");
      const status = await invoke<LlmStatus>("local_model_status").catch(() => null);
      set({ llmStatus: status });
      get().addToast("Model unloaded", "info");
    } catch (error) {
      console.error("Failed to unload model:", error);
    }
  },

  bootstrapSona: async () => {
    try {
      const result = await invoke<{ trajectories_recorded: number; patterns_extracted: number; message: string }>("bootstrap_sona");
      get().addToast(result.message, "success");
      // Refresh dashboard data to reflect new patterns
      get().loadDashboardData();
    } catch (error) {
      console.error("SONA bootstrap failed:", error);
      get().addToast(`Bootstrap failed: ${error}`, "error");
    }
  },

  bootstrapKnowledge: async (sources: string[]) => {
    set({ bootstrapRunning: true, bootstrapProgress: null, bootstrapResult: null });
    try {
      const result = await invoke<KnowledgeBootstrapResult>("bootstrap_knowledge", { sources });
      set({ bootstrapResult: result, bootstrapRunning: false });
      get().addToast(
        `Bootstrap complete: ${result.total_memories_created} memories created in ${result.duration_secs.toFixed(1)}s`,
        "success"
      );
      // Refresh dashboard data to reflect new memories/graph
      get().loadDashboardData();
      get().loadStatus();
    } catch (error) {
      console.error("Knowledge bootstrap failed:", error);
      set({ bootstrapRunning: false });
      get().addToast(`Bootstrap failed: ${error}`, "error");
    }
  },

  // ---- Pin / Expand actions ----

  setPinned: (pinned: boolean) => {
    invoke("set_pinned", { pinned });
    set({ pinned });
  },

  setExpandedView: (expanded: boolean) => {
    set({ expandedView: expanded });
    if (expanded) {
      // Expanding auto-pins
      invoke("set_pinned", { pinned: true });
      set({ pinned: true });
    }
  },

  // ---- Chat actions ----

  enterChatMode: () => {
    set({ mode: "chat" });
  },

  exitChatMode: () => {
    set({ mode: "search" });
  },

  sendChatMessage: async (content: string) => {
    const userMsg: ChatMessage = {
      id: `user-${Date.now()}`,
      role: "user",
      content,
      timestamp: Date.now(),
    };
    set({
      chatMessages: [...get().chatMessages, userMsg],
      chatLoading: true,
    });

    try {
      const result = await invoke<ThinkResult>("think", { input: content });
      const assistantMsg: ChatMessage = {
        id: `assistant-${Date.now()}`,
        role: "assistant",
        content: result.response,
        timestamp: Date.now(),
        ai_enhanced: result.ai_enhanced,
        confidence: result.confidence,
        memory_count: result.memory_count,
      };
      set({
        chatMessages: [...get().chatMessages, assistantMsg],
        chatLoading: false,
      });
    } catch (error) {
      const errorMsg: ChatMessage = {
        id: `error-${Date.now()}`,
        role: "assistant",
        content: "Sorry, I couldn't process that. The AI provider may be unavailable.",
        timestamp: Date.now(),
      };
      set({
        chatMessages: [...get().chatMessages, errorMsg],
        chatLoading: false,
      });
    }
  },

  clearChat: () => {
    set({ chatMessages: [] });
  },
}));
