/**
 * Transport abstraction for DeepBrain.
 *
 * In Tauri mode: delegates to `@tauri-apps/api/core` invoke().
 * In browser mode: delegates to fetch() against the Axum REST API.
 */

// ---- Environment detection ----

export function isTauri(): boolean {
  return !!(window as unknown as Record<string, unknown>).__TAURI_INTERNALS__;
}

// ---- Auth token (browser mode only) ----

const TOKEN_KEY = "deepbrain_api_token";

export function getAuthToken(): string | null {
  return localStorage.getItem(TOKEN_KEY);
}

export function setAuthToken(token: string): void {
  localStorage.setItem(TOKEN_KEY, token);
}

export function clearAuthToken(): void {
  localStorage.removeItem(TOKEN_KEY);
}

// ---- camelCase → snake_case conversion ----

function toSnakeCase(str: string): string {
  return str.replace(/[A-Z]/g, (c) => `_${c.toLowerCase()}`);
}

function convertArgs(args: Record<string, unknown>): Record<string, unknown> {
  const converted: Record<string, unknown> = {};
  for (const [key, value] of Object.entries(args)) {
    converted[toSnakeCase(key)] = value;
  }
  return converted;
}

// ---- Command-to-REST mapping ----

interface Route {
  method: "GET" | "POST" | "DELETE" | "PATCH";
  path: string;
}

const COMMAND_MAP: Record<string, Route> = {
  // Core cognition
  think:               { method: "POST", path: "/api/think" },
  remember:            { method: "POST", path: "/api/remember" },
  recall:              { method: "POST", path: "/api/recall" },

  // Status & settings
  get_status:          { method: "GET",  path: "/api/status" },
  get_settings:        { method: "GET",  path: "/api/settings" },
  update_settings:     { method: "POST", path: "/api/settings" },

  // Search
  search_files:        { method: "POST", path: "/api/search/files" },
  search_emails:       { method: "POST", path: "/api/search/emails" },
  spotlight_search:    { method: "POST", path: "/api/spotlight/search" },
  mail_search:         { method: "POST", path: "/api/mail/search" },

  // Memory CRUD
  list_memories:       { method: "POST", path: "/api/memories" },
  get_memory:          { method: "GET",  path: "/api/memories/:id" },
  delete_memory:       { method: "DELETE", path: "/api/memories/:id" },
  update_memory:       { method: "PATCH", path: "/api/memories/:id" },

  // Files & indexing
  list_indexed_files:  { method: "POST", path: "/api/files/list" },
  get_file_chunks:     { method: "POST", path: "/api/files/chunks" },
  index_files:         { method: "POST", path: "/api/indexer/scan" },
  add_indexed_folder:  { method: "POST", path: "/api/indexer/folder" },

  // Thoughts & stats
  get_thoughts:        { method: "GET",  path: "/api/thoughts" },
  get_stats:           { method: "GET",  path: "/api/stats" },

  // Brain actions
  evolve:              { method: "POST", path: "/api/brain/evolve" },
  cycle:               { method: "POST", path: "/api/brain/cycle" },
  flush:               { method: "POST", path: "/api/brain/flush" },

  // Email
  email_stats:         { method: "GET",  path: "/api/email/stats" },
  index_emails:        { method: "POST", path: "/api/email/index" },
  get_recent_emails:   { method: "GET",  path: "/api/email/recent" },

  // Clipboard
  get_clipboard_history: { method: "GET", path: "/api/clipboard" },

  // Workflow
  run_workflow:        { method: "POST", path: "/api/workflow" },

  // Storage
  get_storage_metrics: { method: "GET",  path: "/api/storage/health" },
  verify_storage:      { method: "GET",  path: "/api/storage/verify" },
  migrate_from_v1:     { method: "POST", path: "/api/storage/migrate" },

  // Dashboard / DeepBrain subsystems
  get_sona_stats:      { method: "GET",  path: "/api/sona/stats" },
  bootstrap_sona:      { method: "POST", path: "/api/sona/bootstrap" },
  get_nervous_stats:   { method: "GET",  path: "/api/nervous/stats" },
  verify_all:          { method: "POST", path: "/api/verify" },

  // LLM
  load_local_model:    { method: "POST", path: "/api/llm/load" },
  unload_local_model:  { method: "POST", path: "/api/llm/unload" },
  local_model_status:  { method: "GET",  path: "/api/llm/status" },

  // Ollama
  check_ollama:        { method: "GET",  path: "/api/ollama/status" },

  // Folders
  get_common_folders:  { method: "GET",  path: "/api/folders/common" },
};

// Commands that only work in Tauri desktop mode
const TAURI_ONLY = new Set(["pick_folder", "set_pinned", "get_pinned"]);

// ---- Main invoke function ----

export async function invoke<T>(command: string, args?: Record<string, unknown>): Promise<T> {
  // Tauri mode: delegate to real IPC
  if (isTauri()) {
    const { invoke: tauriInvoke } = await import("@tauri-apps/api/core");
    return tauriInvoke<T>(command, args);
  }

  // Browser mode: Tauri-only commands return defaults
  if (TAURI_ONLY.has(command)) {
    if (command === "get_pinned") return false as T;
    return undefined as T;
  }

  const route = COMMAND_MAP[command];
  if (!route) {
    throw new Error(`No REST mapping for command: ${command}`);
  }

  // Resolve path parameters (e.g., :id)
  let path = route.path;
  const body = args ? convertArgs(args) : {};
  if (path.includes(":id") && body.id) {
    path = path.replace(":id", encodeURIComponent(String(body.id)));
    delete body.id;
  }

  const token = getAuthToken();
  const headers: Record<string, string> = { "Content-Type": "application/json" };
  if (token) headers["Authorization"] = `Bearer ${token}`;

  const isGet = route.method === "GET";
  // For GET requests with params, append as query string
  let url = path;
  if (isGet && args && Object.keys(body).length > 0) {
    const params = new URLSearchParams();
    for (const [k, v] of Object.entries(body)) {
      if (v !== null && v !== undefined) params.set(k, String(v));
    }
    url += `?${params.toString()}`;
  }

  const resp = await fetch(url, {
    method: route.method,
    headers,
    body: !isGet ? JSON.stringify(body) : undefined,
  });

  if (!resp.ok) {
    const err = await resp.json().catch(() => ({ error: resp.statusText }));
    throw new Error((err as { error?: string }).error || `HTTP ${resp.status}`);
  }

  // Handle empty responses (204 No Content or empty body)
  const text = await resp.text();
  if (!text) return undefined as T;
  return JSON.parse(text) as T;
}
