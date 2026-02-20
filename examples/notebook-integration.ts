/**
 * DeepBrain API client for Node.js / Electron apps.
 *
 * Usage:
 *   const brain = new DeepBrainService();
 *   const status = await brain.status();
 *   const result = await brain.think("What did I learn today?");
 */

interface ThinkResponse {
  response: string;
  confidence: number;
  thought_id: string;
  memory_count: number;
  ai_enhanced: boolean;
}

interface RememberResponse {
  id: string;
  memory_count: number;
}

interface RecallItem {
  id: string;
  content: string;
  similarity: number;
  memory_type: string;
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
  data?: unknown;
}

interface ClipboardEntry {
  content: string;
  timestamp: number;
}

export class DeepBrainService {
  private baseUrl: string;
  private token?: string;

  constructor(baseUrl = "http://127.0.0.1:19519", token?: string) {
    this.baseUrl = baseUrl;
    this.token = token;
  }

  private headers(): Record<string, string> {
    const h: Record<string, string> = { "Content-Type": "application/json" };
    if (this.token) h["Authorization"] = `Bearer ${this.token}`;
    return h;
  }

  private async get<T>(path: string): Promise<T> {
    const res = await fetch(`${this.baseUrl}${path}`, { headers: this.headers() });
    if (!res.ok) throw new Error(`${res.status}: ${await res.text()}`);
    return res.json() as Promise<T>;
  }

  private async post<T>(path: string, body: unknown): Promise<T> {
    const res = await fetch(`${this.baseUrl}${path}`, {
      method: "POST",
      headers: this.headers(),
      body: JSON.stringify(body),
    });
    if (!res.ok) throw new Error(`${res.status}: ${await res.text()}`);
    return res.json() as Promise<T>;
  }

  async health(): Promise<boolean> {
    const r = await this.get<{ ok: boolean }>("/api/health");
    return r.ok;
  }

  async status(): Promise<SystemStatus> {
    return this.get("/api/status");
  }

  async think(input: string): Promise<ThinkResponse> {
    return this.post("/api/think", { input });
  }

  async remember(content: string, memoryType = "semantic", importance?: number): Promise<RememberResponse> {
    return this.post("/api/remember", { content, memory_type: memoryType, importance });
  }

  async recall(query: string, limit = 5): Promise<RecallItem[]> {
    return this.post("/api/recall", { query, limit });
  }

  async searchFiles(query: string, limit = 10): Promise<FileResult[]> {
    return this.post("/api/search/files", { query, limit });
  }

  async runWorkflow(action: string, query?: string): Promise<WorkflowResult> {
    return this.post("/api/workflow", { action, query });
  }

  async clipboard(): Promise<ClipboardEntry[]> {
    return this.get("/api/clipboard");
  }
}
