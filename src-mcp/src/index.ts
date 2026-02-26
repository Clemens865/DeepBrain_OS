#!/usr/bin/env node
import { McpServer } from "@modelcontextprotocol/sdk/server/mcp.js";
import { StdioServerTransport } from "@modelcontextprotocol/sdk/server/stdio.js";
import { z } from "zod";
import { readFileSync } from "node:fs";
import { join } from "node:path";
import { homedir } from "node:os";

// ---- Config ----

const PORT = process.env.DEEPBRAIN_PORT ?? "19519";
const BASE_URL = `http://127.0.0.1:${PORT}`;

function loadToken(): string | null {
  // Primary: data dir token file
  const tokenPath = join(
    homedir(),
    "Library",
    "Application Support",
    "DeepBrain",
    ".api_token"
  );
  try {
    return readFileSync(tokenPath, "utf-8").trim();
  } catch {
    return null;
  }
}

async function apiCall(
  path: string,
  body: Record<string, unknown>
): Promise<unknown> {
  const token = loadToken();
  if (!token) {
    throw new Error(
      `API token not found at ~/Library/Application Support/DeepBrain/.api_token — is DeepBrain running?`
    );
  }

  let res: Response;
  try {
    res = await fetch(`${BASE_URL}${path}`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        Authorization: `Bearer ${token}`,
      },
      body: JSON.stringify(body),
    });
  } catch {
    throw new Error(
      "DeepBrain is not running. Start the DeepBrain app first."
    );
  }

  if (!res.ok) {
    const text = await res.text().catch(() => "");
    throw new Error(`DeepBrain API error ${res.status}: ${text}`);
  }

  return res.json();
}

// ---- MCP Server ----

const server = new McpServer({
  name: "deepbrain",
  version: "1.0.0",
});

// Tool: deepbrain_remember
server.tool(
  "deepbrain_remember",
  `Save information to DeepBrain's long-term memory. Choose the right memory type:
- semantic: Facts, knowledge, documentation — "what things are"
- episodic: Events, conversations, experiences — "what happened"
- procedural: How-to instructions, steps, processes — "how to do things"
- emotional: Preferences, likes/dislikes, sentiments — "how I feel about X"
- goal: Plans, intentions, objectives — "what I want to achieve"
- causal: Cause-and-effect relationships — "X leads to Y"
- meta: Self-knowledge, patterns, reflections — "insights about myself"
- working: Temporary context for the current task (may expire)`,
  {
    content: z.string().describe("The text content to remember"),
    importance: z
      .number()
      .min(0)
      .max(1)
      .optional()
      .describe("Importance score from 0.0 to 1.0 (default: 0.5)"),
    memory_type: z
      .enum([
        "semantic",
        "episodic",
        "procedural",
        "emotional",
        "goal",
        "causal",
        "meta",
        "working",
      ])
      .optional()
      .describe("Memory type (default: semantic)"),
  },
  async ({ content, importance, memory_type }) => {
    try {
      const result = (await apiCall("/api/remember", {
        content,
        importance: importance ?? 0.5,
        memory_type: memory_type ?? "semantic",
      })) as { id: string; memory_count: number };

      return {
        content: [
          {
            type: "text" as const,
            text: `Saved memory ${result.id}. Total memories: ${result.memory_count}`,
          },
        ],
      };
    } catch (e) {
      return {
        content: [
          { type: "text" as const, text: `Error: ${(e as Error).message}` },
        ],
        isError: true,
      };
    }
  }
);

// Tool: deepbrain_recall
server.tool(
  "deepbrain_recall",
  "Search DeepBrain's memory for information relevant to a query. Returns the most similar memories with similarity scores.",
  {
    query: z.string().describe("Search query to find relevant memories"),
    limit: z
      .number()
      .int()
      .min(1)
      .max(50)
      .optional()
      .describe("Maximum number of results (default: 10)"),
  },
  async ({ query, limit }) => {
    try {
      const results = (await apiCall("/api/recall", {
        query,
        limit: limit ?? 10,
      })) as Array<{
        id: string;
        content: string;
        similarity: number;
        memory_type: string;
      }>;

      if (results.length === 0) {
        return {
          content: [
            {
              type: "text" as const,
              text: "No matching memories found.",
            },
          ],
        };
      }

      const formatted = results
        .map(
          (r, i) =>
            `${i + 1}. [${r.memory_type}] (similarity: ${r.similarity.toFixed(3)})\n   ${r.content}`
        )
        .join("\n\n");

      return {
        content: [
          {
            type: "text" as const,
            text: `Found ${results.length} memories:\n\n${formatted}`,
          },
        ],
      };
    } catch (e) {
      return {
        content: [
          { type: "text" as const, text: `Error: ${(e as Error).message}` },
        ],
        isError: true,
      };
    }
  }
);

// ---- Start ----

async function main() {
  const transport = new StdioServerTransport();
  await server.connect(transport);
}

main().catch((err) => {
  console.error("Fatal:", err);
  process.exit(1);
});
