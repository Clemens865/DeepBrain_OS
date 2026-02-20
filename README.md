# DeepBrain

A cognitive macOS desktop app that acts as a personal memory layer — semantic search, file indexing, self-learning, and AI assistance — all running locally on your Mac.

Built with [Tauri 2](https://tauri.app) (Rust + React), powered by the [RuVector](https://github.com/Clemens865) ecosystem for crash-safe vector storage, 3-tier self-learning, and local LLM inference.

## What It Does

DeepBrain sits in your menu bar and provides a Spotlight-like overlay (`Cmd+Shift+Space`) that lets you:

- **Search your memories** — Store and retrieve information using 384-dim semantic vector similarity
- **Search your files** — ~/Documents, ~/Desktop, ~/Downloads indexed and searchable by meaning
- **Search your emails** — macOS Mail.app messages indexed and semantically searchable
- **Think with context** — Cognitive engine uses SONA self-learning to improve responses over time
- **Run workflows** — Quick actions for clipboard capture, learning digests, and more
- **Connect to AI** — Ollama (local), Claude (cloud), or ruvllm (in-process Metal/ANE) for enhanced responses

## Architecture

```
Menu Bar Icon (left-click: toggle, right-click: menu)
         |
   Overlay (Cmd+Shift+Space)
   Search bar -> Response / Memories / Files / Emails tabs
   Quick Actions: Remember, Clipboard, Digest, Status
         | Tauri IPC
   Rust Backend
   ├── brain/         Cognitive engine, HNSW index, GNN re-ranker
   ├── deepbrain/     RVF vector store, SONA bridge, nervous system, ruvllm
   ├── ai/            Ollama + Claude + ruvllm providers
   ├── indexer/       File watcher, email indexer, chunker, parser (36+ types)
   ├── api.rs         HTTP REST API (port 19519) for CLI/scripts
   ├── keychain.rs    macOS Keychain integration
   └── autostart.rs   LaunchAgent auto-start at login
```

### Tech Stack

| Layer | Technology |
|-------|-----------|
| Desktop framework | Tauri 2 (macOS menu bar app) |
| Backend | Rust (tokio, axum, rusqlite, ort, notify, dashmap, parking_lot) |
| Frontend | React 18 + TypeScript + Tailwind CSS + Zustand |
| Embeddings | ONNX all-MiniLM-L6-v2 (384-dim), Ollama fallback, hash fallback |
| Vector storage | RVF append-only crash-safe format (rvf-runtime, rvf-types, rvf-crypto) |
| Self-learning | SONA 3-tier: MicroLoRA (instant) + K-means++ (hourly) + EWC++ (weekly) |
| Nervous system | Hopfield associative memory, predictive coding, oscillatory routing |
| Local LLM | ruvllm (in-process, Metal/ANE acceleration, GGUF support) |
| AI (optional) | Ollama (local) or Claude API (cloud) |
| Persistence | SQLite (WAL) for metadata + RVF .rvtext files for vectors |
| Security | macOS Keychain for secrets, SHAKE-256/Ed25519 witness chains |

## Getting Started

### Prerequisites

- macOS 12+
- Rust toolchain (`rustup`, min rustc 1.87)
- Node.js 18+
- [Ollama](https://ollama.ai) (optional, for AI-enhanced responses)

### Development

```bash
# Clone the repository
git clone https://github.com/Clemens865/DeepBrain_OS.git
cd DeepBrain_OS

# Install frontend dependencies
npm install

# Run in development mode
cargo tauri dev
```

The app will:
1. Start a Vite dev server on port 1420
2. Compile the Rust backend (first build pulls ~70 crates)
3. Launch as a menu bar app (no dock icon)
4. Open Cmd+Shift+Space to show the overlay

### Production Build

```bash
# Creates .app bundle + .dmg installer
cargo tauri build
```

Output: `src-tauri/target/release/bundle/macos/DeepBrain.app`

### Running Tests

```bash
# Rust tests
cd src-tauri && cargo test

# Frontend build verification
npx vite build
```

## CLI

DeepBrain includes a companion CLI for terminal access:

```bash
# Build the CLI
cargo build -p deepbrain-cli --release

# Usage
deepbrain status              # System status
deepbrain search "query"      # Semantic memory search
deepbrain remember "content"  # Store a memory
deepbrain think "question"    # AI-enhanced thinking
deepbrain files "query"       # Search indexed files
deepbrain health              # Health check

# Configuration
export DEEPBRAIN_URL=http://127.0.0.1:19519
export DEEPBRAIN_TOKEN=your-api-token
```

## HTTP API

DeepBrain runs an HTTP API on `127.0.0.1:19519` for integration with scripts and tools:

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/health` | GET | Health check |
| `/api/status` | GET | System status |
| `/api/think` | POST | Process input through cognitive engine |
| `/api/remember` | POST | Store a memory |
| `/api/recall` | POST | Semantic memory search |
| `/api/search/files` | POST | Search indexed files |
| `/api/search/emails` | POST | Search indexed emails |
| `/api/workflow` | POST | Run a workflow action |
| `/api/clipboard` | GET | Clipboard history |
| `/api/migrate` | POST | Migrate from v1 (SuperBrain) data |

API token is auto-generated and stored in macOS Keychain. Retrieve it with:
```bash
security find-generic-password -s DeepBrain -a api_token -w
```

## Project Structure

```
deepbrain-app/
├── src/                          # React frontend
│   ├── components/
│   │   ├── SearchBar.tsx         # Auto-focused search input
│   │   ├── ResultsList.tsx       # Tabbed: Response / Memories / Files
│   │   ├── Sidebar.tsx           # Navigation sidebar
│   │   ├── Settings.tsx          # AI provider, theme, privacy config
│   │   └── BrowserLogin.tsx      # API token login for browser access
│   ├── store/appStore.ts         # Zustand state management
│   ├── hooks/useBrain.ts         # Tauri IPC hooks
│   └── App.tsx                   # Main app shell
├── src-tauri/                    # Rust backend
│   └── src/
│       ├── main.rs               # Entry point, tray, shortcuts, background tasks
│       ├── commands.rs           # 30+ IPC command handlers
│       ├── api.rs                # HTTP REST API (axum)
│       ├── state.rs              # AppState: all subsystems
│       ├── brain/
│       │   ├── cognitive.rs      # CognitiveEngine (think, remember, recall, evolve)
│       │   ├── memory.rs         # DashMap vector memory with cosine search
│       │   ├── learning.rs       # Reinforcement learning + meta-learning
│       │   ├── embeddings.rs     # ONNX + Ollama + Hash embedding providers
│       │   ├── hnsw.rs           # HNSW approximate nearest-neighbor index
│       │   ├── gnn.rs            # GNN-based search re-ranker
│       │   └── persistence.rs    # SQLite storage
│       ├── deepbrain/
│       │   ├── vector_store.rs   # RvfStore wrapper (crash-safe vector storage)
│       │   ├── sona_bridge.rs    # SONA 3-tier self-learning bridge
│       │   ├── nervous_bridge.rs # Hopfield, predictive coding, routing
│       │   ├── llm_bridge.rs     # ruvllm local LLM inference
│       │   ├── migration.rs      # SQLite -> RVF migration tool
│       │   ├── id_map.rs         # String UUID <-> u64 ID mapping
│       │   └── metrics.rs        # Storage metrics and verification
│       ├── ai/
│       │   ├── ollama.rs         # Ollama REST client
│       │   └── claude.rs         # Anthropic Messages API client
│       ├── indexer/
│       │   ├── mod.rs            # FileIndexer: scan, index, semantic search
│       │   ├── email.rs          # macOS Mail.app email indexer
│       │   ├── parser.rs         # File content extraction (36+ extensions)
│       │   ├── chunker.rs        # 512-token chunks with 128-token overlap
│       │   └── watcher.rs        # notify-based filesystem watcher
│       ├── keychain.rs           # macOS Keychain integration
│       ├── autostart.rs          # LaunchAgent management
│       ├── overlay.rs            # Window show/hide/toggle
│       ├── tray.rs               # System tray + context menu
│       └── workflows.rs          # Built-in workflow actions
├── deepbrain-cli/                # Companion CLI tool
├── examples/                     # API client examples
└── Cargo.toml                    # Workspace (app + CLI)
```

## RuVector Integration

DeepBrain uses several crates from the RuVector ecosystem as local path dependencies:

| Crate | Purpose |
|-------|---------|
| `rvf-runtime` | Append-only, crash-safe vector file I/O |
| `rvf-types` | Vector metadata types, dimension handling |
| `rvf-crypto` | SHAKE-256 hashing, Ed25519 witness chains |
| `ruvector-sona` | SONA self-learning engine (3-tier: instant/hourly/weekly) |
| `ruvector-nervous-system` | Hopfield memory, predictive coding, oscillatory routing |
| `ruvllm` | Local LLM inference (Metal/ANE, GGUF, quantization) |

## Version History

| Version | Highlights |
|---------|-----------|
| v0.1.0 | Core cognitive engine, SQLite persistence, hash embeddings |
| v0.2.0 | AI-enhanced think, recursive file scanning, PDF support, onboarding |
| v0.3.0 | ONNX embeddings, macOS Keychain, battery-aware throttling, auto-update |
| v0.4.0 | RVF crash-safe storage, SONA self-learning, nervous system, ruvllm, email indexing, HTTP API, CLI |

## Privacy

- All data stored locally at `~/Library/Application Support/DeepBrain/`
- No telemetry, no phone-home
- Cloud AI (Claude) only when explicitly enabled in settings
- Privacy mode toggle disables all cloud features
- Secrets stored in macOS Keychain, never in plaintext

## License

MIT
