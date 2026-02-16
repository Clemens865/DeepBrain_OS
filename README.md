# SuperBrain

An intelligent macOS menu bar app that acts as a personal cognitive layer: semantic memory, file search, contextual assistance, and workflow automation — all running locally.

## What It Does

SuperBrain sits in your menu bar and provides a Spotlight-like overlay (triggered by `Cmd+Shift+Space`) that lets you:

- **Search your memories** — Store and retrieve information using semantic vector similarity
- **Search your files** — Indexed files from ~/Documents, ~/Desktop, ~/Downloads are searchable by meaning, not just keywords
- **Think with context** — The cognitive engine uses reinforcement learning to improve responses over time
- **Run workflows** — Quick actions for clipboard capture, learning digests, and more
- **Connect to AI** — Optional Ollama (local) or Claude (cloud) for enhanced responses

## Architecture

```
┌─ Menu Bar Icon ──────────────────────────────────────┐
│  Left-click: toggle overlay | Right-click: menu      │
└──────────────────────┬───────────────────────────────┘
                       │
┌─ Overlay (Cmd+Shift+Space) ─────────────────────────┐
│  Search bar → Response / Memories / Files tabs       │
│  Quick Actions: Remember, Clipboard, Digest, Status  │
└──────────────────────┬───────────────────────────────┘
                       │ Tauri IPC
┌─ Rust Backend ───────┴──────────────────────────────┐
│  brain/      Cognitive engine, 384-dim vectors,      │
│              Q-learning, DashMap memory store         │
│  ai/         Ollama + Claude providers               │
│  indexer/    File watcher, chunker, parser (35+ ext) │
│  context.rs  Clipboard history                       │
│  workflows.rs  Automated cognitive actions           │
│  persistence  SQLite with WAL mode                   │
└──────────────────────────────────────────────────────┘
```

### Tech Stack

| Layer | Technology |
|-------|-----------|
| Desktop framework | Tauri 2 |
| Backend | Rust (tokio, rusqlite, reqwest, ort, notify, dashmap, parking_lot) |
| Frontend | React 18 + TypeScript + Tailwind CSS + Zustand |
| Embeddings | ONNX all-MiniLM-L6-v2 (384-dim) with Ollama + hash fallback |
| Persistence | SQLite (WAL mode) for memories, Q-table, file index |
| AI (optional) | Ollama (local) or Claude API (cloud) |

## Getting Started

### Prerequisites

- macOS 12+
- Rust toolchain (`rustup`)
- Node.js 18+
- [Ollama](https://ollama.ai) (optional, for AI-enhanced responses)

### Development

```bash
# Install frontend dependencies
npm install

# Run in development mode
cargo tauri dev
```

The app will:
1. Start a Vite dev server on port 1420
2. Compile the Rust backend
3. Launch as a menu bar app (no dock icon)

### Build

```bash
# Production build (creates .dmg / .app)
cargo tauri build
```

### Running Tests

```bash
# Rust tests (38 tests)
cd src-tauri && cargo test

# Frontend build verification
npx vite build
```

## Project Structure

```
superbrain-app/
├── src/                          # React frontend
│   ├── components/
│   │   ├── SearchBar.tsx         # Auto-focused search input
│   │   ├── ResultsList.tsx       # Tabbed: Response / Memories / Files
│   │   ├── MemoryFeed.tsx        # Status dashboard
│   │   ├── QuickActions.tsx      # Remember, Clipboard, Digest, Settings
│   │   └── Settings.tsx          # AI provider, theme, privacy config
│   ├── store/appStore.ts         # Zustand state management
│   ├── hooks/useBrain.ts         # Tauri IPC hooks
│   └── App.tsx                   # Main app shell
├── src-tauri/                    # Rust backend
│   └── src/
│       ├── main.rs               # Tauri entry, tray, shortcuts, watcher
│       ├── commands.rs           # 15 IPC command handlers
│       ├── tray.rs               # System tray + context menu
│       ├── overlay.rs            # Window show/hide/toggle
│       ├── state.rs              # AppState: engine + embeddings + persistence
│       ├── brain/
│       │   ├── cognitive.rs      # CognitiveEngine (think, remember, recall, evolve)
│       │   ├── memory.rs         # DashMap vector memory with cosine search
│       │   ├── learning.rs       # Q-Learning + experience replay + meta-learning
│       │   ├── embeddings.rs     # ONNX + Ollama + Hash embedding providers
│       │   ├── persistence.rs    # SQLite storage for memories, Q-table, config
│       │   ├── types.rs          # Shared type definitions
│       │   └── utils.rs          # Vector math, similarity, normalization
│       ├── keychain.rs            # macOS Keychain (Claude API key storage)
│       ├── ai/
│       │   ├── mod.rs            # AiProvider trait
│       │   ├── ollama.rs         # Ollama REST client
│       │   └── claude.rs         # Anthropic Messages API client
│       ├── indexer/
│       │   ├── mod.rs            # FileIndexer: scan, index, semantic search
│       │   ├── parser.rs         # File content extraction (35+ extensions)
│       │   ├── chunker.rs        # 512-token chunks with 128-token overlap
│       │   └── watcher.rs        # notify-based filesystem watcher
│       ├── context.rs            # Clipboard history manager
│       └── workflows.rs          # Built-in workflow actions
└── package.json
```

## IPC Commands

| Command | Description |
|---------|-------------|
| `think` | Process input through cognitive engine |
| `remember` | Store content as a memory with embedding |
| `recall` | Semantic search across memories |
| `search_files` | Semantic search across indexed files |
| `index_files` | Trigger full re-index of watched directories |
| `run_workflow` | Execute a built-in workflow (digest, clipboard, etc.) |
| `get_status` | System status (memories, thoughts, indexed files) |
| `get_settings` / `update_settings` | App configuration |
| `get_thoughts` / `get_stats` | Cognitive engine introspection |
| `evolve` / `cycle` | Trigger learning evolution / cognitive cycle |
| `check_ollama` | Detect Ollama availability and list models |
| `flush` | Persist all state to disk |

## Test Coverage

38 tests (19 lib + 19 bin), all passing:

| Module | Tests | What's Tested |
|--------|-------|---------------|
| utils | 2 | Vector normalization, cosine similarity |
| memory | 2 | Store/search with f64 and f32 vectors |
| learning | 1 | Q-learning action selection and updates |
| cognitive | 1 | End-to-end think/remember/recall |
| embeddings | 2 | Dimension correctness, hash similarity |
| persistence | 4 | Memory round-trip, Q-table, config, batch ops |
| chunker | 4 | Overlapping chunks, empty input, paragraphs |
| parser | 3 | Text cleaning, markup stripping, extension support |

## Current Status

### Completed (v0.1.0)

- [x] **Phase 1: Production Core** — Ported cognitive engine from NAPI to pure Rust, hash embeddings (384-dim), SQLite persistence
- [x] **Phase 2: Tauri Shell** — Menu bar app, system tray, `Cmd+Shift+Space` overlay, frameless transparent window
- [x] **Phase 3: Frontend UI** — React + Tailwind dark theme, search with debounce, tabbed results, quick actions, settings
- [x] **Phase 4: AI Providers** — Ollama (local) and Claude (cloud) provider abstraction with async-trait
- [x] **Phase 5: OS Integration** — File watcher (notify), chunker (512/128), parser (35+ ext), clipboard context, 4 workflows
- [x] **Phase 6: Polish** — Zero warnings, 38 tests passing, frontend builds clean

### v0.2.0

- [x] **AI-Enhanced Think** — Ollama/Claude wired into `think` command with memory context, stores interactions as episodic memories
- [x] **Recursive File Scanning** — Walks subdirectories (max depth 10), skips node_modules/.git/target/etc.
- [x] **PDF Support** — PDF text extraction via `pdf-extract` crate (36+ file types now)
- [x] **First-Launch Onboarding** — 3-step wizard: welcome, AI detection (auto-detects Ollama), ready
- [x] **Dynamic AI Provider** — Settings changes refresh the AI provider without restart

### v0.3.0

- [x] **ONNX Embeddings** — all-MiniLM-L6-v2 (384-dim) auto-downloaded from HuggingFace, priority: ONNX > Ollama > Hash
- [x] **macOS Keychain** — Claude API key stored in Keychain via `security-framework`, stripped from SQLite
- [x] **Battery-Aware Throttling** — Cognitive cycle: 60s when plugged in, 300s on battery (`battery` crate)
- [x] **Auto-Update** — `tauri-plugin-updater` configured with GitHub Releases endpoint

### Roadmap

- [ ] **Code Signing** — Apple Developer certificate, notarization, DMG packaging
- [ ] **Universal Binary** — arm64 + x86_64 fat binary for Intel Macs
- [ ] **Onboarding Polish** — Folder picker for indexed directories, custom hotkey binding

## Privacy

- All data stored locally at `~/Library/Application Support/SuperBrain/`
- No telemetry, no phone-home
- Cloud AI (Claude) only when explicitly enabled in settings
- Privacy mode toggle disables all cloud features

## License

MIT
