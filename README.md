# QORA — Multi-Model Multi-Modality AI Platform

Pure-Rust API server fronting an entire QORA model family. **One binary** serves text + vision (LLM), text-to-speech, speech-to-text, embeddings, and reranking — all with an OpenAI-compatible API plus a unified `/v1/inference` endpoint.

No Python. No CUDA. No external ML runtime. Single binary + model weight files = portable, sellable AI infrastructure.

---

## Capabilities

| Capability | Engine | Model formats supported |
|---|---|---|
| Chat / completion (text + vision) | `LlmEngine` | `.qor4b` family — works for QOR2B, QOR3B, QOR4B, QOR9B, QOR27B and any future size |
| Text-to-speech (voice cloning, 25 voices, 10 languages) | `QoraTtsEngine` | `.qora-tts` |
| Speech-to-text (Whisper-based, multi-language) | `QoraSttEngine` | `.qora-stt-small` |
| Sentence embedding (2048-dim, multilingual) | `QorEmbeddingEngine` | qor-rag embedding model |
| Reranking | `QorRerankEngine` | qor-rag reranker model |

**Model size is config-driven, never hardcoded.** Swap from 4B to 9B to 27B by changing one TOML entry — no code change, no recompile.

---

## Repository structure

```
D:\Burn-AI\QOR4B\                   ← workspace root
├── Cargo.toml                       ← workspace manifest
├── crates\
│   ├── qora-llm\                    ← LLM inference engine (workspace member)
│   │   ├── src\                     ← gemv kernels, attention, vision, tokenizer
│   │   ├── tests\                   ← unit + integration tests
│   │   ├── benches\                 ← criterion micro-benchmarks
│   │   └── README.md                ← LLM-specific docs (formerly the root README)
│   └── qora-server\                 ← HTTP API server (workspace member)
│       ├── src\
│       │   ├── api\                 ← unified InferenceRequest, OpenAI types
│       │   ├── auth\                ← redb-backed API keys, Argon2 + HMAC, rate limit
│       │   ├── config.rs            ← TOML config schema
│       │   ├── db\                  ← redb open/init + table definitions
│       │   ├── engine\              ← ModelEngine trait + 5 task traits
│       │   ├── engines\             ← concrete: llm, tts, stt, rag
│       │   ├── error.rs             ← unified ServerError → IntoResponse
│       │   ├── history\             ← chat history with auto-embed + rerank
│       │   ├── http\                ← axum router, handlers, middleware
│       │   ├── loader.rs            ← builds Registry from TOML config
│       │   ├── metrics.rs           ← Prometheus instrumentation
│       │   ├── registry.rs          ← runtime model lookup table
│       │   ├── usage\               ← per-request usage tracking
│       │   └── bin\qora_server.rs   ← the executable entry point
│       └── tests\                   ← integration tests
├── model\                           ← weight files (configured paths in TOML)
│   ├── model.qor4b                  ← LLM weights (~3 GB for 4B, larger for 9B/27B)
│   └── tokenizer.json
├── config\
│   ├── qora-server.toml.example     ← canonical config template
│   └── qora-server.service          ← systemd unit
├── docs\
│   ├── api.md                       ← endpoint reference (auth, schemas, errors)
│   ├── deployment.md                ← operator guide (TLS, backup, ops)
│   ├── production-load-test.md      ← EPYC validation runbook
│   └── superpowers\                 ← design specs + implementation plans
└── scripts\
    ├── install.sh                   ← idempotent install for Ubuntu 24.04+
    └── soak_24h.sh                  ← memory-drift soak test
```

### External Rust crates (path-deps from qora-server)

The TTS / STT / RAG engine code lives outside this repo and is pulled in as Cargo path-deps — no source code is duplicated:

| External crate | Source location | Used by |
|---|---|---|
| `qora-tts` | `D:\Burn-AI\QORA-TTS\` | `qora-server::engines::tts::QoraTtsEngine` |
| `qora-stt-small` | `D:\Burn-AI\QORA-STT-Small\` | `qora-server::engines::stt::QoraSttEngine` |
| `qor-rag` | `C:\Projects\AI\qor-main\qor-rs\crates\qor-rag\` | `qora-server::engines::rag::Qor*Engine` |
| `qora-simd` | `C:\Projects\AI\qor-main\qor-rs\crates\qora-simd\` | transitive dep of qor-rag |

These crates are **already compiled into `qora-server.exe` at build time** — at runtime, the server is a single binary. The .exe binaries that ship with each external crate (`qora-tts.exe`, `qora-stt-small.exe`) are **standalone CLI tools** and are NOT used by the server.

### Frontend projects (separate from this repo)

```
C:\Projects\Chat\fruntend\
├── chat.jsx              ← single-file user UI (Claude Artifacts deployable)
├── admin.jsx             ← single-file admin console
├── chat\                 ← Vite project for local dev (port 5174)
└── admin\                ← Vite project for local dev (port 5173)
```

Each frontend project is independent: `cd <dir> && npm install && npm run dev`. They both talk to the qora-server HTTP API.

---

## How everything fits together

```
┌──────────────────────────────────────────────────────────────────────┐
│                       qora-server  (single binary)                    │
│                                                                       │
│  HTTP Layer                                                           │
│  ├── /v1/inference          ← unified, dispatched by `task` field     │
│  ├── /v1/chat/completions   ← OpenAI alias → ChatEngine               │
│  ├── /v1/audio/speech       ← OpenAI alias → TtsEngine                │
│  ├── /v1/audio/transcriptions ← OpenAI alias → SttEngine              │
│  ├── /v1/embeddings         ← OpenAI alias → EmbeddingEngine          │
│  ├── /v1/rerank             ← custom → RerankEngine                   │
│  ├── /v1/conversations*     ← chat history (per-user)                 │
│  ├── /v1/admin/*            ← admin console endpoints                 │
│  ├── /v1/models             ← lists registered models                 │
│  ├── /health  /metrics      ← infra                                   │
│                                                                       │
│  Auth Layer  (redb + Argon2 + HMAC + sliding-window rate limit)       │
│                                                                       │
│  Registry  (HashMap<id, Arc<dyn Engine>> built from TOML at startup)  │
│       │                                                               │
│       ▼                                                               │
│  Concrete Engines                                                     │
│  ├── LlmEngine          ──→  qora-llm crate (in this repo)            │
│  ├── QoraTtsEngine      ──→  qora-tts crate (D:\Burn-AI\QORA-TTS\)    │
│  ├── QoraSttEngine      ──→  qora-stt-small (D:\Burn-AI\QORA-STT-Small\)│
│  ├── QorEmbeddingEngine ──→  qor-rag (C:\Projects\AI\qor-main\...\)   │
│  └── QorRerankEngine    ──→  qor-rag (same)                            │
└──────────────────────────────────────────────────────────────────────┘
```

---

## Adding a new model

Three scenarios:

### 1. Different size of an existing model class (e.g. swap QOR4B → QOR9B)

Just update `qora-server.toml`:

```toml
[[models]]
id = "qora-llm"                                     # keep id stable for clients
kind = "chat"
path = "/var/lib/qora/llm/model.qor9b"              # ← new weight file
tokenizer = "/var/lib/qora/llm/tokenizer.json"
```

Restart the server. **No code change. No recompile.** This is the value of the registry approach.

### 2. Run multiple sizes side-by-side (e.g. QOR4B + QOR27B)

Add multiple `[[models]]` entries with different `id` fields:

```toml
[[models]]
id = "qora-fast"
kind = "chat"
path = "/var/lib/qora/4b/model.qor4b"
tokenizer = "/var/lib/qora/4b/tokenizer.json"

[[models]]
id = "qora-large"
kind = "chat"
path = "/var/lib/qora/27b/model.qor27b"
tokenizer = "/var/lib/qora/27b/tokenizer.json"
```

Clients then send `"model": "qora-fast"` or `"model": "qora-large"` in the request. The Registry routes to the right engine.

### 3. New model class (e.g. image generation, video gen)

Bigger work — requires:

- A new trait (`ImageGenEngine`) in `crates/qora-server/src/engine/`
- A new variant in the `Task` enum
- A new field in `Registry`
- A new arm in the `/v1/inference` task dispatch
- A wrapper crate in `crates/qora-server/src/engines/`
- An external crate (or path-dep) doing the actual inference

The trait hierarchy is designed for this — adding a new task type is additive, doesn't break existing engines.

---

## Build & Run

### Prerequisites
- Rust toolchain (edition 2024, latest stable)
- The QORA-TTS, QORA-STT-Small, qor-rag, qora-simd source trees at the paths Cargo.toml expects

### Build the server

```bash
cd D:\Burn-AI\QOR4B
cargo build --release --workspace --bin qora-server
```

Build artifacts at `target\release\qora-server.exe`.

### Run locally (development)

1. Set required environment variables:

```bash
$env:QORA_SERVER_SECRET = "any-32-byte-random-hex"
$env:QORA_ADMIN_TOKEN   = "any-admin-secret"
```

2. Edit `config\qora-server.toml.example` → save as `qora-server.toml` with paths pointing to your local model files.

3. Run:

```bash
.\target\release\qora-server.exe qora-server.toml
```

Server boots, loads all configured models eagerly, listens on the configured bind address.

### Run the frontends locally

In two separate terminals:

```bash
# Admin console (port 5173)
cd C:\Projects\Chat\fruntend\admin
npm install
npm run dev
```

```bash
# User chat panel (port 5174)
cd C:\Projects\Chat\fruntend\chat
npm install
npm run dev
```

In the admin login screen, enter `http://localhost:8080` (or wherever qora-server is listening) and the `QORA_ADMIN_TOKEN` you set above.

---

## Production Deployment

For deployment to a Linux box (e.g. EPYC server):

1. Build for Linux: `cargo build --release --workspace --bin qora-server` (or cross-compile)
2. Copy binary + model weight files to the server
3. Run the install script: `sudo ./scripts/install.sh`
4. Edit `/etc/qora/qora-server.toml` with model paths
5. Set secrets in `/etc/qora/qora.env`
6. `systemctl enable --now qora-server`
7. Validate with `docs/production-load-test.md` runbook

Full deployment guide: [`docs/deployment.md`](docs/deployment.md).

---

## API Reference

OpenAI-compatible — most clients work with just a base-URL change.

| Endpoint | Method | Purpose |
|---|---|---|
| `/v1/chat/completions` | POST | text + vision chat (streaming or non-stream) |
| `/v1/audio/speech` | POST | TTS — text in, audio bytes out |
| `/v1/audio/transcriptions` | POST | STT — audio in (multipart), text out |
| `/v1/embeddings` | POST | text → vector |
| `/v1/rerank` | POST | rank documents against a query |
| `/v1/inference` | POST | unified — set `task` field to dispatch |
| `/v1/models` | GET | list all registered models |
| `/v1/conversations` | POST/GET | create / list user's chat conversations |
| `/v1/conversations/{id}` | GET/DELETE | get conversation + messages / delete |
| `/v1/conversations/{id}/export` | GET | download user's data (with embeddings) |
| `/v1/admin/*` | various | admin console (token-protected) |
| `/health` | GET | liveness |
| `/metrics` | GET | Prometheus |

Full schemas + curl examples: [`docs/api.md`](docs/api.md). Drop-in OpenAI SDK examples (Python + Node): [`docs/examples/`](docs/examples/).

---

## Configuration (qora-server.toml)

Complete schema in [`config/qora-server.toml.example`](config/qora-server.toml.example). Key sections:

```toml
[server]            # bind address + optional TLS cert/key paths
[scheduler]         # slot count, queue size, timeouts (LLM-specific tuning)
[auth]              # admin_token env var name + redb path
[limits.default]    # default per-key rate limits (rpm, tpm)
[observability]     # Prometheus bind, log level, log format
[loading]           # eager (default) vs lazy
[history]           # which embedding model to use for chat history retrieval

[[models]]          # one per registered engine — repeat for as many as you want
id = "..."
kind = "chat" | "tts" | "stt" | "embedding" | "rerank"
path = "..."
tokenizer = "..."   # optional, depending on kind
description = "..."
```

---

## Architectural Decisions (worth remembering)

### Why a Cargo workspace with path-deps?
- `qora-llm` is a workspace member because we modified it heavily during development.
- TTS / STT / RAG are external crates because they're already complete — we just consume their public APIs.
- For deployment, only the compiled binary + model weight files matter. Source layout is a developer ergonomics question.

### Why `redb` instead of SQLite?
- QORA's promise is "pure Rust, single binary, no C deps." sqlx + libsqlite3-sys breaks that.
- `redb` is a pure-Rust embedded ACID DB used in production by Iroh and others.
- API keys, usage records, and chat history all live in one redb file with separate logical tables.

### Why HMAC index + Argon2 verify for API keys?
- Plain Argon2 verify is intentionally slow (~50–200 ms). Iterating all keys per request would be 50+ seconds at scale.
- HMAC-SHA256 with a server-side secret gives O(1) lookup; Argon2 verify on the matched record is defense-in-depth.
- Same pattern Stripe and similar APIs use.

### Why per-engine concurrency semaphores?
- TTS / STT / vision encoding can each allocate ~GB of intermediate state per request.
- Without limits, N parallel requests = N×GB transient memory → OOM.
- Defaults: vision=1 (serial), tts=2, stt=2, embed=8 (light), rerank=4. Tuneable.

### Why a separate trait per task type instead of one mega-trait?
- Each engine has different request/response shapes (chat streams; TTS returns binary; rerank needs query+docs).
- One trait would need every engine to implement every method — wasteful and confusing.
- The `ModelEngine` base trait carries the shared metadata (`id()`, `task()`, `description()`); task traits are specialized.

### Why one TOML config file lists all models, not auto-discovery?
- Auto-discovery is fragile (file system probing, ambiguous types).
- Explicit config means: operator decides what to serve, in what order, with what id, and can audit it.
- Adding a new model = add one `[[models]]` block + restart. No code change, no UI, no surprises.

---

## Status

- ✅ Core engines: LLM (chat + vision), TTS, STT, Embedding, Rerank
- ✅ Auth + admin keys + rate limiting + redb
- ✅ Chat history with auto-embed + rerank-as-context + per-user export
- ✅ Admin console (React)
- ✅ User chat panel (React, exists separately at C:\Projects\Chat\fruntend)
- ✅ Prometheus metrics, structured tracing
- ✅ TLS via rustls, systemd deployment
- ⏳ Production EPYC load test — runbook ready, awaiting hardware deployment

---

## Where to look when you forget something

| Question | File |
|---|---|
| What does each crate do? | This README, "Repository structure" section |
| How do I add a new model size? | This README, "Adding a new model" |
| What's the API surface? | [`docs/api.md`](docs/api.md) |
| How do I deploy? | [`docs/deployment.md`](docs/deployment.md) |
| What's the multi-model design rationale? | [`docs/superpowers/specs/2026-04-26-qora-multimodel-design.md`](docs/superpowers/specs/2026-04-26-qora-multimodel-design.md) |
| Why was there an earlier (now-removed) v1.0.0 tag? | The original tag was a single-model slice. We expanded scope; the real v1.0.0 covers multi-model. |
| What's in qora-llm specifically? | [`crates/qora-llm/README.md`](crates/qora-llm/README.md) |
| How does chat history retrieval work? | `crates/qora-server/src/history/retrieve.rs` (cosine similarity over stored embeddings) |
| Where are API keys stored? | redb at `auth.db_path` — never plaintext, only HMAC index + Argon2 hash |

---

## License

Apache 2.0. The base models (Qwen3.5-4B, Qwen3-TTS, etc.) are released by their respective authors under Apache 2.0.
