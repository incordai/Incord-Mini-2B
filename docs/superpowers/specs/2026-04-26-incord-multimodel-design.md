# Incord Multi-Model + Multi-Modality API Server (pre-launch v1)

**Status**: Approved (replaces single-model design)
**Date**: 2026-04-26
**Replaces**: `2026-04-25-incord-multiuser-server-design.md` (single-model v1, never shipped)
**Versioning note**: this IS v1. The earlier `v1.0.0` tag was premature — removed. The actual v1.0.0 ships when this expanded scope is complete and validated on the EPYC server.

## Goal

Ship one HTTP API server that fronts the entire Incord model family, exposing a flexible API where adding a new model is config-only (no server code changes).

## Models in scope

| Family | Existing crate | Format | Modality |
|---|---|---|---|
| **QOR2B / 3B / 4B / 8B / 27B** | this repo (`src/`) | `.incord`-family | text + vision (LLM) |
| **TTS 1.7B** | separate repo (has its own .exe + lib) | `.incord-tts` | text → audio (TTS, voice cloning, 25 voices, 10 lang) |
| **STT-Small** | separate repo (Whisper-based crate) | `.incord-stt-small` | audio → text (STT, multi-lang) |
| **RAG (Qwen3-VL-2B)** | separate repo | `.qor-rag-family` | embedding + reranking |

All four are existing **pure-Rust** crates with their own inference engines. The server's job is to wrap them behind a common trait, route requests, and handle HTTP/auth/billing/metrics.

## Non-goals

- Reimplementing TTS/STT/embed/rerank inference (we wrap existing crates)
- Real-time WebSocket streaming for audio (chunked HTTP only in this iteration)
- Per-tenant model isolation (single shared registry)
- Fine-tuning APIs

## Architecture

### Repository structure

Convert the current crate into a Cargo workspace:

```
Incord-Mini-2B/                 (workspace root)
├── Cargo.toml                  (workspace manifest)
├── crates/
│   ├── incord-llm/               (current src/* — text + vision LLM)
│   ├── incord-tts/               (dep on the TTS engine crate)
│   ├── incord-stt/               (dep on the STT engine crate)
│   ├── incord-rag/               (dep on the RAG engine crate)
│   └── incord-server/            (HTTP layer + auth + scheduler + engine trait + registry)
├── config/                     (TOML examples)
├── docs/                       (specs + plans + API + deployment)
└── tests/                      (integration)
```

The current `src/` becomes `crates/incord-llm/src/`. The CLI binary `incord` stays in incord-llm.

### Trait hierarchy

```rust
// crates/incord-server/src/engine/mod.rs

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Task {
    Chat,        // text-in (+ vision parts), text-out
    Tts,         // text-in, audio-out
    Stt,         // audio-in, text-out
    Embedding,   // text-in, vector-out
    Rerank,      // (query, docs)-in, scores-out
}

pub trait ModelEngine: Send + Sync + 'static {
    fn id(&self) -> &str;             // matches request's `model` field
    fn task(&self) -> Task;
    fn description(&self) -> &str;
}

#[async_trait]
pub trait ChatEngine: ModelEngine {
    async fn chat(&self, req: ChatRequest, sse_tx: mpsc::Sender<Event>) -> Result<()>;
}

#[async_trait]
pub trait TtsEngine: ModelEngine {
    async fn synthesize(&self, req: TtsRequest) -> Result<TtsOutput>;  // WAV bytes
}

#[async_trait]
pub trait SttEngine: ModelEngine {
    async fn transcribe(&self, req: SttRequest) -> Result<SttOutput>;  // text
}

#[async_trait]
pub trait EmbeddingEngine: ModelEngine {
    async fn embed(&self, req: EmbedRequest) -> Result<EmbedOutput>;   // Vec<Vec<f32>>
}

#[async_trait]
pub trait RerankEngine: ModelEngine {
    async fn rerank(&self, req: RerankRequest) -> Result<RerankOutput>; // [(idx, score)]
}
```

### Model registry

A `Registry` holds `HashMap<String, Arc<dyn ModelEngine>>` populated at startup from config. Plus separate downcast maps per task for fast dispatch:

```rust
pub struct Registry {
    by_id:        HashMap<String, Arc<dyn ModelEngine>>,
    chat_engines: HashMap<String, Arc<dyn ChatEngine>>,
    tts_engines:  HashMap<String, Arc<dyn TtsEngine>>,
    stt_engines:  HashMap<String, Arc<dyn SttEngine>>,
    embed_engines:  HashMap<String, Arc<dyn EmbeddingEngine>>,
    rerank_engines: HashMap<String, Arc<dyn RerankEngine>>,
}
```

Server config:

```toml
[[models]]
id = "incord-mini-2b"
kind = "chat"
path = "/var/lib/incord/models/incord-mini-2b/model.incord"
tokenizer = "/var/lib/incord/models/incord-mini-2b/tokenizer.json"

[[models]]
id = "incord-27b"
kind = "chat"
path = "/var/lib/incord/models/incord-27b/model.qor27b"
tokenizer = "/var/lib/incord/models/incord-27b/tokenizer.json"

[[models]]
id = "incord-tts-1.7b"
kind = "tts"
path = "/var/lib/incord/models/incord-tts/model.incord-tts"
tokenizer = "/var/lib/incord/models/incord-tts/tokenizer.json"
voices_dir = "/var/lib/incord/models/incord-tts/voices"

[[models]]
id = "incord-stt-small"
kind = "stt"
path = "/var/lib/incord/models/incord-stt/model.incord-stt-small"
tokenizer = "/var/lib/incord/models/incord-stt/tokenizer.json"

[[models]]
id = "incord-embed-2b"
kind = "embedding"
path = "/var/lib/incord/models/incord-rag/embed-model.bin"
tokenizer = "/var/lib/incord/models/incord-rag/embed-tokenizer.json"

[[models]]
id = "incord-rerank-2b"
kind = "rerank"
path = "/var/lib/incord/models/incord-rag/rerank-model.bin"
tokenizer = "/var/lib/incord/models/incord-rag/rerank-tokenizer.json"
```

Loading is **eager** by default (boot loads all models, ~30-90 s). **Lazy** mode optional via `[loading] mode = "lazy"`.

### Endpoints

```
GET  /health                                — liveness
GET  /metrics                               — Prometheus
GET  /v1/models                             — registry list

POST /v1/inference                          — UNIFIED, task field selects engine
POST /v1/chat/completions                   — alias: task=chat
POST /v1/audio/speech                       — alias: task=tts
POST /v1/audio/transcriptions               — alias: task=stt (multipart)
POST /v1/embeddings                         — alias: task=embedding
POST /v1/rerank                             — alias: task=rerank (custom, not OpenAI)

POST /v1/admin/keys                         — admin, unchanged
DELETE /v1/admin/keys/{id}                  — admin, unchanged
```

### Unified inference request shape

```json
POST /v1/inference
Authorization: Bearer incord_<key>

{
  "task": "chat" | "tts" | "stt" | "embedding" | "rerank",
  "model": "incord-mini-2b" | "incord-tts-1.7b" | ...,

  // chat
  "messages": [...],
  "max_tokens": 200,
  "stream": true,

  // tts
  "input": "text to synthesize",
  "voice": "luna",                  // built-in name OR base64 reference WAV
  "format": "wav" | "mp3",

  // stt
  "audio": "base64 audio bytes",
  "audio_format": "wav" | "mp3" | "opus",
  "language": "english",

  // embedding
  "input": ["text 1", "text 2"],     // single string or array

  // rerank
  "query": "search query",
  "documents": ["doc 1", "doc 2", "doc 3"],
  "top_k": 5
}
```

OpenAI-compat aliases are thin wrappers — they translate the OpenAI request shape to a unified `InferenceRequest` and forward to the same handler. Customers using the `openai` SDK get exact wire compatibility.

### Per-engine scheduling

Different engines have different optimal patterns:

| Engine | Pattern | Implementation |
|---|---|---|
| Chat | Continuous batching | Existing v1 Batcher (carries over for LlmEngine) |
| TTS | One-shot per request, tokio::spawn_blocking on a worker pool | New TtsScheduler with N workers |
| STT | Same as TTS | New SttScheduler with N workers |
| Embedding | Light, batch-friendly (multiple inputs per call already) | Inline tokio::spawn_blocking, no special scheduler |
| Rerank | Same as embedding | Inline |

The scheduler module gains a `Strategy` enum so each engine declares its scheduling needs.

## Implementation modules

| # | Module | Estimated effort |
|---|---|---|
| M7  | Workspace restructure (move src/ → crates/incord-llm/) | 2 days |
| M8  | ModelEngine + ChatEngine traits + LlmEngine wrapper around existing inference | 1 week |
| M9  | Model registry + TOML loader + endpoint dispatch | 3 days |
| M10 | Unified `/v1/inference` + OpenAI aliases | 4 days |
| M11 | TtsEngine wrapper around incord-tts | 1.5 weeks |
| M12 | SttEngine wrapper around incord-stt (audio decode for MP3/Opus via symphonia) | 1.5 weeks |
| M13 | EmbeddingEngine + RerankEngine wrappers around qor-rag | 1 week |
| M14 | Per-engine scheduling (worker pools for non-batched engines) | 1 week |
| M15 | Multi-LLM (load 27B alongside 4B; per-model rate limits + pricing) | 1 week |
| M16 | Integration tests + load tests + acceptance + v1.0.0 launch | 1 week |

**Total estimated effort: ~9-12 weeks** (one developer, full focus).

This is on top of the M1-M6 work already done (which becomes the LLM engine inside this larger architecture). M1-M6's HTTP/auth/scheduler/metrics/deploy infrastructure all carries over with minor refactor (parameterize over `&dyn Engine` instead of hardcoded LLM types).

## Acceptance for actual v1.0.0 (when launching)

- [ ] Workspace builds clean: `cargo build --release --workspace`
- [ ] LLM regression: chat_completions integration tests pass via the new LlmEngine path (no behavior change vs M1-M6)
- [ ] Multi-LLM: server with two LLM sizes both registered serves both via `/v1/chat/completions`
- [ ] TTS smoke: `POST /v1/audio/speech {model:incord-tts-1.7b, input:"hello", voice:"luna"}` returns valid 24kHz WAV
- [ ] STT smoke: `POST /v1/audio/transcriptions {model:incord-stt-small, file:<wav>}` returns transcribed text
- [ ] Embedding smoke: `POST /v1/embeddings {model:incord-embed-2b, input:["hello"]}` returns vector
- [ ] Rerank smoke: `POST /v1/rerank {model:incord-rerank-2b, query:"foo", documents:["bar","baz"]}` returns scores
- [ ] Unified endpoint: `POST /v1/inference {task:tts,...}` works identically to OpenAI alias
- [ ] OpenAI Python SDK works against `/v1/chat/completions`, `/v1/audio/speech`, `/v1/audio/transcriptions`, `/v1/embeddings`
- [ ] Production load test on EPYC: mixed traffic across all 5 task types sustains target throughput

## Future-friendliness

The trait hierarchy supports adding more engines by adding a trait + downcast map:

- `ImageGenEngine` for diffusion / image-out
- `VideoGenEngine` for video output
- `Codec`-style engines for audio reformat / re-encode

Adding a new engine type touches:
1. New trait in `engine/mod.rs`
2. New field in `Registry`
3. New variant in `Task` enum
4. New endpoint or new arm in unified handler
5. (One new wrapper crate or path-dep to integrate the inference)

Existing engines untouched.
