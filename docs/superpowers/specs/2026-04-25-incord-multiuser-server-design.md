# Incord Multi-User API Server — v1 Design

**Status**: Approved
**Date**: 2026-04-25
**Owner**: blockmandev
**Target hardware**: 1× AMD EPYC 9354 (32C/64T, AVX-512), 256 GB DDR5, 2× 3.84 TB SSD, 2 Gbit/s network, no GPU
**Timeline**: 12 weeks, single developer, full focus

## 1. Goal

Run Incord-Mini-2B as a complete, sellable, OpenAI-compatible HTTP API on one CPU server. Day-one feature set must be sufficient to charge customers without "we'll add it later" caveats. The deferred concern is multi-server scaling (load balancer + horizontal fleet) — that is *not* part of this v1, but the architecture must not preclude it.

### Non-goals (explicitly deferred)
- Multi-server / multi-node deployment
- Horizontal autoscaling
- Geographic / multi-region replication
- Fine-tuning APIs
- Audio (speech / TTS) endpoints
- Embedding endpoints (Incord is a decoder-only model)
- Function calling / tool use (base model does not support)

## 2. Capacity target

- **Comfortable**: 100–150 concurrent active decoders, ~10–20 tok/s per user, ~1500–1800 tok/s aggregate
- **Burst ceiling**: up to ~300 concurrent users at degraded per-user speed (5–7 tok/s); over-capacity admission returns 503 with `Retry-After`
- **Memory budget**: ~20 GB resident (model 3.5 GB + 64 slots × ~260 MB = ~17 GB + overhead). 230+ GB free, leaving room to grow slot count later.
- **Memory bandwidth-bound**: EPYC 9354's 12-channel DDR5 ≈ 400 GB/s sustained is the throughput limit. Increasing slot count above ~150 reduces per-user latency without increasing aggregate throughput.

## 3. Architecture

### 3.1 Component diagram

```
                  ┌─ TLS clients (HTTPS) ─┐
                  ▼                        ▼
       ┌──────────────────────────────────────┐
       │   axum HTTP server  (port 443, TLS)  │
       └──────────────────────────────────────┘
                       │ JSON / multipart
                       ▼
       ┌──────────────────────────────────────┐
       │  Auth layer  (API key → user)        │
       │  Rate limit (per-key sliding window) │
       │  Request log → SQLite                │
       └──────────────────────────────────────┘
                       │
                       ▼
       ┌──────────────────────────────────────┐
       │  Request router                      │
       │  /v1/chat/completions  (text+vision) │
       │  /v1/models     /health     /metrics │
       └──────────────────────────────────────┘
                       │
                       ▼
       ┌──────────────────────────────────────┐
       │   Scheduler queue  (FIFO, bounded)   │
       └──────────────────────────────────────┘
                       │ admit when slot free
                       ▼
       ┌──────────────────────────────────────┐
       │   Slot pool  (64 slots)              │
       │   each slot owns:                    │
       │   • HybridCache (DeltaNet+KV)        │
       │   • token history                    │
       │   • per-slot sampler state           │
       │   • SSE channel back to HTTP handler │
       └──────────────────────────────────────┘
                       │
                       ▼
       ┌──────────────────────────────────────┐
       │  Batched forward pass                │
       │  • Arc<ModelWeights> (3.5 GB shared) │
       │  • One pass processes all active     │
       │    slots together, reading weights   │
       │    once per pass (the core economy)  │
       └──────────────────────────────────────┘
                       │ N tokens out per pass
                       ▼
       ┌──────────────────────────────────────┐
       │  Token streamer                      │
       │  • Per-slot SSE chunk → HTTP         │
       │  • Token count → usage tracker       │
       └──────────────────────────────────────┘
```

### 3.2 Data flow — one chat completion request

1. Client sends `POST /v1/chat/completions` with `Authorization: Bearer incord_<key>` and JSON body.
2. axum handler parses + validates against OpenAI schema.
3. Auth middleware: API key lookup in SQLite, rate-limit check per key. Reject with 401 / 429 if necessary.
4. Convert OpenAI `messages` → Incord chat-template tokens via existing `tokenizer.rs` (chat template logic already present).
5. Build a `Session { id, cache, prompt_tokens, params, sse_tx, deadline }`.
6. Push session to scheduler queue. Reject 503 if queue depth exceeds `MAX_QUEUE`.
7. Scheduler thread admits session into a free slot. Runs prefill on this slot's `HybridCache`.
8. Slot enters decode phase. Joins next batched forward pass.
9. Each batched pass: `forward_decode_batch(weights, &mut slots) → Vec<Vec<f32>>`. Per slot: sampling, append token, send SSE event, increment usage counter.
10. Slot freed when EOS, max_tokens reached, deadline exceeded, or client disconnects.
11. Final SSE chunk `{ choices: [{ finish_reason }] }` + `[DONE]` written. Usage row written to SQLite.

## 4. Component inventory

| # | Component | Status | Estimated lines |
|---|---|---|---|
| 1 | HTTP server (axum + tokio + tower) | NEW | ~600 |
| 2 | Auth + API keys (SQLite + sqlx) | NEW | ~400 |
| 3 | Rate limiter (sliding window per key) | NEW | ~250 |
| 4 | Usage tracker / billing log | NEW | ~300 |
| 5 | OpenAI request/response types (serde) | NEW | ~500 |
| 6 | `Session` struct (per-request state) | NEW | ~200 |
| 7 | Scheduler queue + admission control | NEW | ~400 |
| 8 | Slot pool with continuous batching | NEW | ~600 |
| 9 | `forward_decode_batch` — DeltaNet path | NEW (kernel) | ~400 |
| 10 | `forward_decode_batch` — FullAttn path | NEW (kernel) | ~600 |
| 11 | Batched embedding + lm_head | NEW (kernel) | ~200 |
| 12 | f16 KV cache conversion | MODIFY `gemv.rs` | ~300 |
| 13 | Vision endpoint (image + video upload) | NEW + reuse `vision.rs` | ~400 |
| 14 | TLS termination (rustls) | Wired via axum-server | ~150 |
| 15 | Tracing / metrics (Prometheus) | NEW | ~250 |
| 16 | Config (TOML) | NEW | ~150 |
| 17 | systemd service + deployment scripts | NEW | ~150 |
| 18 | Test suite (unit + integration + load) | NEW | ~1500 |
| | **Total new code** | | **~7,000 lines** |
| | **Existing Incord core** (gemv, simd, vision, generate, tokenizer) | UNCHANGED | ~4,400 |

## 5. Memory layout (verified against existing code)

References: `gemv.rs:152` (DeltaNet state size), `gemv.rs:187` (KV cache layout), `config.rs` (architecture parameters).

| Item | Per-slot | × 64 slots | Notes |
|---|---|---|---|
| Model weights (Arc, shared) | — | 3.5 GB | Loaded once at startup |
| DeltaNet recurrent state (24 layers × 32 v_heads × 128 × 128 × 4 B) | 50 MB | 3.2 GB | Fixed regardless of context length |
| FullAttn KV cache (8 layers × 4 kv_heads × 256 head_dim × 8K context × 2 B f16 × 2 for k+v) | 130 MB | 8.3 GB | Grows with sequence length |
| Per-slot scratch (logits, hidden, etc.) | ~20 MB | 1.3 GB | |
| Auth DB + log buffers | — | ~2 GB | |
| OS + axum + tracing overhead | — | ~2 GB | |
| **Total resident** | | **~20 GB** | out of 256 GB |
| **Free for OS / page cache / growth** | | **~236 GB** | massive headroom |

## 6. The kernel work

This is the load-bearing engineering effort. Today: `forward_decode(weights, token_id, &mut cache) -> Vec<f32>` at `gemv.rs:1056`. Single token, single cache.

### 6.1 New core function

```rust
pub fn forward_decode_batch(
    weights: &ModelWeights,
    slots: &mut [&mut SlotInferenceState],
) -> Vec<Vec<f32>>  // logits per slot
```

### 6.2 What changes

- **DeltaNet batched decode** (currently `forward_deltanet_decode`): trivial parallelization. `slots.par_iter_mut().map(|s| forward_deltanet_decode(...))`. Each slot's recurrent state is independent.
- **FullAttn batched decode** (currently `forward_attn_decode`): per-slot variable `seq_len` prevents collapsing the attention step into one GEMM. Solution: par_iter the attention step over slots; batch the Q/K/V/O projections by stacking slot hidden states as GEMM rows (uses existing `gemm()`).
- **Embedding lookup + lm_head**: stack N slot tokens, batch the embedding gather + lm_head GEMM. Reuses existing `lm_head_parallel()`.
- **Per-slot RoPE positions**: each slot is at a different `seq_len`. Add a `rope_pos: usize` field to each slot. Rope lookup uses precomputed `rope_cos`/`rope_sin` tables already present.
- **f16 KV cache**: change `KvCacheEntry.k: Vec<f32>` → `Vec<f16>`, dequant on read, quant on write. ~50% memory savings, slight compute cost. Critical for fitting more slots.

### 6.3 What stays unchanged

- All AVX-512 SIMD kernels in `simd.rs` — they operate on per-slot vectors, batching wraps them.
- Vision encoder (`vision.rs`) — reused as-is, called from prefill path.
- Tokenizer, chat templates, sampling logic.
- Q4 dequantization, model loading, config parsing.
- Existing `forward_decode` and `prefill` functions retained for tooling/testing.

### 6.4 Invariant test

For all `n ≤ 64`, calling `forward_decode_batch` with N=1 slot must return logits **bitwise identical** to `forward_decode` for the same token_id and equivalent cache state. This invariant is the regression guard while batched kernels are developed.

## 7. API surface (OpenAI-compatible)

### 7.1 Implemented in v1

- `POST /v1/chat/completions` — text and vision; streaming SSE & non-streaming
- `GET /v1/models` — returns `{ data: [{ id: "incord-mini-2b", ... }] }`
- `GET /health` — liveness probe
- `GET /metrics` — Prometheus format
- `POST /v1/admin/keys` — create/revoke API key (admin-only, separate auth)

### 7.2 Vision message format (OpenAI-compatible)

```json
{
  "messages": [{
    "role": "user",
    "content": [
      { "type": "text", "text": "What's in this?" },
      { "type": "image_url", "image_url": { "url": "data:image/png;base64,..." }}
    ]
  }]
}
```

Vision encoder runs on a dedicated thread with single-concurrent-encoding admission to avoid memory spikes during the encode phase. Multiple image messages in one request are supported.

### 7.3 Sampling parameters supported

`temperature`, `top_p`, `max_tokens`, `presence_penalty`, `frequency_penalty`, `n` (up to 4), `stop` (string or array), `stream`, `user` (passed through to logs).

Incord-specific extensions (included in v1 as additive request-body fields; OpenAI clients ignore unknown fields):
- `think_budget` (integer) — defaults to 1024 unless overridden; clamped by server config
- `think` (boolean, default true) — set false to disable thinking mode (equivalent to current `--no-think` CLI flag)
- `show_think` (boolean, default false) — if true, include thinking tokens in the streamed output (default: thinking tokens are suppressed and only the final answer is returned)

## 8. Auth, billing, ops

### 8.1 Auth

- SQLite `keys` table: `(key_hash, user_id, created_at, last_used_at, is_active, rpm_limit, tpm_limit)`.
- Keys are stored as Argon2 hashes; raw keys shown to admin only at creation.
- Admin endpoint `POST /v1/admin/keys` creates/revokes; protected by separate static admin token loaded from config.

### 8.2 Rate limiting

Sliding window per key, requests/min and tokens/min, configurable per-key. On limit exceeded: 429 with `Retry-After`.

### 8.3 Usage tracking

Every completed request → SQLite `usage` table:
`(user_id, key_id, input_tokens, output_tokens, model, started_at, duration_ms, status, request_id)`.

CSV export endpoint for billing reconciliation. Stripe metered billing integration is **not** part of v1 but the schema is designed to support it — adding Stripe is a 1-day add-on after launch.

### 8.4 Observability

- `tracing` crate with JSON logs to journald.
- Prometheus metrics: `incord_requests_total`, `incord_tokens_per_second`, `incord_slot_utilization`, `incord_queue_depth`, `incord_request_duration_seconds`, `incord_active_slots`, `incord_errors_total`.
- Grafana dashboard JSON shipped in repo.

### 8.5 TLS

`rustls` via `axum-server`. Certs from Let's Encrypt (acme integration) or user-provided.

### 8.6 Deployment

- systemd unit, restart on failure, log to journald.
- Single binary, model + tokenizer + config + DB in `/opt/incord/`.
- Backups: SQLite WAL + nightly snapshot to second SSD.
- Start: `systemctl start incord-api`. Stop: graceful drain (stop accepting new requests, finish in-flight, exit).

## 9. Error handling

| Failure | Behavior |
|---|---|
| Invalid API key | 401 immediate |
| Rate limit exceeded | 429 with `Retry-After` |
| Queue full (> `MAX_QUEUE`, default 1000) | 503 with `Retry-After` |
| Per-request timeout (default 90s wall-clock) | 408, free slot, log incident |
| Client disconnects mid-stream | Detect via SSE channel close, free slot immediately |
| Inference panic (rare — `gemv.rs:1072` guards) | `catch_unwind` in slot worker, free slot, return 500 to client, log incident with full state |
| Model file missing at boot | Server fails to start (no graceful — restart needed) |
| OOM (should not happen given 256 GB) | Slot allocation rejected, return 503 |
| Disk write failure (usage log) | Log to stderr, continue serving (don't take API down for log failure) |
| Auth DB corruption | Server fails to start; restore from backup |

All recoverable failures must free their slot and not leak resources.

## 10. Testing strategy

### 10.1 Unit tests

- Each kernel verified: `forward_decode_batch` with N=1 must equal `forward_decode` exactly (bitwise identical comparison).
- Scheduler admission ordering, eviction, priority handling.
- Rate limiter under burst conditions.
- Auth flow: key creation, hashing, lookup, revocation.
- f16 KV cache round-trip accuracy (must not regress logits beyond float-equality tolerance).

### 10.2 Integration tests

- Full HTTP roundtrip with real model on a small prompt set (10 prompts).
- SSE streaming verified end-to-end.
- Multi-slot scenario: 4 concurrent simulated clients, verify token interleaving.
- Vision: image upload + completion.

### 10.3 Load tests

- Ramp 1 → 200 concurrent simulated users with a generated workload (varied prompt lengths, max_tokens).
- Record p50/p95/p99 latency, throughput, slot utilization.
- **Pass criteria**: 150 concurrent users, p95 < 30 s for a 500-token completion, no errors except expected 429/503.

### 10.4 Soak tests

- 100 concurrent users for 24 hours.
- Verify: no memory growth (RSS stable ±5%), no slot leaks, no panic accumulation, no DB lock contention.

### 10.5 Stress tests

- 500 concurrent users (over capacity).
- Verify: graceful 503s with `Retry-After`, no crashes, server recovers when load drops.

## 11. Timeline (12 weeks)

| Weeks | Work | Deliverable |
|---|---|---|
| 1–2 | HTTP layer, auth, API keys, OpenAI request/response types, SQLite | Auth + key creation + 401/429 paths working end-to-end |
| 3–4 | `Session` struct, scheduler queue, single-slot integration (uses existing `forward_decode`) | One concurrent user end-to-end via the API |
| 5–8 | Batched kernels: DeltaNet batch, FullAttn batch, batched embedding/lm_head, f16 KV | `forward_decode_batch` passes invariant tests at N=1..64 |
| 9–10 | Slot pool with continuous batching, prefill+decode mixing | 64 concurrent slots verified under load test |
| 11 | Vision endpoint, multi-image support, vision encoder concurrency control | Vision parity with CLI mode |
| 12 | Observability, deployment scripts, load testing, fixes | Production-ready single-server release |

Nothing ships partial. The whole product launches together at end of week 12.

## 12. What is sold on day 1

A complete, OpenAI-compatible API at `https://api.<your-domain>/v1/`:

- Drop-in replacement for `openai` SDK (just change `base_url` and key)
- Text + vision chat completions, streaming
- API keys + rate limits + usage records
- Production observability + TLS
- 100–150 concurrent users on one €600/mo server
- Documented endpoints + curl examples + SDK compatibility notes
- 99.5% uptime SLO achievable via systemd restart + health checks (single-node SLO ceiling)

## 13. Configuration (TOML, sketch)

```toml
[server]
bind = "0.0.0.0:443"
tls_cert = "/opt/incord/certs/fullchain.pem"
tls_key = "/opt/incord/certs/privkey.pem"

[model]
path = "/opt/incord/model.incord"
tokenizer = "/opt/incord/tokenizer.json"

[scheduler]
slot_count = 64
max_queue = 1000
max_context = 8192
request_timeout_secs = 90

[auth]
admin_token = "from_env"
db_path = "/opt/incord/auth.db"

[limits.default]
rpm = 60
tpm = 60000

[observability]
prometheus_bind = "127.0.0.1:9090"
log_level = "info"
log_format = "json"
```

## 14. Future-friendliness

While this design is single-server, the architecture does not preclude horizontal scaling later:

- Sessions are stateless across requests (matches OpenAI API model). No sticky sessions required when adding a load balancer.
- Auth + usage stored in SQLite locally; future migration to PostgreSQL is straightforward (sqlx supports both).
- Metrics endpoint follows Prometheus conventions and works with Thanos / Mimir for fleet-wide views.

When ready to scale out: stand up an L7 load balancer (nginx, Caddy, or a managed LB), run N copies of the binary on N servers behind it, share the auth DB via Postgres. No code changes to the inference layer required.

## 15. Appendix — references to existing code

- Forward decode: `src/gemv.rs:1056` — `forward_decode(weights, token_id, &mut cache) -> Vec<f32>`
- Prefill: `src/gemv.rs:1248` — `prefill(weights, token_ids, &mut cache) -> Vec<f32>`
- KV cache layout: `src/gemv.rs:164-168` — `KvCacheEntry { k: Vec<f32>, v: Vec<f32>, seq_len: usize }`
- DeltaNet state: `src/gemv.rs:133-156` — fixed `[num_v_heads, head_k_dim, head_v_dim]`
- AVX-512 kernels: `src/simd.rs` — Q4 GEMV, F16 GEMV, DeltaNet state update, fused gate+up
- Vision encoder: `src/vision.rs` — ViT + 2x2 spatial merger, image/video loading
- Tokenizer + chat template: `src/tokenizer.rs`
- Cache snapshot/restore: `src/gemv.rs:198-232` — `HybridCache::snapshot()` / `restore_from()` (already useful primitive for slot management)
- System awareness: `src/system.rs` — RAM/CPU detection (will be replaced by config-driven limits in server mode)
