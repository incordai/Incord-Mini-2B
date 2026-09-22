# Incord Multi-User Server — Master Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Ship a complete OpenAI-compatible HTTP API server for Incord-Mini-2B, capable of 100–150 concurrent users on one EPYC 9354 / 256 GB DDR5 server, as one cohesive v1 release.

**Spec:** `docs/superpowers/specs/2026-04-25-incord-multiuser-server-design.md`

**Architecture:** axum HTTP front + SQLite-backed auth/billing + 64-slot continuous-batching scheduler over `Arc<ModelWeights>` + new batched kernels in `gemv_batch.rs`. Existing Incord core (`gemv.rs`, `simd.rs`, `vision.rs`, `tokenizer.rs`, `generate.rs`) stays unchanged; server is added as a new binary target sharing the existing library.

**Tech Stack:**
- Rust 2024 edition (existing)
- axum 0.8 + tokio 1 (full features) + tower 0.5 + tower-http
- sqlx 0.8 (SQLite, runtime-tokio-rustls)
- argon2 0.5 (password hashing for API keys)
- rustls 0.23 + axum-server
- tracing 0.1 + tracing-subscriber
- prometheus 0.13
- uuid 1
- serde 1, serde_json 1
- async-trait 0.1
- futures 0.3
- bytes 1
- base64 0.22 (for vision data URLs)
- reqwest 0.12 (test client)
- tempfile 3 (test fixtures)

---

## Module Map

This master plan sequences six module plans. Modules build on each other; do not start a module until its prerequisites are complete. **The product launches only when all six modules are complete.** No partial release.

```
M1 ─→ M2 ─→ M3 ─→ M4 ─→ M5 ─→ M6
HTTP  Auth  Sess  Kernel  Slot  Vision  Obs+Deploy
                                     (M5 and M6 partly parallel)
```

| # | Module | Plan file | Est. duration | Status |
|---|---|---|---|---|
| M1 | HTTP foundation + OpenAI types + config + binary | `2026-04-25-incord-m1-http-foundation.md` | Week 1 | pending |
| M2 | Auth, API keys, rate limiting, usage tracking | `2026-04-25-incord-m2-auth-billing.md` | Week 2 | pending |
| M3 | Session + scheduler + serial-parallel decode (single-slot path) | `2026-04-25-incord-m3-session-serial.md` | Weeks 3–4 | pending |
| M4 | Batched kernels (`gemv_batch.rs`) — DeltaNet batch, FullAttn batch, batched embed/lm_head, f16 KV | `2026-04-25-incord-m4-batched-kernels.md` | Weeks 5–8 | pending |
| M5 | Slot pool with continuous batching, prefill+decode mixing, scheduling policy | `2026-04-25-incord-m5-slot-pool.md` | Weeks 9–10 | pending |
| M6 | Vision endpoint + observability + deployment + load-test passes | `2026-04-25-incord-m6-vision-ops-launch.md` | Weeks 11–12 | pending |

---

## File Structure (final state at v1 launch)

```
src/
  main.rs                    [unchanged — CLI entry stays for batch use]
  lib.rs                     [+ new server module exports]
  config.rs                  [unchanged]
  gemv.rs                    [unchanged — keep existing forward_decode/prefill]
  gemv_batch.rs              [NEW — batched forward functions]
  simd.rs                    [unchanged — kernels operate on per-slot vectors]
  generate.rs                [unchanged — CLI generation path]
  tokenizer.rs               [unchanged]
  vision.rs                  [unchanged]
  save.rs                    [unchanged]
  convert.rs                 [unchanged]
  system.rs                  [unchanged]
  gpu_loader.rs              [unchanged]
  gpu_inference.rs           [unchanged]
  bin/
    incord_server.rs           [NEW — server main entry point]
  server/
    mod.rs                   [NEW — module root]
    config.rs                [NEW — TOML config loading]
    http/
      mod.rs                 [NEW]
      router.rs              [NEW — axum routes]
      handlers/
        chat.rs              [NEW — /v1/chat/completions]
        models.rs            [NEW — /v1/models]
        admin.rs             [NEW — /v1/admin/keys]
        health.rs            [NEW — /health]
        metrics.rs           [NEW — /metrics]
      sse.rs                 [NEW — SSE streaming helpers]
      tls.rs                 [NEW — rustls config loader]
    openai/
      mod.rs                 [NEW]
      request.rs             [NEW — OpenAI request types]
      response.rs            [NEW — OpenAI response types]
      streaming.rs           [NEW — SSE chunk types]
      vision.rs              [NEW — image_url parsing]
    auth/
      mod.rs                 [NEW]
      keys.rs                [NEW — API key model + hashing]
      middleware.rs          [NEW — axum auth middleware]
      ratelimit.rs           [NEW — sliding-window limiter]
      admin.rs               [NEW — admin token auth]
    db/
      mod.rs                 [NEW — sqlx pool + migrations]
      migrations/
        0001_initial.sql     [NEW]
    usage/
      mod.rs                 [NEW]
      tracker.rs             [NEW — usage logging]
    session/
      mod.rs                 [NEW]
      session.rs             [NEW — Session struct]
      sampler.rs             [NEW — sampling state per session]
    scheduler/
      mod.rs                 [NEW]
      queue.rs               [NEW — bounded request queue]
      slot.rs                [NEW — Slot struct]
      pool.rs                [NEW — slot pool]
      batcher.rs             [NEW — continuous batching loop]
    metrics/
      mod.rs                 [NEW — Prometheus collectors]
    error.rs                 [NEW — server-wide error type]

tests/
  integration/
    http_basic.rs            [NEW]
    auth_flow.rs             [NEW]
    chat_completions.rs      [NEW]
    streaming.rs             [NEW]
    vision.rs                [NEW]
    rate_limit.rs            [NEW]
  load/
    ramp_to_150.rs           [NEW]
    soak_24h.rs              [NEW]

config/
  incord-server.toml.example   [NEW]
  prometheus.yml.example     [NEW]
  incord-server.service        [NEW — systemd unit]

docs/
  api.md                     [NEW — endpoint documentation]
  deployment.md              [NEW — install / ops guide]
  examples/
    curl/                    [NEW]
    openai-python/           [NEW]
    openai-node/             [NEW]
```

---

## Cross-Module Conventions

### Error type

Every module returns `Result<T, server::error::ServerError>` where `ServerError` is an enum with variants for each failure category (Auth, RateLimit, QueueFull, Timeout, Inference, Db, Internal). It implements `IntoResponse` for axum so handlers can `?` errors into HTTP responses with correct status codes.

### Logging

Every request gets a `request_id: Uuid`. All structured logs include this ID. Log levels: `error` for unexpected failures, `warn` for client errors (4xx), `info` for request start/end, `debug` for inference details, `trace` for token-level events (off in production).

### Test layout

- Unit tests next to code: `#[cfg(test)] mod tests { ... }` at the bottom of each file.
- Integration tests: `tests/integration/*.rs` — spin up a real server on a random port, hit it with `reqwest`.
- All tests use a small fixture model OR mock weights — full 3.5 GB model is only used in `tests/load/`.

### Commit policy

Every task ends with a commit. Commits are atomic — one logical change per commit. Each commit must leave the build green (`cargo check --all-targets` passes) even if not all features are complete.

### Branching

Work on a `feature/incord-server` branch off main. Each module gets a sub-branch (`feature/incord-server-m1-http`, etc.) merged back to `feature/incord-server` when complete. Final v1 launch merges `feature/incord-server` to `main`.

---

## Module Plan Headers

Each module plan opens with:
- The list of files it creates/modifies (subset of the global file structure above)
- Its prerequisites (what must be merged before starting)
- Its acceptance test (the integration/invariant test that must pass before the module is "done")
- Its bite-sized TDD tasks

---

## Module Acceptance Tests (each must pass before its module is considered done)

| Module | Acceptance test |
|---|---|
| M1 | `cargo test --test http_basic` — server starts on random port, `/health` returns 200, `/v1/models` returns mock model list, malformed JSON returns 400 |
| M2 | `cargo test --test auth_flow` — admin endpoint creates a key, valid key passes auth middleware, invalid returns 401, rate-limited key returns 429, usage row written to DB |
| M3 | `cargo test --test chat_completions -- single_slot` — POST /v1/chat/completions with stream=false against a tiny mock model returns valid completion; with stream=true returns valid SSE chunks ending in `[DONE]` |
| M4 | `cargo test --test gemv_batch_invariants` — for each layer type, `forward_decode_batch` with N=1 returns logits bitwise-identical to `forward_decode` over a battery of token sequences and cache states |
| M5 | `cargo test --test slot_pool_concurrency` — 64 concurrent simulated sessions complete without slot leaks, scheduler maintains FIFO admission, queue overflow returns 503 |
| M6 | (a) `cargo test --test vision` — image upload + completion roundtrip works; (b) `cargo run --bin load_test -- --concurrency 150 --duration 600` — 150 concurrent over 10 min hits p95 < 30s for 500-token completion |

---

## Final Launch Checklist (gate to merging `feature/incord-server` → `main`)

- [ ] All module acceptance tests pass on a clean checkout
- [ ] `cargo clippy --all-targets -- -D warnings` clean
- [ ] `cargo fmt --check` clean
- [ ] Soak test: 100 concurrent for 24 hours, RSS stable ±5%, no panics
- [ ] Stress test: 500 concurrent for 10 min, graceful 503s, server recovers
- [ ] Production config example reviewed and committed
- [ ] systemd unit file tested on a clean Ubuntu 24.04 install
- [ ] API documentation published (curl + Python + Node examples)
- [ ] TLS certificate procurement documented
- [ ] Backup procedure documented and tested (auth.db restore)
- [ ] Admin token rotation procedure documented
- [ ] Rate-limit defaults reviewed
- [ ] Pricing/billing integration plan documented (out of v1 scope but should be ready to wire after launch)

When all boxes ticked → merge → tag `v1.0.0` → deploy → start selling.
