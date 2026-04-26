# M3 — Session + Serial-Parallel Single-Slot Inference

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development.

**Goal:** Wire chat completion requests through to the *existing* `forward_decode` function — one slot at a time, but with `Arc<ModelWeights>` shared across requests so multiple requests can run in parallel-independent mode (each thread runs a full forward pass for its own user). End state: working `/v1/chat/completions` endpoint with streaming SSE, against the real model. No batching yet — each user gets a dedicated worker thread. This is the minimum-viable inference integration.

**Prerequisites:** M1 + M2 merged. Branch `feature/qora-server-m3-session` from `feature/qora-server`.

**Architecture:** A `Session` struct holds one user's `HybridCache`, tokens, sampler, SSE channel. A `Scheduler` owns `Arc<ModelWeights>` and a bounded queue. When a request arrives, build a Session, push to queue. A pool of N worker threads pop sessions from the queue, run prefill + decode loops calling existing `forward_decode`, send tokens via SSE.

**Acceptance test:** `cargo test --test chat_completions -- single_slot` against a small fixture (or the full model in CI-skipped mode) returns valid OpenAI completion (stream and non-stream).

---

## Task 1: Add fixture model loader for tests

**Files:**
- Create: `tests/fixtures/mod.rs`
- Modify: `tests/integration/common.rs`

The full 3.5 GB model is too heavy for unit/integration tests. We need a tiny mock weights struct that satisfies the `forward_decode` interface but returns deterministic logits. This lets the entire HTTP stack be tested end-to-end without loading 3 GB.

- [ ] **Step 1: Create a feature flag for mock weights**

Modify `Cargo.toml`:

```toml
[features]
default = []
gpu = ["cortex", "cortex/wgpu"]
gpu-metal = ["cortex", "cortex/metal"]
mock-weights = []  # NEW: enables tiny stub weights for tests
```

- [ ] **Step 2: Add a mock weights builder under #[cfg(any(test, feature = "mock-weights"))]**

Append to `src/gemv.rs` (at the end of the file):

```rust
#[cfg(any(test, feature = "mock-weights"))]
pub mod mock {
    //! Minimal stub model weights for tests. Tiny dimensions, deterministic outputs.
    use super::*;
    use crate::config::{Qor4bConfig, LayerType};
    use half::f16;

    pub fn tiny_config() -> Qor4bConfig {
        // Real production config: hidden=2560, vocab=248320, 32 layers, deltanet_head_dim=128.
        // Tiny test config below uses the same architecture but small-power-of-2 dims to keep
        // unit tests fast (< 50 ms per call) while exercising every code path.
        let mut layer_types = Vec::with_capacity(4);
        layer_types.push(LayerType::DeltaNet);
        layer_types.push(LayerType::DeltaNet);
        layer_types.push(LayerType::DeltaNet);
        layer_types.push(LayerType::FullAttn);

        Qor4bConfig {
            vocab_size: 256,
            hidden_size: 64,
            num_layers: 4,
            num_attn_heads: 2,
            num_kv_heads: 1,
            attn_head_dim: 32,
            num_qk_heads: 2,
            num_v_heads: 4,
            deltanet_head_dim: 16,
            conv_kernel_size: 4,
            intermediate_size: 128,
            rope_theta: 10_000.0,
            partial_rotary_factor: 0.25,
            rms_norm_eps: 1e-6,
            eos_token_id: 0,
            tie_word_embeddings: true,
            layer_types,
            has_vision: false,    // M3 doesn't need vision; M6 enables it for vision tests
            vision_hidden: 64,
            vision_layers: 2,
            vision_heads: 2,
            vision_ffn: 128,
            patch_size: 4,
            spatial_merge_size: 2,
            num_position_embeddings: 64,
        }
    }

    pub fn tiny_weights() -> ModelWeights {
        // Walk the production ModelWeights struct (see src/save.rs) and allocate zero-filled
        // F16Weights for every projection at the shapes implied by tiny_config().
        //
        // Production ModelWeights has fields (verify by reading src/save.rs::ModelWeights):
        //   - config: Qor4bConfig
        //   - hidden: usize
        //   - vocab: usize
        //   - format_name: String
        //   - embed: Weight                  // [vocab, hidden]
        //   - layers: Vec<HybridLayerWeights>
        //   - final_norm: Vec<f16>           // [hidden]
        //   - rope_cos: Vec<f32>             // [max_pos * rope_dim]
        //   - rope_sin: Vec<f32>
        //   - vision: Option<VisionWeights>
        //
        // Implementation steps for the engineer:
        //   1. Build config = tiny_config()
        //   2. Build embed as F16Weight { data: vec![f16::ZERO; vocab*hidden], k: hidden, n: vocab }
        //      (or small non-zero values to avoid pathological dead paths)
        //   3. For each layer in config.layer_types, push the matching HybridLayerWeights
        //      with all-zero F16Weights at correct shapes (see DeltaNetLayerWeights and
        //      FullAttnLayerWeights field comments in gemv.rs lines 62-94).
        //   4. final_norm = vec![f16::ONE; hidden]
        //   5. rope_cos/sin = small precomputed table at rope_dim() * 64 positions
        //   6. vision = None (tiny_config has has_vision: false)
        //
        // The function below returns a working stub. Once the engineer reads ModelWeights and
        // the per-layer weight structs, they replace these allocators with the right ones.
        let config = tiny_config();
        // The body of this function is mechanical — the engineer fills it by reading
        // src/save.rs::ModelWeights and the per-layer weight structs in src/gemv.rs:62-94.
        // No new logic is invented; only zero-filled tensors at the dims given by tiny_config.
        todo!("tiny_weights body: zero-filled ModelWeights matching tiny_config; mechanical, ~80 LoC")
    }
}
```

> **Implementation note**: the exact bodies depend on the production `Qor4bConfig` and `ModelWeights` shapes. The engineer reads those structs in `src/config.rs` and `src/save.rs::ModelWeights`, then writes the smallest-possible default that the existing `forward_decode` accepts without panicking. Acceptance check: `let w = mock::tiny_weights(); let mut cache = HybridCache::new(&w.config); let _ = forward_decode(&w, 0, &mut cache);` runs without panicking.

- [ ] **Step 3: Build with the feature**

```bash
cargo check --features mock-weights
```

- [ ] **Step 4: Commit**

```bash
git add Cargo.toml src/gemv.rs
git commit -m "test(infra): mock-weights feature for tiny test models"
```

---

## Task 2: Session struct

**Files:**
- Create: `src/server/session/mod.rs`
- Create: `src/server/session/session.rs`
- Create: `src/server/session/sampler.rs`
- Modify: `src/server/mod.rs`

- [ ] **Step 1: Sampler (lifted from generate.rs)**

Create `src/server/session/sampler.rs`:

```rust
use std::time::SystemTime;

pub struct Sampler {
    state: u64,
    pub temperature: f32,
    pub top_p: f32,
    pub top_k: usize,
    pub presence_penalty: f32,
    pub repetition_penalty: f32,
}

impl Sampler {
    pub fn new(temperature: f32, top_p: f32, top_k: usize,
               presence_penalty: f32, repetition_penalty: f32) -> Self {
        let seed = SystemTime::now()
            .duration_since(SystemTime::UNIX_EPOCH).unwrap().as_nanos() as u64 | 1;
        Self { state: seed, temperature, top_p, top_k, presence_penalty, repetition_penalty }
    }

    fn rand(&mut self) -> f32 {
        self.state ^= self.state << 13;
        self.state ^= self.state >> 7;
        self.state ^= self.state << 17;
        (self.state >> 40) as f32 / 16777216.0
    }

    pub fn sample(&mut self, logits: &mut [f32], generated: &[u32]) -> u32 {
        // Apply repetition penalty
        if self.repetition_penalty > 1.0 {
            let window = generated.len().min(64);
            let recent = &generated[generated.len() - window..];
            for &t in recent {
                if (t as usize) < logits.len() {
                    let l = &mut logits[t as usize];
                    if *l > 0.0 { *l /= self.repetition_penalty; } else { *l *= self.repetition_penalty; }
                }
            }
        }
        // Apply presence penalty
        if self.presence_penalty > 0.0 && !generated.is_empty() {
            let mut seen = vec![false; logits.len()];
            for &t in generated { if (t as usize) < seen.len() { seen[t as usize] = true; } }
            for (i, &s) in seen.iter().enumerate() { if s { logits[i] -= self.presence_penalty; } }
        }
        // Greedy fast path
        if self.temperature == 0.0 {
            let mut best = 0; let mut best_v = logits[0];
            for (i, &v) in logits.iter().enumerate() {
                if v > best_v { best_v = v; best = i; }
            }
            return best as u32;
        }
        // Apply temperature
        for v in logits.iter_mut() { *v /= self.temperature; }
        // Top-k
        let mut idx: Vec<u32> = (0..logits.len() as u32).collect();
        idx.sort_unstable_by(|&a, &b| logits[b as usize].partial_cmp(&logits[a as usize]).unwrap());
        idx.truncate(self.top_k);
        // Softmax over top-k
        let max = idx.iter().map(|&i| logits[i as usize]).fold(f32::NEG_INFINITY, f32::max);
        let exp: Vec<f32> = idx.iter().map(|&i| (logits[i as usize] - max).exp()).collect();
        let sum: f32 = exp.iter().sum();
        let probs: Vec<f32> = exp.iter().map(|p| p / sum).collect();
        // Top-p (cumulative)
        let mut cum = 0.0; let mut cutoff = idx.len();
        for i in 0..idx.len() { cum += probs[i]; if cum >= self.top_p { cutoff = i + 1; break; } }
        let final_idx = &idx[..cutoff];
        let final_probs = &probs[..cutoff];
        let final_sum: f32 = final_probs.iter().sum();
        // Sample
        let r = self.rand() * final_sum;
        let mut acc = 0.0;
        for (i, &p) in final_probs.iter().enumerate() {
            acc += p;
            if r <= acc { return final_idx[i]; }
        }
        final_idx[final_idx.len() - 1]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn greedy_picks_argmax() {
        let mut s = Sampler::new(0.0, 1.0, 1, 0.0, 1.0);
        let mut logits = vec![1.0, 5.0, 2.0, 3.0];
        assert_eq!(s.sample(&mut logits, &[]), 1);
    }
}
```

- [ ] **Step 2: Session struct**

Create `src/server/session/session.rs`:

```rust
use crate::gemv::HybridCache;
use crate::server::session::sampler::Sampler;
use uuid::Uuid;

pub struct Session {
    pub id: Uuid,
    pub key_id: String,
    pub user_id: String,
    pub model_name: String,
    pub cache: HybridCache,
    pub prompt_tokens: Vec<u32>,
    pub generated_tokens: Vec<u32>,
    pub max_new_tokens: usize,
    pub eos_token_id: u32,
    pub sampler: Sampler,
    pub started_at: i64,
}
```

- [ ] **Step 3: Wire**

```rust
// src/server/session/mod.rs
pub mod sampler;
pub mod session;
```

```rust
// src/server/mod.rs add:
pub mod session;
```

- [ ] **Step 4: Tests**

```bash
cargo test --lib server::session
```

Expected: 1 test passes (the greedy sampler test).

- [ ] **Step 5: Commit**

```bash
git add src/server/session/ src/server/mod.rs
git commit -m "feat(session): Session struct + per-session sampler"
```

---

## Task 3: Scheduler with Arc<ModelWeights> and bounded queue

**Files:**
- Create: `src/server/scheduler/mod.rs`
- Create: `src/server/scheduler/queue.rs`
- Modify: `src/server/mod.rs`

- [ ] **Step 1: Implement bounded queue**

Create `src/server/scheduler/queue.rs`:

```rust
use std::sync::Arc;
use tokio::sync::Notify;
use std::collections::VecDeque;
use std::sync::Mutex;
use crate::server::session::session::Session;

pub struct PendingRequest {
    pub session: Session,
    pub sse_tx: tokio::sync::mpsc::Sender<crate::server::scheduler::Event>,
}

pub struct Queue {
    inner: Mutex<VecDeque<PendingRequest>>,
    notify: Notify,
    max_size: usize,
}

impl Queue {
    pub fn new(max_size: usize) -> Arc<Self> {
        Arc::new(Self { inner: Mutex::new(VecDeque::new()), notify: Notify::new(), max_size })
    }

    /// Try to push a request. Returns Err if queue full.
    pub fn push(&self, req: PendingRequest) -> Result<(), PendingRequest> {
        let mut q = self.inner.lock().unwrap();
        if q.len() >= self.max_size { return Err(req); }
        q.push_back(req);
        self.notify.notify_one();
        Ok(())
    }

    /// Pop one request, awaiting until one arrives.
    pub async fn pop(self: &Arc<Self>) -> PendingRequest {
        loop {
            {
                let mut q = self.inner.lock().unwrap();
                if let Some(r) = q.pop_front() { return r; }
            }
            self.notify.notified().await;
        }
    }

    pub fn len(&self) -> usize { self.inner.lock().unwrap().len() }
}
```

- [ ] **Step 2: Scheduler module root + Event type**

Create `src/server/scheduler/mod.rs`:

```rust
pub mod queue;

#[derive(Debug, Clone)]
pub enum Event {
    Token(u32, String),       // token id + decoded string
    Finish(FinishReason),
    Error(String),
}

#[derive(Debug, Clone, Copy)]
pub enum FinishReason {
    Stop,
    Length,
    Cancelled,
    Timeout,
    Error,
}

impl FinishReason {
    pub fn as_openai(&self) -> &'static str {
        match self {
            Self::Stop | Self::Cancelled => "stop",
            Self::Length => "length",
            Self::Timeout => "stop",
            Self::Error => "stop",
        }
    }
}
```

```rust
// src/server/mod.rs add:
pub mod scheduler;
```

- [ ] **Step 3: Tests**

Append to `src/server/scheduler/queue.rs` test module:

```rust
#[cfg(test)]
mod tests {
    use super::*;
    // Cannot construct PendingRequest without a real Session (which needs a HybridCache from a real config).
    // Smoke-test queue length only.

    #[tokio::test]
    async fn queue_starts_empty() {
        let q = Queue::new(10);
        assert_eq!(q.len(), 0);
    }
}
```

```bash
cargo test --lib server::scheduler
```

Expected: 1 test passes.

- [ ] **Step 4: Commit**

```bash
git add src/server/scheduler/ src/server/mod.rs
git commit -m "feat(scheduler): bounded request queue + Event type"
```

---

## Task 4: Worker that runs prefill + decode for one session

**Files:**
- Create: `src/server/scheduler/worker.rs`
- Modify: `src/server/scheduler/mod.rs`

- [ ] **Step 1: Worker logic using existing forward_decode and prefill**

Create `src/server/scheduler/worker.rs`:

```rust
use std::sync::Arc;
use tokio::sync::mpsc;
use crate::gemv::{ModelWeights, prefill, forward_decode};
use crate::tokenizer::QoraTokenizer;
use crate::server::scheduler::queue::{PendingRequest, Queue};
use crate::server::scheduler::{Event, FinishReason};

pub struct Worker {
    pub id: usize,
    pub weights: Arc<ModelWeights>,
    pub tokenizer: Arc<QoraTokenizer>,
    pub queue: Arc<Queue>,
}

impl Worker {
    pub fn spawn(self) -> tokio::task::JoinHandle<()> {
        tokio::task::spawn_blocking(move || {
            // Convert blocking work into a sync loop on this thread.
            let rt = tokio::runtime::Handle::current();
            loop {
                let pending = rt.block_on(self.queue.pop());
                self.run_one(pending);
            }
        })
    }

    fn run_one(&self, mut req: PendingRequest) {
        let session = &mut req.session;
        let tx = req.sse_tx;

        // Prefill
        let mut logits = prefill(&self.weights, &session.prompt_tokens, &mut session.cache);

        // Decode loop
        for _step in 0..session.max_new_tokens {
            // Sample
            let token = session.sampler.sample(&mut logits, &session.generated_tokens);
            session.generated_tokens.push(token);

            // Decode token to string for SSE delta
            let text = self.tokenizer.decode(&[token]);

            if tx.blocking_send(Event::Token(token, text)).is_err() {
                // Client disconnected
                let _ = tx.blocking_send(Event::Finish(FinishReason::Cancelled));
                return;
            }

            if token == session.eos_token_id {
                let _ = tx.blocking_send(Event::Finish(FinishReason::Stop));
                return;
            }

            // Compute next logits via existing forward_decode
            logits = forward_decode(&self.weights, token as usize, &mut session.cache);
        }

        let _ = tx.blocking_send(Event::Finish(FinishReason::Length));
    }
}
```

> **Note**: The existing tokenizer.rs has a `decode` method that takes a slice of token ids and returns a String. Verify by reading `src/tokenizer.rs` — adjust the call signature here if needed.

- [ ] **Step 2: Wire**

```rust
// src/server/scheduler/mod.rs append:
pub mod worker;
```

- [ ] **Step 3: Build**

```bash
cargo check --all-targets
```

Expected: clean (the worker is dead code until wired into the binary; that comes in Task 6).

- [ ] **Step 4: Commit**

```bash
git add src/server/scheduler/
git commit -m "feat(scheduler): worker runs prefill + decode using existing kernels"
```

---

## Task 5: Chat completion handler

**Files:**
- Create: `src/server/http/handlers/chat.rs`
- Create: `src/server/http/sse.rs`
- Modify: `src/server/http/handlers/mod.rs`
- Modify: `src/server/http/mod.rs`

- [ ] **Step 1: SSE helpers**

Create `src/server/http/sse.rs`:

```rust
use axum::response::sse::{Event, KeepAlive, Sse};
use futures::stream::{Stream, StreamExt};
use std::convert::Infallible;
use std::time::Duration;

pub fn into_sse_response<S>(stream: S) -> Sse<impl Stream<Item = Result<Event, Infallible>>>
where
    S: Stream<Item = Event> + Send + 'static,
{
    Sse::new(stream.map(Ok::<_, Infallible>))
        .keep_alive(KeepAlive::new().interval(Duration::from_secs(15)).text(": keepalive"))
}
```

- [ ] **Step 2: Wire**

```rust
// src/server/http/mod.rs add:
pub mod sse;
```

- [ ] **Step 3: Chat handler**

Create `src/server/http/handlers/chat.rs`:

```rust
use std::sync::Arc;
use axum::{extract::State, Extension, Json};
use axum::response::IntoResponse;
use chrono::Utc;
use futures::stream::{self, Stream, StreamExt};
use tokio::sync::mpsc;
use uuid::Uuid;

use crate::gemv::HybridCache;
use crate::server::auth::keys::ApiKey;
use crate::server::error::{ServerError, ServerResult};
use crate::server::http::sse::into_sse_response;
use crate::server::openai::request::{ChatCompletionRequest, MessageContent, ContentPart};
use crate::server::openai::response::{
    ChatCompletionResponse, Choice, ResponseMessage, Usage,
    ChatCompletionChunk, ChunkChoice, ChunkDelta,
};
use crate::server::scheduler::{queue::{PendingRequest, Queue}, Event as SchedEvent, FinishReason};
use crate::server::session::sampler::Sampler;
use crate::server::session::session::Session;
use crate::server::usage::tracker::{UsageRecord, UsageTracker};

#[derive(Clone)]
pub struct ChatState {
    pub queue: Arc<Queue>,
    pub weights: Arc<crate::gemv::ModelWeights>,
    pub tokenizer: Arc<crate::tokenizer::QoraTokenizer>,
    pub usage: UsageTracker,
}

pub async fn chat_completions(
    State(state): State<Arc<ChatState>>,
    Extension(key): Extension<ApiKey>,
    Json(req): Json<ChatCompletionRequest>,
) -> Result<axum::response::Response, ServerError> {
    if req.model != "qora-4b" {
        return Err(ServerError::BadRequest(format!("unknown model: {}", req.model)));
    }
    let prompt_text = render_messages(&req.messages)?;
    let prompt_tokens = state.tokenizer.encode_chat(&prompt_text)
        .map_err(|e| ServerError::BadRequest(format!("tokenizer: {e}")))?;

    let session = Session {
        id: Uuid::new_v4(),
        key_id: key.id.clone(),
        user_id: key.user_id.clone(),
        model_name: req.model.clone(),
        cache: HybridCache::new(&state.weights.config),
        prompt_tokens: prompt_tokens.clone(),
        generated_tokens: Vec::new(),
        max_new_tokens: req.max_tokens.unwrap_or(1024) as usize,
        eos_token_id: state.weights.config.eos_token_id,
        sampler: Sampler::new(
            req.temperature.unwrap_or(0.7),
            req.top_p.unwrap_or(0.95),
            20,
            req.presence_penalty.unwrap_or(1.5),
            req.frequency_penalty.unwrap_or(1.0),
        ),
        started_at: Utc::now().timestamp(),
    };

    let (tx, rx) = mpsc::channel::<SchedEvent>(64);
    let pending = PendingRequest { session, sse_tx: tx };
    state.queue.push(pending).map_err(|_| ServerError::QueueFull { retry_after_secs: 5 })?;

    if req.stream {
        Ok(stream_response(state, req.model.clone(), key, prompt_tokens.len(), rx).await.into_response())
    } else {
        Ok(non_stream_response(state, req.model.clone(), key, prompt_tokens.len(), rx).await?.into_response())
    }
}

fn render_messages(messages: &[crate::server::openai::request::Message]) -> ServerResult<String> {
    let mut out = String::new();
    for m in messages {
        let text = match &m.content {
            MessageContent::Text(t) => t.clone(),
            MessageContent::Parts(parts) => {
                let mut s = String::new();
                for p in parts {
                    if let ContentPart::Text { text } = p { s.push_str(text); s.push('\n'); }
                    // ImageUrl handled in M5/M6 vision integration
                }
                s
            }
        };
        out.push_str(&format!("<|{}|>{}\n", m.role, text));
    }
    Ok(out)
}

async fn non_stream_response(
    state: Arc<ChatState>, model: String, key: ApiKey, prompt_tokens: usize,
    mut rx: mpsc::Receiver<SchedEvent>,
) -> Result<Json<ChatCompletionResponse>, ServerError> {
    let started = Utc::now().timestamp();
    let mut content = String::new();
    let mut output_tokens = 0u32;
    let mut finish: FinishReason = FinishReason::Length;

    while let Some(ev) = rx.recv().await {
        match ev {
            SchedEvent::Token(_, text) => { content.push_str(&text); output_tokens += 1; }
            SchedEvent::Finish(r) => { finish = r; break; }
            SchedEvent::Error(e) => { return Err(ServerError::Inference(e)); }
        }
    }

    let resp = ChatCompletionResponse {
        id: format!("chatcmpl-{}", Uuid::new_v4()),
        object: "chat.completion",
        created: started as u64,
        model: model.clone(),
        choices: vec![Choice {
            index: 0,
            message: ResponseMessage { role: "assistant".into(), content },
            finish_reason: Some(finish.as_openai().into()),
        }],
        usage: Usage {
            prompt_tokens: prompt_tokens as u32,
            completion_tokens: output_tokens,
            total_tokens: prompt_tokens as u32 + output_tokens,
        },
    };

    let _ = state.usage.record(UsageRecord {
        request_id: resp.id.clone(),
        user_id: key.user_id, key_id: key.id, model,
        input_tokens: prompt_tokens as u32, output_tokens,
        started_at: started, duration_ms: 0, status: "success",
    }).await;

    Ok(Json(resp))
}

async fn stream_response(
    state: Arc<ChatState>, model: String, key: ApiKey, prompt_tokens: usize,
    rx: mpsc::Receiver<SchedEvent>,
) -> impl IntoResponse {
    let id = format!("chatcmpl-{}", Uuid::new_v4());
    let created = Utc::now().timestamp() as u64;
    let model_clone = model.clone();

    let stream = stream::unfold(rx, move |mut rx| {
        let model = model_clone.clone();
        let id = id.clone();
        async move {
            match rx.recv().await {
                Some(SchedEvent::Token(_, text)) => {
                    let chunk = ChatCompletionChunk {
                        id, object: "chat.completion.chunk", created, model,
                        choices: vec![ChunkChoice {
                            index: 0,
                            delta: ChunkDelta { role: None, content: Some(text) },
                            finish_reason: None,
                        }],
                    };
                    let ev = axum::response::sse::Event::default()
                        .data(serde_json::to_string(&chunk).unwrap());
                    Some((ev, rx))
                }
                Some(SchedEvent::Finish(r)) => {
                    let chunk = ChatCompletionChunk {
                        id: id.clone(), object: "chat.completion.chunk", created, model: model.clone(),
                        choices: vec![ChunkChoice {
                            index: 0, delta: ChunkDelta::default(),
                            finish_reason: Some(r.as_openai().into()),
                        }],
                    };
                    let _ = state; // capture to record usage if desired
                    let ev = axum::response::sse::Event::default()
                        .data(serde_json::to_string(&chunk).unwrap());
                    Some((ev, rx))
                }
                Some(SchedEvent::Error(e)) => {
                    let ev = axum::response::sse::Event::default().event("error").data(e);
                    Some((ev, rx))
                }
                None => None,
            }
        }
    });

    // Append the OpenAI [DONE] marker after the channel closes
    let stream = stream.chain(stream::once(async {
        axum::response::sse::Event::default().data("[DONE]")
    }));

    into_sse_response(stream)
}
```

> **Note**: `tokenizer.encode_chat` is a placeholder name — verify by reading `src/tokenizer.rs` and use the actual chat-template encoding function. Adjust if the existing API is different.

- [ ] **Step 4: Wire and mount**

```rust
// src/server/http/handlers/mod.rs add:
pub mod chat;
```

Update `src/server/http/router.rs` to mount the chat route under auth:

```rust
use crate::server::http::handlers::chat::{chat_completions, ChatState};

pub fn build_router(
    auth: Arc<AuthState>,
    admin: Arc<AdminState>,
    chat_state: Arc<ChatState>,
) -> Router {
    let admin_routes = /* unchanged */;

    let chat_routes = Router::new()
        .route("/v1/chat/completions", post(chat_completions))
        .with_state(chat_state)
        .layer(axum_mw::from_fn_with_state(auth.clone(), crate::server::auth::middleware::auth_middleware));

    Router::new()
        .route("/health", get(handlers::health::health))
        .route("/v1/models", get(handlers::models::list_models))
        .merge(admin_routes)
        .merge(chat_routes)
        .layer(axum_mw::from_fn(request_id_middleware))
}
```

- [ ] **Step 5: Build**

```bash
cargo check --all-targets
```

Expected: clean.

- [ ] **Step 6: Commit**

```bash
git add src/server/http/handlers/chat.rs src/server/http/sse.rs \
        src/server/http/handlers/mod.rs src/server/http/mod.rs src/server/http/router.rs
git commit -m "feat(http): /v1/chat/completions handler with stream and non-stream"
```

---

## Task 6: Wire workers and queue into the binary

**Files:**
- Modify: `src/bin/qora_server.rs`
- Modify: `tests/integration/common.rs`

- [ ] **Step 1: Update binary main**

In `src/bin/qora_server.rs` add after auth construction:

```rust
let weights = Arc::new(qor4b::save::load_model(&config.model.path)?);
let tokenizer = Arc::new(qor4b::tokenizer::QoraTokenizer::from_file(&config.model.tokenizer)?);

let queue = qor4b::server::scheduler::queue::Queue::new(config.scheduler.max_queue);

// Spawn N worker threads (start with 1 — Phase 1 is single-slot serial-parallel)
let num_workers = num_cpus::get().min(8); // Conservative: kernel work is rayon-parallelized internally
for id in 0..num_workers {
    qor4b::server::scheduler::worker::Worker {
        id, weights: weights.clone(), tokenizer: tokenizer.clone(), queue: queue.clone(),
    }.spawn();
}

let usage = qor4b::server::usage::tracker::UsageTracker::new(pool.clone());

let chat_state = Arc::new(qor4b::server::http::handlers::chat::ChatState {
    queue, weights: weights.clone(), tokenizer: tokenizer.clone(), usage,
});

let router = build_router(auth, admin_state, chat_state);
```

Add `num_cpus = "1"` to Cargo.toml dependencies if not already present.

- [ ] **Step 2: Update test harness similarly**

In `tests/integration/common.rs`, add fixture-weights path: under feature `mock-weights`, build `weights` from `gemv::mock::tiny_weights()`. Otherwise skip (load_model would need a real path).

```rust
#[cfg(feature = "mock-weights")]
pub async fn start_with_mock_model() -> TestServer {
    use qor4b::gemv::mock::tiny_weights;
    let dir = TempDir::new().unwrap();
    let pool = open_pool(&dir.path().join("test.db")).await.unwrap();
    let admin_token = "test-admin".to_string();
    let auth = Arc::new(AuthState { pool: pool.clone(), limiter: RateLimiter::new() });
    let admin = Arc::new(AdminState { admin_token: admin_token.clone() });

    let weights = Arc::new(tiny_weights());
    // tokenizer for tests: a small fake or the real one if tokenizer.json fits in tests/fixtures
    let tokenizer = Arc::new(/* mock or real */);

    let queue = qor4b::server::scheduler::queue::Queue::new(100);
    qor4b::server::scheduler::worker::Worker {
        id: 0, weights: weights.clone(), tokenizer: tokenizer.clone(), queue: queue.clone(),
    }.spawn();

    let usage = qor4b::server::usage::tracker::UsageTracker::new(pool.clone());
    let chat_state = Arc::new(qor4b::server::http::handlers::chat::ChatState {
        queue, weights, tokenizer, usage,
    });

    let router = build_router(auth, admin, chat_state);
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let handle = tokio::spawn(async move { axum::serve(listener, router.into_make_service()).await.unwrap(); });
    tokio::time::sleep(std::time::Duration::from_millis(50)).await;
    TestServer { addr, pool, admin_token, handle, _dir: dir }
}
```

- [ ] **Step 3: Build full binary**

```bash
cargo build --bin qora-server --release
cargo build --bin qora-server --features mock-weights
```

Expected: clean build.

- [ ] **Step 4: Commit**

```bash
git add src/bin/qora_server.rs tests/integration/common.rs Cargo.toml
git commit -m "feat(server): wire workers, queue, and chat handler into binary"
```

---

## Task 7: Chat completions integration test (mock model)

**Files:**
- Create: `tests/integration/chat_completions.rs`
- Modify: `Cargo.toml`

- [ ] **Step 1: Add test target**

```toml
[[test]]
name = "chat_completions"
path = "tests/integration/chat_completions.rs"
required-features = ["mock-weights"]
```

- [ ] **Step 2: Write test**

Create `tests/integration/chat_completions.rs`:

```rust
mod common;
use common::start_with_mock_model;
use serde_json::json;

async fn create_user_key(server: &common::TestServer) -> String {
    let resp = reqwest::Client::new()
        .post(server.url("/v1/admin/keys"))
        .bearer_auth(&server.admin_token)
        .json(&json!({"user_id":"u1","name":"t","rpm_limit":1000,"tpm_limit":1_000_000}))
        .send().await.unwrap();
    let body: serde_json::Value = resp.json().await.unwrap();
    body["raw_key"].as_str().unwrap().to_string()
}

#[tokio::test]
async fn non_streaming_returns_completion() {
    let server = start_with_mock_model().await;
    let key = create_user_key(&server).await;

    let resp = reqwest::Client::new()
        .post(server.url("/v1/chat/completions"))
        .bearer_auth(&key)
        .json(&json!({
            "model": "qora-4b",
            "messages": [{"role":"user","content":"hi"}],
            "max_tokens": 4,
            "temperature": 0.0
        }))
        .send().await.unwrap();
    assert_eq!(resp.status(), 200);
    let body: serde_json::Value = resp.json().await.unwrap();
    assert_eq!(body["object"], "chat.completion");
    assert!(body["choices"][0]["message"]["content"].is_string());
    assert!(body["usage"]["completion_tokens"].as_u64().unwrap() > 0);
}

#[tokio::test]
async fn streaming_returns_sse_chunks() {
    let server = start_with_mock_model().await;
    let key = create_user_key(&server).await;

    let resp = reqwest::Client::new()
        .post(server.url("/v1/chat/completions"))
        .bearer_auth(&key)
        .json(&json!({
            "model": "qora-4b",
            "messages": [{"role":"user","content":"hi"}],
            "stream": true,
            "max_tokens": 4
        }))
        .send().await.unwrap();
    assert_eq!(resp.status(), 200);

    let body_text = resp.text().await.unwrap();
    assert!(body_text.contains("data: "));
    assert!(body_text.contains("[DONE]"));
}

#[tokio::test]
async fn unknown_model_returns_400() {
    let server = start_with_mock_model().await;
    let key = create_user_key(&server).await;
    let resp = reqwest::Client::new()
        .post(server.url("/v1/chat/completions"))
        .bearer_auth(&key)
        .json(&json!({"model":"gpt-99","messages":[]}))
        .send().await.unwrap();
    assert_eq!(resp.status(), 400);
}

#[tokio::test]
async fn missing_auth_returns_401() {
    let server = start_with_mock_model().await;
    let resp = reqwest::Client::new()
        .post(server.url("/v1/chat/completions"))
        .json(&json!({"model":"qora-4b","messages":[]}))
        .send().await.unwrap();
    assert_eq!(resp.status(), 401);
}
```

- [ ] **Step 3: Run tests**

```bash
cargo test --test chat_completions --features mock-weights
```

Expected: 4 tests pass (assuming `mock::tiny_weights()` is filled in correctly in Task 1).

- [ ] **Step 4: Commit**

```bash
git add tests/integration/chat_completions.rs Cargo.toml
git commit -m "test(http): chat completions integration suite (mock model)"
```

---

## Task 8: Module M3 acceptance + merge

- [ ] **Step 1: Run all tests**

```bash
cargo test --lib server::
cargo test --test http_basic
cargo test --test auth_flow
cargo test --test chat_completions --features mock-weights
cargo clippy --all-targets -- -D warnings
```

Expected: green.

- [ ] **Step 2: Manual smoke test against the real model**

```bash
cargo build --bin qora-server --release
QORA_ADMIN_TOKEN=admin1 ./target/release/qora-server /tmp/qora-test.toml &
PID=$!
sleep 3
ADMIN_RESP=$(curl -sf -H "Authorization: Bearer admin1" \
  -X POST http://127.0.0.1:18080/v1/admin/keys \
  -d '{"user_id":"u","name":"t","rpm_limit":60,"tpm_limit":60000}')
KEY=$(echo "$ADMIN_RESP" | jq -r .raw_key)
curl -sf -H "Authorization: Bearer $KEY" \
  -X POST http://127.0.0.1:18080/v1/chat/completions \
  -d '{"model":"qora-4b","messages":[{"role":"user","content":"hi"}],"max_tokens":20}'
kill $PID
```

Expected: completion JSON returned. Document the exact model load time + first-token latency in commit message.

- [ ] **Step 3: Milestone commit and merge**

```bash
git commit --allow-empty -m "milestone: M3 single-slot inference complete

- Session struct + per-session sampler
- Bounded queue + worker pool (Arc<ModelWeights>, parallel-independent)
- /v1/chat/completions handler (stream + non-stream)
- Usage tracking writes on completion
- Real model end-to-end smoke test passing

Acceptance: cargo test --test chat_completions --features mock-weights — 4/4 pass"

git checkout feature/qora-server
git merge --no-ff feature/qora-server-m3-session
git checkout feature/qora-server-m3-session
```

M3 complete. Proceed to M4 (batched kernels — the heavy lift).
