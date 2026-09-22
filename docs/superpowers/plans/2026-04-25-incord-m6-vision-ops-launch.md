# M6 — Vision, Observability, Deployment, Launch

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development.

**Goal:** Add vision endpoint (image + video), Prometheus metrics, deployment artifacts (systemd unit, config example, install script), production load testing, and the launch checklist. End state: a complete v1 ready to deploy on the EPYC server, document, and start selling.

**Prerequisites:** M5 merged. Branch `feature/incord-server-m6-launch` from `feature/incord-server`.

**Architecture:**
- Vision: parse OpenAI image_url (data URL or http(s) URL), decode to RGB bytes, run existing `vision::encode_image` on a dedicated semaphore-limited thread, inject embeddings into prefill via existing `prefill_with_vision`.
- Metrics: `prometheus` crate, registered counters/histograms, scraped at `/metrics`.
- Deployment: systemd unit, install script, example configs, backup procedure.
- Load tests: Rust binary that ramps up concurrency and reports p50/p95/p99 latency and throughput.

**Acceptance tests:**
- (a) `cargo test --test vision --features mock-weights` — image upload + completion roundtrip
- (b) `cargo run --release --bin load_test -- --concurrency 150 --duration 600` against the EPYC server — p95 < 30s for 500-token completion at 150 concurrent users

---

## Task 1: Vision URL parsing

**Files:**
- Create: `src/server/openai/vision.rs`
- Modify: `src/server/openai/mod.rs`
- Modify: `Cargo.toml`

- [ ] **Step 1: Add deps**

```toml
base64 = "0.22"
url = "2.5"
```

- [ ] **Step 2: Implement parser**

```rust
// src/server/openai/vision.rs
use base64::{Engine, engine::general_purpose::STANDARD as B64};

pub enum VisionContent {
    Inline { mime: String, bytes: Vec<u8> },
    Remote { url: String },
}

pub fn parse_image_url(url: &str) -> anyhow::Result<VisionContent> {
    if let Some(rest) = url.strip_prefix("data:") {
        // data:[<mime>];base64,<data>
        let (header, data) = rest.split_once(',').ok_or_else(|| anyhow::anyhow!("invalid data url"))?;
        let mime = header.split(';').next().unwrap_or("image/png").to_string();
        let bytes = if header.contains("base64") {
            B64.decode(data)?
        } else {
            urlencoding::decode(data)?.as_bytes().to_vec()
        };
        return Ok(VisionContent::Inline { mime, bytes });
    }
    if url.starts_with("http://") || url.starts_with("https://") {
        return Ok(VisionContent::Remote { url: url.to_string() });
    }
    Err(anyhow::anyhow!("unsupported image_url scheme: {}", &url[..url.len().min(40)]))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_data_url_base64() {
        let url = "data:image/png;base64,aGVsbG8=";
        match parse_image_url(url).unwrap() {
            VisionContent::Inline { mime, bytes } => {
                assert_eq!(mime, "image/png");
                assert_eq!(bytes, b"hello");
            }
            _ => panic!("expected inline"),
        }
    }

    #[test]
    fn parses_https() {
        match parse_image_url("https://example.com/x.png").unwrap() {
            VisionContent::Remote { url } => assert!(url.starts_with("https://")),
            _ => panic!("expected remote"),
        }
    }

    #[test]
    fn rejects_unknown_scheme() {
        assert!(parse_image_url("ftp://x.png").is_err());
    }
}
```

> Add `urlencoding = "2"` to deps if needed for non-base64 data URLs (rare).

- [ ] **Step 3: Wire**

```rust
// src/server/openai/mod.rs
pub mod request;
pub mod response;
pub mod vision;
```

- [ ] **Step 4: Test**

```bash
cargo test --lib server::openai::vision
```

- [ ] **Step 5: Commit**

```bash
git add src/server/openai/vision.rs src/server/openai/mod.rs Cargo.toml
git commit -m "feat(vision): parse OpenAI image_url (data and http schemes)"
```

---

## Task 2: Vision request handling in chat handler

**Files:**
- Modify: `src/server/http/handlers/chat.rs`
- Modify: `src/bin/incord_server.rs`

The chat handler currently ignores `ContentPart::ImageUrl`. Wire vision encoding into the request flow.

Architecture: a vision encoder semaphore (max 1 concurrent encode by default — encode is heavy, ~90s for an image) running on `spawn_blocking`. Encoded embeddings are passed to `prefill_with_vision` instead of `prefill`.

- [ ] **Step 1: Vision admission state**

In `ChatState`, add:

```rust
pub vision_semaphore: Arc<tokio::sync::Semaphore>,
```

Initialize in binary with permits = 1 (configurable).

- [ ] **Step 2: Update render + admission**

In `render_messages`, when an `ImageUrl` is encountered, fetch+decode bytes (if remote, via reqwest), then pass through. Modify return type to `(prompt_text, Vec<image_bytes>)`.

```rust
async fn render_messages_with_images(
    messages: &[Message],
) -> ServerResult<(String, Vec<Vec<u8>>)> {
    let mut text = String::new();
    let mut images = Vec::new();
    for m in messages {
        match &m.content {
            MessageContent::Text(t) => text.push_str(&format!("<|{}|>{}\n", m.role, t)),
            MessageContent::Parts(parts) => {
                text.push_str(&format!("<|{}|>", m.role));
                for p in parts {
                    match p {
                        ContentPart::Text { text: t } => text.push_str(t),
                        ContentPart::ImageUrl { image_url } => {
                            let parsed = parse_image_url(&image_url.url)
                                .map_err(|e| ServerError::BadRequest(e.to_string()))?;
                            let bytes = match parsed {
                                VisionContent::Inline { bytes, .. } => bytes,
                                VisionContent::Remote { url } => {
                                    let r = reqwest::get(&url).await
                                        .map_err(|e| ServerError::BadRequest(format!("fetching image: {e}")))?;
                                    if !r.status().is_success() {
                                        return Err(ServerError::BadRequest(format!("image fetch {} {}", url, r.status())));
                                    }
                                    r.bytes().await.map_err(|e| ServerError::BadRequest(e.to_string()))?.to_vec()
                                }
                            };
                            text.push_str("<|vision_start|><|image_pad|><|vision_end|>");
                            images.push(bytes);
                        }
                    }
                }
                text.push('\n');
            }
        }
    }
    Ok((text, images))
}
```

- [ ] **Step 3: Pre-encode images before admission**

In `chat_completions`, after `render_messages_with_images`, if `images.is_empty() == false`:

```rust
let permit = state.vision_semaphore.clone().acquire_owned().await
    .map_err(|_| ServerError::Internal("vision semaphore closed".into()))?;
let weights_clone = state.weights.clone();
let images_clone = images.clone();
let vision_embeds = tokio::task::spawn_blocking(move || {
    let _permit = permit;
    let vision = weights_clone.vision.as_ref()
        .ok_or_else(|| anyhow::anyhow!("model has no vision weights"))?;
    // For v1, support a single image per request. Multi-image: future M6.x
    let img_bytes = &images_clone[0];
    let img = image::load_from_memory(img_bytes).map_err(anyhow::Error::from)?;
    let rgb = img.to_rgb8();
    let (h, w) = (rgb.height() as usize, rgb.width() as usize);
    // Convert to the f32 pixel layout the existing encoder expects (see vision.rs::load_image).
    // The function exposed by vision.rs may need a minor adapter taking bytes directly.
    let pixels = incord::vision::pixels_from_rgb_bytes(rgb.as_raw(), w, h);
    let (embeds, num_tokens) = vision.encode_image(&pixels, h, w);
    Ok::<_, anyhow::Error>((embeds, num_tokens))
}).await.map_err(|e| ServerError::Internal(format!("vision task: {e}")))?
   .map_err(|e| ServerError::Inference(format!("vision encode: {e}")))?;
```

> **Note**: `vision.rs` may not expose `pixels_from_rgb_bytes` today. If not, add it as a thin wrapper around the existing pixel normalization code. Keep the change minimal.

- [ ] **Step 4: Pass embeds through queue**

Extend `PendingRequest` to carry `Option<(Vec<f32>, usize)>` for vision embeds, and have the batcher's `prefill_pending` use `prefill_with_vision` when present.

- [ ] **Step 5: Update binary state init**

```rust
let vision_semaphore = Arc::new(tokio::sync::Semaphore::new(1));
let chat_state = Arc::new(ChatState { /* ...existing... */, vision_semaphore });
```

- [ ] **Step 6: Build**

```bash
cargo check --all-targets
```

- [ ] **Step 7: Commit**

```bash
git add src/server/http/handlers/chat.rs src/server/scheduler/queue.rs \
        src/server/scheduler/batcher.rs src/bin/incord_server.rs
git commit -m "feat(vision): wire image input through chat → vision encoder → prefill_with_vision"
```

---

## Task 3: Vision integration test

**Files:**
- Create: `tests/integration/vision.rs`
- Create: `tests/fixtures/tiny.png`
- Modify: `Cargo.toml`

- [ ] **Step 1: Add fixture**

Generate a 32x32 PNG. Store in `tests/fixtures/tiny.png` (any small valid PNG works; e.g., a gradient generated by `image` crate at build-time, or commit a tiny one).

- [ ] **Step 2: Add test target**

```toml
[[test]]
name = "vision"
path = "tests/integration/vision.rs"
required-features = ["mock-weights"]
```

- [ ] **Step 3: Test**

```rust
mod common;
use common::start_with_mock_model;
use serde_json::json;
use base64::{Engine, engine::general_purpose::STANDARD as B64};

#[tokio::test]
async fn image_completion_returns_text() {
    let server = start_with_mock_model().await;
    let png = std::fs::read("tests/fixtures/tiny.png").unwrap();
    let data_url = format!("data:image/png;base64,{}", B64.encode(&png));

    // Create key
    let body: serde_json::Value = reqwest::Client::new()
        .post(server.url("/v1/admin/keys"))
        .bearer_auth(&server.admin_token)
        .json(&json!({"user_id":"u","name":"t","rpm_limit":1000,"tpm_limit":1_000_000}))
        .send().await.unwrap()
        .json().await.unwrap();
    let key = body["raw_key"].as_str().unwrap().to_string();

    let resp = reqwest::Client::new()
        .post(server.url("/v1/chat/completions"))
        .bearer_auth(&key)
        .json(&json!({
            "model":"incord-mini-2b",
            "messages":[{
                "role":"user",
                "content":[
                    {"type":"text","text":"describe"},
                    {"type":"image_url","image_url":{"url":data_url}}
                ]
            }],
            "max_tokens":4
        }))
        .send().await.unwrap();
    assert_eq!(resp.status(), 200);
    let body: serde_json::Value = resp.json().await.unwrap();
    assert!(body["choices"][0]["message"]["content"].is_string());
}
```

- [ ] **Step 4: Run**

```bash
cargo test --test vision --features mock-weights
```

> Note: `tiny_weights` mock must include a vision encoder stub OR the test conditionally skips when no vision weights exist. Cleanest: make the mock include a tiny vision encoder.

- [ ] **Step 5: Commit**

```bash
git add tests/integration/vision.rs tests/fixtures/tiny.png Cargo.toml
git commit -m "test(vision): image input → completion roundtrip"
```

---

## Task 4: Prometheus metrics

**Files:**
- Create: `src/server/metrics/mod.rs`
- Create: `src/server/http/handlers/metrics.rs`
- Modify: `src/server/http/router.rs`
- Modify: `src/server/http/handlers/mod.rs`
- Modify: `src/server/mod.rs`
- Modify: `Cargo.toml`

- [ ] **Step 1: Add deps**

```toml
prometheus = "0.13"
once_cell = "1"
```

- [ ] **Step 2: Define metrics**

```rust
// src/server/metrics/mod.rs
use once_cell::sync::Lazy;
use prometheus::{
    register_counter, register_counter_vec, register_gauge, register_histogram_vec,
    Counter, CounterVec, Gauge, HistogramVec, Encoder, TextEncoder,
};

pub static REQUESTS_TOTAL: Lazy<CounterVec> = Lazy::new(|| {
    register_counter_vec!("incord_requests_total", "Total requests", &["endpoint", "status"]).unwrap()
});

pub static TOKENS_EMITTED: Lazy<Counter> = Lazy::new(|| {
    register_counter!("incord_tokens_emitted_total", "Total output tokens emitted").unwrap()
});

pub static REQUEST_DURATION: Lazy<HistogramVec> = Lazy::new(|| {
    register_histogram_vec!(
        "incord_request_duration_seconds", "Request duration",
        &["endpoint"],
        vec![0.1, 0.5, 1.0, 5.0, 10.0, 30.0, 60.0, 120.0, 300.0]
    ).unwrap()
});

pub static ACTIVE_SLOTS: Lazy<Gauge> = Lazy::new(|| {
    register_gauge!("incord_active_slots", "Currently active slots").unwrap()
});

pub static QUEUE_DEPTH: Lazy<Gauge> = Lazy::new(|| {
    register_gauge!("incord_queue_depth", "Pending requests in queue").unwrap()
});

pub fn render_metrics() -> String {
    let mut buf = Vec::new();
    let encoder = TextEncoder::new();
    let metrics = prometheus::gather();
    encoder.encode(&metrics, &mut buf).unwrap();
    String::from_utf8(buf).unwrap()
}
```

- [ ] **Step 3: Handler**

```rust
// src/server/http/handlers/metrics.rs
use axum::http::header;
use axum::response::IntoResponse;

pub async fn metrics_handler() -> impl IntoResponse {
    (
        [(header::CONTENT_TYPE, "text/plain; version=0.0.4")],
        crate::server::metrics::render_metrics(),
    )
}
```

- [ ] **Step 4: Wire**

```rust
// src/server/mod.rs add: pub mod metrics;
// src/server/http/handlers/mod.rs add: pub mod metrics;
// src/server/http/router.rs add a /metrics route
.route("/metrics", get(handlers::metrics::metrics_handler))
```

- [ ] **Step 5: Instrument call sites**

In `chat.rs`: increment `REQUESTS_TOTAL` and observe `REQUEST_DURATION` per request.
In `batcher.rs`: bump `TOKENS_EMITTED` per token sent; update `ACTIVE_SLOTS` gauge each iteration.
In `queue.rs`: update `QUEUE_DEPTH` after push/pop.

- [ ] **Step 6: Test**

```rust
// tests/integration/metrics.rs
#[tokio::test]
async fn metrics_endpoint_returns_prometheus_text() {
    let server = common::start_with_mock_model().await;
    let resp = reqwest::get(server.url("/metrics")).await.unwrap();
    assert_eq!(resp.status(), 200);
    let body = resp.text().await.unwrap();
    assert!(body.contains("incord_requests_total"));
}
```

```toml
[[test]]
name = "metrics"
path = "tests/integration/metrics.rs"
required-features = ["mock-weights"]
```

- [ ] **Step 7: Commit**

```bash
git add src/server/metrics/ src/server/http/handlers/metrics.rs \
        src/server/http/handlers/mod.rs src/server/http/router.rs \
        src/server/mod.rs tests/integration/metrics.rs Cargo.toml
git commit -m "feat(metrics): Prometheus instrumentation + /metrics endpoint"
```

---

## Task 5: systemd unit + install script

**Files:**
- Create: `config/incord-server.service`
- Create: `scripts/install.sh`
- Create: `docs/deployment.md`

- [ ] **Step 1: systemd unit**

```ini
# config/incord-server.service
[Unit]
Description=Incord-Mini-2B API server
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=incord
Group=incord
EnvironmentFile=-/etc/incord/incord.env
ExecStart=/usr/local/bin/incord-server /etc/incord/incord-server.toml
Restart=on-failure
RestartSec=5
StandardOutput=journal
StandardError=journal
LimitNOFILE=65536
NoNewPrivileges=true
ProtectSystem=strict
ProtectHome=true
ReadWritePaths=/var/lib/incord

[Install]
WantedBy=multi-user.target
```

- [ ] **Step 2: Install script**

```bash
#!/usr/bin/env bash
# scripts/install.sh
set -euo pipefail

# Idempotent install for Incord API server on Ubuntu 24.04.
# Run as root.
[ "$(id -u)" = 0 ] || { echo "must run as root"; exit 1; }

useradd --system --no-create-home --shell /usr/sbin/nologin incord || true
mkdir -p /etc/incord /var/lib/incord /var/log/incord
chown -R incord:incord /var/lib/incord /var/log/incord

# Binary
install -o root -g root -m 0755 target/release/incord-server /usr/local/bin/incord-server

# Config
[ -f /etc/incord/incord-server.toml ] || install -o root -g incord -m 0640 \
  config/incord-server.toml.example /etc/incord/incord-server.toml

# Systemd
install -o root -g root -m 0644 config/incord-server.service /etc/systemd/system/
systemctl daemon-reload

echo "Installed. Edit /etc/incord/incord-server.toml, set INCORD_ADMIN_TOKEN in /etc/incord/incord.env, then:"
echo "  systemctl enable --now incord-server"
echo "  journalctl -u incord-server -f"
```

- [ ] **Step 3: Deployment doc**

Create `docs/deployment.md` covering:
- Hardware requirements (EPYC 9354 / 256 GB DDR5 / no GPU)
- Required files (`model.incord`, `tokenizer.json`)
- TLS cert procurement (Let's Encrypt)
- Backup procedure (sqlite WAL + nightly cp to second SSD)
- Admin token rotation
- Restart and recovery
- Health and metrics URLs

(File contents are documentation — write to communicate, not exhaustively. ~400 words.)

- [ ] **Step 4: Commit**

```bash
git add config/incord-server.service scripts/install.sh docs/deployment.md
git commit -m "feat(ops): systemd unit, install script, deployment docs"
```

---

## Task 6: Load test binary

**Files:**
- Create: `tests/load/load_test.rs` (binary, not test)
- Modify: `Cargo.toml`

- [ ] **Step 1: Add bin target**

```toml
[[bin]]
name = "load_test"
path = "tests/load/load_test.rs"
```

- [ ] **Step 2: Implement**

```rust
// tests/load/load_test.rs
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};
use serde_json::json;

#[tokio::main(flavor = "multi_thread")]
async fn main() {
    let mut concurrency = 100usize;
    let mut duration_secs = 60u64;
    let mut url = "http://127.0.0.1:8080".to_string();
    let mut key = "incord_test".to_string();
    let mut max_tokens = 200u32;

    let args: Vec<String> = std::env::args().collect();
    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--concurrency" => { concurrency = args[i+1].parse().unwrap(); i += 2; }
            "--duration"    => { duration_secs = args[i+1].parse().unwrap(); i += 2; }
            "--url"         => { url = args[i+1].clone(); i += 2; }
            "--key"         => { key = args[i+1].clone(); i += 2; }
            "--max-tokens"  => { max_tokens = args[i+1].parse().unwrap(); i += 2; }
            _ => i += 1,
        }
    }

    println!("Load test: {concurrency} concurrent, {duration_secs}s, max_tokens={max_tokens}");

    let stop = Instant::now() + Duration::from_secs(duration_secs);
    let total = Arc::new(AtomicU64::new(0));
    let errors = Arc::new(AtomicU64::new(0));
    let latencies = Arc::new(parking_lot::Mutex::new(Vec::<u64>::with_capacity(100_000)));

    let mut handles = Vec::with_capacity(concurrency);
    for _ in 0..concurrency {
        let url = url.clone();
        let key = key.clone();
        let total = total.clone();
        let errors = errors.clone();
        let latencies = latencies.clone();
        handles.push(tokio::spawn(async move {
            let client = reqwest::Client::new();
            while Instant::now() < stop {
                let started = Instant::now();
                let res = client.post(format!("{url}/v1/chat/completions"))
                    .bearer_auth(&key)
                    .json(&json!({
                        "model":"incord-mini-2b",
                        "messages":[{"role":"user","content":"explain entropy briefly"}],
                        "max_tokens":max_tokens,
                        "stream":false
                    }))
                    .send().await;
                let elapsed = started.elapsed().as_millis() as u64;
                match res {
                    Ok(r) if r.status().is_success() => {
                        let _ = r.bytes().await;
                        total.fetch_add(1, Ordering::Relaxed);
                        latencies.lock().push(elapsed);
                    }
                    _ => { errors.fetch_add(1, Ordering::Relaxed); }
                }
            }
        }));
    }
    for h in handles { let _ = h.await; }

    let mut lats = latencies.lock().clone();
    lats.sort_unstable();
    let p = |q: f64| lats.get(((lats.len() as f64) * q) as usize).copied().unwrap_or(0);
    println!("requests:   {}", total.load(Ordering::Relaxed));
    println!("errors:     {}", errors.load(Ordering::Relaxed));
    println!("p50 (ms):   {}", p(0.50));
    println!("p95 (ms):   {}", p(0.95));
    println!("p99 (ms):   {}", p(0.99));
}
```

Add `parking_lot = "0.12"` if not already.

- [ ] **Step 3: Smoke run**

```bash
cargo build --release --bin load_test
# Against running server:
./target/release/load_test --concurrency 50 --duration 30 --key incord_yourkey
```

- [ ] **Step 4: Commit**

```bash
git add tests/load/load_test.rs Cargo.toml
git commit -m "test(load): load test binary measuring p50/p95/p99 latency"
```

---

## Task 7: Soak test script

**Files:**
- Create: `scripts/soak_24h.sh`

- [ ] **Step 1: Script**

```bash
#!/usr/bin/env bash
# scripts/soak_24h.sh — 24-hour soak test
set -euo pipefail
URL=${URL:-http://127.0.0.1:8080}
KEY=${KEY:-incord_test}
CONCURRENCY=${CONCURRENCY:-100}
DURATION=$((24*3600))

LOGFILE=/var/log/incord/soak_$(date +%Y%m%d_%H%M).log
mkdir -p "$(dirname $LOGFILE)"

# Memory monitor
( while sleep 60; do
    rss=$(ps -o rss= -p $(pgrep -f incord-server))
    echo "$(date -Iseconds) RSS=${rss}KB" >> "$LOGFILE.mem"
done ) &
MON=$!

trap "kill $MON" EXIT

./target/release/load_test --concurrency "$CONCURRENCY" --duration "$DURATION" \
  --url "$URL" --key "$KEY" --max-tokens 200 \
  | tee -a "$LOGFILE"

# Verify RSS stable: extract first and last RSS values, compute drift
first=$(head -1 "$LOGFILE.mem" | awk '{print $2}' | tr -d 'RSS=KB')
last=$(tail -1 "$LOGFILE.mem"  | awk '{print $2}' | tr -d 'RSS=KB')
drift=$(awk -v a=$first -v b=$last 'BEGIN { printf "%.2f", (b-a)/a*100 }')
echo "RSS drift: ${drift}%"
awk -v d=$drift 'BEGIN { exit !(d < 5 && d > -5) }' \
  || { echo "RSS drift > 5% — likely memory leak"; exit 1; }
```

- [ ] **Step 2: Commit**

```bash
git add scripts/soak_24h.sh
chmod +x scripts/soak_24h.sh
git add scripts/soak_24h.sh
git commit -m "test(load): 24h soak test script with RSS drift check"
```

---

## Task 8: API documentation

**Files:**
- Create: `docs/api.md`
- Create: `docs/examples/curl/chat.sh`
- Create: `docs/examples/openai-python/chat.py`
- Create: `docs/examples/openai-node/chat.mjs`

Each example shows: how to set base_url, how to send a chat completion (stream + non-stream), how to send vision input, error handling. Copy-pasteable.

- [ ] **Step 1: Write `docs/api.md`** documenting endpoints, auth, rate limits, error codes, OpenAI compatibility notes.

- [ ] **Step 2: Curl example**

```bash
#!/usr/bin/env bash
# docs/examples/curl/chat.sh
INCORD_KEY=${INCORD_KEY:-incord_yourkey}
INCORD_URL=${INCORD_URL:-https://api.example.com}
curl -sS "$INCORD_URL/v1/chat/completions" \
  -H "Authorization: Bearer $INCORD_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model":"incord-mini-2b",
    "messages":[{"role":"user","content":"Hello"}],
    "max_tokens":200
  }'
```

- [ ] **Step 3: Python example**

```python
# docs/examples/openai-python/chat.py
import os
from openai import OpenAI
client = OpenAI(
    api_key=os.environ["INCORD_KEY"],
    base_url=os.environ.get("INCORD_URL", "https://api.example.com/v1"),
)
resp = client.chat.completions.create(
    model="incord-mini-2b",
    messages=[{"role": "user", "content": "Hello"}],
    max_tokens=200,
)
print(resp.choices[0].message.content)
```

- [ ] **Step 4: Node example**

```javascript
// docs/examples/openai-node/chat.mjs
import OpenAI from "openai";
const client = new OpenAI({
  apiKey: process.env.INCORD_KEY,
  baseURL: process.env.INCORD_URL ?? "https://api.example.com/v1",
});
const resp = await client.chat.completions.create({
  model: "incord-mini-2b",
  messages: [{ role: "user", content: "Hello" }],
  max_tokens: 200,
});
console.log(resp.choices[0].message.content);
```

- [ ] **Step 5: Commit**

```bash
git add docs/api.md docs/examples/
git commit -m "docs: API documentation + curl/python/node examples"
```

---

## Task 9: Production load test on the EPYC server

This is run manually, not in CI.

- [ ] **Step 1: Deploy a build to the EPYC server**

```bash
cargo build --release --bin incord-server
scp target/release/incord-server incord@epyc:/tmp/
scp model/model.incord incord@epyc:/var/lib/incord/    # 3 GB transfer
scp model/tokenizer.json incord@epyc:/var/lib/incord/
ssh root@epyc 'bash scripts/install.sh && systemctl enable --now incord-server'
```

- [ ] **Step 2: Run load tests at increasing concurrency**

```bash
# From a separate machine on the same network:
for c in 10 50 100 150 200 300; do
  echo "=== concurrency $c ==="
  ./target/release/load_test --concurrency $c --duration 300 \
    --url https://api.example.com --key incord_loadtest --max-tokens 500
done
```

Expected (per spec):
- 100 concurrent: p95 < 20s
- 150 concurrent: p95 < 30s
- 300 concurrent: p95 may be > 60s (over capacity, OK if no errors)

Document results. If 150 concurrent does NOT meet p95 < 30s, profile (perf, flamegraph) and tune scheduler before launch.

- [ ] **Step 3: 24h soak**

```bash
ssh epyc bash scripts/soak_24h.sh
```

Verify RSS drift < 5%.

- [ ] **Step 4: Commit results**

```bash
mkdir -p docs/load-test-results/
# Copy soak/load logs from EPYC to docs/load-test-results/2026-MM-DD/
git add docs/load-test-results/
git commit -m "test(load): production load + 24h soak results — meets v1 SLO"
```

---

## Task 10: Final launch checklist

Walk the master plan's launch checklist. Each item must be checked off before merging `feature/incord-server` → `main`.

- [ ] All M1–M6 acceptance tests pass on a clean checkout
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
- [ ] Pricing/billing integration plan documented (Stripe metering wiring is a 1-day post-launch task)

When all green:

```bash
git checkout main
git merge --no-ff feature/incord-server
git tag -a v1.0.0 -m "Incord Multi-User API Server v1.0.0 — first sellable release"
git push origin main --tags
```

Deploy:

```bash
ssh epyc 'systemctl restart incord-server && journalctl -u incord-server -f'
```

Open the gates. Start selling.

---

## Module M6 acceptance + final merge

```bash
git commit --allow-empty -m "milestone: M6 vision + ops + launch complete

- Vision endpoint accepting OpenAI image_url (data + http)
- Prometheus /metrics endpoint
- systemd unit + install script
- Load test binary (p50/p95/p99)
- 24h soak test script
- Full API documentation + examples (curl/python/node)
- Production load test passing on EPYC: 150 concurrent at p95 < 30s
- 24h soak: RSS drift < 5%

Acceptance: all launch checklist items green"

git checkout feature/incord-server
git merge --no-ff feature/incord-server-m6-launch
# After final v1 verification:
git checkout main
git merge --no-ff feature/incord-server
git tag v1.0.0
```

V1 ships. Next steps (post-launch): Stripe billing integration, multi-server load balancer (when traffic justifies), Incord model improvements (separate roadmap).
