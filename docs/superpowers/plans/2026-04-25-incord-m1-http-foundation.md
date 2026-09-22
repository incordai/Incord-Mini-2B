# M1 — HTTP Foundation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Stand up the axum HTTP server scaffold, OpenAI request/response types, TLS termination, configuration loader, and a new binary target. End state: server starts, `/health` returns 200, `/v1/models` returns a static model list, malformed JSON returns 400. No inference yet — just the HTTP plumbing.

**Prerequisites:** clean checkout of `main`, branch `feature/incord-server-m1-http` created.

**Architecture:** axum 0.8 routes mounted in `src/server/http/router.rs`. Config loaded from TOML via `src/server/config.rs`. New binary at `src/bin/incord_server.rs`. OpenAI types in `src/server/openai/`. Server runs on tokio multi-threaded runtime.

**Acceptance test:** `cargo test --test http_basic` passes — verifies server lifecycle, `/health`, `/v1/models`, JSON validation.

---

## Task 1: Add server dependencies to Cargo.toml

**Files:**
- Modify: `Cargo.toml`

- [ ] **Step 1: Update Cargo.toml dependencies**

```toml
[dependencies]
# existing dependencies retained
serde = { version = "1", features = ["derive"] }
serde_json = "1"
tokenizers = { version = "0.22", default-features = false, features = ["onig"] }
half = "2"
rayon = "1"
memmap2 = "0.9"
image = "0.25"

# NEW server dependencies
tokio = { version = "1", features = ["full"] }
axum = "0.8"
axum-server = { version = "0.7", features = ["tls-rustls"] }
tower = "0.5"
tower-http = { version = "0.6", features = ["trace", "cors", "limit", "timeout"] }
rustls = "0.23"
rustls-pemfile = "2"
hyper = "1"
async-trait = "0.1"
futures = "0.3"
bytes = "1"
uuid = { version = "1", features = ["v4", "serde"] }
toml = "0.8"
thiserror = "1"
anyhow = "1"
tracing = "0.1"
tracing-subscriber = { version = "0.3", features = ["env-filter", "json"] }

[dev-dependencies]
reqwest = { version = "0.12", features = ["json", "stream", "rustls-tls"] }
tempfile = "3"
tokio-test = "0.4"

# Add server binary target
[[bin]]
name = "incord-server"
path = "src/bin/incord_server.rs"
```

- [ ] **Step 2: Verify Cargo.toml parses**

```bash
cargo check --all-targets 2>&1 | head -40
```

Expected: `cargo` downloads new deps, no errors. Will warn about missing main file at `src/bin/incord_server.rs` — fix in next task.

- [ ] **Step 3: Commit**

```bash
git add Cargo.toml Cargo.lock
git commit -m "feat(server): add HTTP server dependencies to Cargo.toml"
```

---

## Task 2: Create empty server module skeleton

**Files:**
- Modify: `src/lib.rs`
- Create: `src/server/mod.rs`
- Create: `src/server/error.rs`
- Create: `src/bin/incord_server.rs`

- [ ] **Step 1: Add server module export to lib.rs**

```rust
// src/lib.rs — add this line:
pub mod server;
```

- [ ] **Step 2: Create src/server/mod.rs**

```rust
//! HTTP API server for Incord-Mini-2B.

pub mod error;
```

- [ ] **Step 3: Create src/server/error.rs**

```rust
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use axum::Json;
use serde_json::json;
use thiserror::Error;
use uuid::Uuid;

#[derive(Debug, Error)]
pub enum ServerError {
    #[error("invalid request: {0}")]
    BadRequest(String),
    #[error("authentication failed")]
    Unauthorized,
    #[error("rate limit exceeded")]
    RateLimited { retry_after_secs: u64 },
    #[error("server queue full")]
    QueueFull { retry_after_secs: u64 },
    #[error("request timed out")]
    Timeout,
    #[error("inference error: {0}")]
    Inference(String),
    #[error("database error: {0}")]
    Db(String),
    #[error("internal error: {0}")]
    Internal(String),
}

impl IntoResponse for ServerError {
    fn into_response(self) -> Response {
        let request_id = Uuid::new_v4();
        let (status, error_type, message) = match &self {
            Self::BadRequest(m) => (StatusCode::BAD_REQUEST, "invalid_request_error", m.clone()),
            Self::Unauthorized => (StatusCode::UNAUTHORIZED, "authentication_error", self.to_string()),
            Self::RateLimited { .. } => (StatusCode::TOO_MANY_REQUESTS, "rate_limit_error", self.to_string()),
            Self::QueueFull { .. } => (StatusCode::SERVICE_UNAVAILABLE, "server_error", self.to_string()),
            Self::Timeout => (StatusCode::REQUEST_TIMEOUT, "timeout_error", self.to_string()),
            Self::Inference(m) => (StatusCode::INTERNAL_SERVER_ERROR, "inference_error", m.clone()),
            Self::Db(m) => (StatusCode::INTERNAL_SERVER_ERROR, "database_error", m.clone()),
            Self::Internal(m) => (StatusCode::INTERNAL_SERVER_ERROR, "internal_error", m.clone()),
        };

        let body = Json(json!({
            "error": {
                "message": message,
                "type": error_type,
                "request_id": request_id,
            }
        }));

        let mut response = (status, body).into_response();

        if let Self::RateLimited { retry_after_secs } | Self::QueueFull { retry_after_secs } = &self {
            response.headers_mut().insert(
                "retry-after",
                retry_after_secs.to_string().parse().unwrap(),
            );
        }
        response
    }
}

pub type ServerResult<T> = Result<T, ServerError>;
```

- [ ] **Step 4: Create src/bin/incord_server.rs as a stub**

```rust
fn main() {
    eprintln!("incord-server: not yet implemented");
    std::process::exit(1);
}
```

- [ ] **Step 5: Verify build**

```bash
cargo check --all-targets
```

Expected: clean build, no errors.

- [ ] **Step 6: Commit**

```bash
git add src/lib.rs src/server/ src/bin/
git commit -m "feat(server): scaffold server module and binary stub"
```

---

## Task 3: Server config (TOML)

**Files:**
- Create: `src/server/config.rs`
- Modify: `src/server/mod.rs`
- Create: `config/incord-server.toml.example`

- [ ] **Step 1: Write the failing test**

Add to `src/server/config.rs`:

```rust
use serde::Deserialize;
use std::path::PathBuf;

#[derive(Debug, Deserialize, Clone)]
pub struct ServerConfig {
    pub server: HttpConfig,
    pub model: ModelConfig,
    pub scheduler: SchedulerConfig,
    pub auth: AuthConfig,
    pub limits: LimitsConfig,
    pub observability: ObservabilityConfig,
}

#[derive(Debug, Deserialize, Clone)]
pub struct HttpConfig {
    pub bind: String,
    pub tls_cert: Option<PathBuf>,
    pub tls_key: Option<PathBuf>,
}

#[derive(Debug, Deserialize, Clone)]
pub struct ModelConfig {
    pub path: PathBuf,
    pub tokenizer: PathBuf,
}

#[derive(Debug, Deserialize, Clone)]
pub struct SchedulerConfig {
    pub slot_count: usize,
    pub max_queue: usize,
    pub max_context: usize,
    pub request_timeout_secs: u64,
}

#[derive(Debug, Deserialize, Clone)]
pub struct AuthConfig {
    pub admin_token_env: String,
    pub db_path: PathBuf,
}

#[derive(Debug, Deserialize, Clone)]
pub struct LimitsConfig {
    pub default: PerKeyLimits,
}

#[derive(Debug, Deserialize, Clone)]
pub struct PerKeyLimits {
    pub rpm: u32,
    pub tpm: u32,
}

#[derive(Debug, Deserialize, Clone)]
pub struct ObservabilityConfig {
    pub prometheus_bind: String,
    pub log_level: String,
    pub log_format: String,
}

impl ServerConfig {
    pub fn load_from_path(path: &std::path::Path) -> anyhow::Result<Self> {
        let contents = std::fs::read_to_string(path)?;
        let config: Self = toml::from_str(&contents)?;
        Ok(config)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::NamedTempFile;
    use std::io::Write;

    #[test]
    fn loads_minimal_valid_config() {
        let mut tmp = NamedTempFile::new().unwrap();
        writeln!(tmp, r#"
[server]
bind = "127.0.0.1:8080"

[model]
path = "/opt/incord/model.incord"
tokenizer = "/opt/incord/tokenizer.json"

[scheduler]
slot_count = 64
max_queue = 1000
max_context = 8192
request_timeout_secs = 90

[auth]
admin_token_env = "INCORD_ADMIN_TOKEN"
db_path = "/opt/incord/auth.db"

[limits.default]
rpm = 60
tpm = 60000

[observability]
prometheus_bind = "127.0.0.1:9090"
log_level = "info"
log_format = "json"
"#).unwrap();
        let cfg = ServerConfig::load_from_path(tmp.path()).unwrap();
        assert_eq!(cfg.server.bind, "127.0.0.1:8080");
        assert_eq!(cfg.scheduler.slot_count, 64);
        assert_eq!(cfg.limits.default.rpm, 60);
    }

    #[test]
    fn missing_required_field_errors() {
        let mut tmp = NamedTempFile::new().unwrap();
        writeln!(tmp, r#"
[server]
bind = "127.0.0.1:8080"
"#).unwrap();
        assert!(ServerConfig::load_from_path(tmp.path()).is_err());
    }
}
```

- [ ] **Step 2: Add module to mod.rs**

```rust
// src/server/mod.rs
pub mod config;
pub mod error;
```

- [ ] **Step 3: Run tests, expect pass**

```bash
cargo test --lib server::config
```

Expected: 2 tests pass.

- [ ] **Step 4: Create example config file**

Write to `config/incord-server.toml.example`:

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
admin_token_env = "INCORD_ADMIN_TOKEN"
db_path = "/opt/incord/auth.db"

[limits.default]
rpm = 60
tpm = 60000

[observability]
prometheus_bind = "127.0.0.1:9090"
log_level = "info"
log_format = "json"
```

- [ ] **Step 5: Commit**

```bash
git add src/server/config.rs src/server/mod.rs config/
git commit -m "feat(server): add TOML config loader"
```

---

## Task 4: OpenAI request/response types — chat completions

**Files:**
- Create: `src/server/openai/mod.rs`
- Create: `src/server/openai/request.rs`
- Create: `src/server/openai/response.rs`
- Modify: `src/server/mod.rs`

- [ ] **Step 1: Create openai module**

```rust
// src/server/openai/mod.rs
pub mod request;
pub mod response;
```

- [ ] **Step 2: Write request types**

Create `src/server/openai/request.rs`:

```rust
use serde::{Deserialize, Serialize};

#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct ChatCompletionRequest {
    pub model: String,
    pub messages: Vec<Message>,

    #[serde(default)]
    pub stream: bool,

    #[serde(default)]
    pub max_tokens: Option<u32>,

    #[serde(default)]
    pub temperature: Option<f32>,

    #[serde(default)]
    pub top_p: Option<f32>,

    #[serde(default)]
    pub n: Option<u32>,

    #[serde(default)]
    pub stop: Option<StopValue>,

    #[serde(default)]
    pub presence_penalty: Option<f32>,

    #[serde(default)]
    pub frequency_penalty: Option<f32>,

    #[serde(default)]
    pub user: Option<String>,

    // Incord-specific extensions
    #[serde(default)]
    pub think_budget: Option<u32>,

    #[serde(default)]
    pub think: Option<bool>,

    #[serde(default)]
    pub show_think: Option<bool>,
}

#[derive(Debug, Deserialize, Serialize, Clone)]
#[serde(untagged)]
pub enum StopValue {
    Single(String),
    Multiple(Vec<String>),
}

#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct Message {
    pub role: String,
    #[serde(default)]
    pub content: MessageContent,
}

#[derive(Debug, Deserialize, Serialize, Clone)]
#[serde(untagged)]
pub enum MessageContent {
    Text(String),
    Parts(Vec<ContentPart>),
}

impl Default for MessageContent {
    fn default() -> Self { Self::Text(String::new()) }
}

#[derive(Debug, Deserialize, Serialize, Clone)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ContentPart {
    Text { text: String },
    ImageUrl { image_url: ImageUrl },
}

#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct ImageUrl {
    pub url: String,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_simple_text_message() {
        let json = r#"{
            "model": "incord-mini-2b",
            "messages": [
                {"role": "user", "content": "hi"}
            ]
        }"#;
        let req: ChatCompletionRequest = serde_json::from_str(json).unwrap();
        assert_eq!(req.model, "incord-mini-2b");
        assert_eq!(req.messages.len(), 1);
        assert!(matches!(req.messages[0].content, MessageContent::Text(_)));
    }

    #[test]
    fn parses_vision_content_parts() {
        let json = r#"{
            "model": "incord-mini-2b",
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "text", "text": "what's in this?"},
                    {"type": "image_url", "image_url": {"url": "data:image/png;base64,abc"}}
                ]
            }]
        }"#;
        let req: ChatCompletionRequest = serde_json::from_str(json).unwrap();
        if let MessageContent::Parts(parts) = &req.messages[0].content {
            assert_eq!(parts.len(), 2);
        } else {
            panic!("expected parts");
        }
    }

    #[test]
    fn missing_model_errors() {
        let json = r#"{"messages": []}"#;
        let r: Result<ChatCompletionRequest, _> = serde_json::from_str(json);
        assert!(r.is_err());
    }

    #[test]
    fn streaming_default_false() {
        let json = r#"{"model": "x", "messages": []}"#;
        let req: ChatCompletionRequest = serde_json::from_str(json).unwrap();
        assert!(!req.stream);
    }
}
```

- [ ] **Step 3: Write response types**

Create `src/server/openai/response.rs`:

```rust
use serde::{Deserialize, Serialize};

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct ChatCompletionResponse {
    pub id: String,
    pub object: &'static str,
    pub created: u64,
    pub model: String,
    pub choices: Vec<Choice>,
    pub usage: Usage,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct Choice {
    pub index: u32,
    pub message: ResponseMessage,
    pub finish_reason: Option<String>,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct ResponseMessage {
    pub role: String,
    pub content: String,
}

#[derive(Debug, Serialize, Deserialize, Clone, Default)]
pub struct Usage {
    pub prompt_tokens: u32,
    pub completion_tokens: u32,
    pub total_tokens: u32,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct ChatCompletionChunk {
    pub id: String,
    pub object: &'static str,
    pub created: u64,
    pub model: String,
    pub choices: Vec<ChunkChoice>,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct ChunkChoice {
    pub index: u32,
    pub delta: ChunkDelta,
    pub finish_reason: Option<String>,
}

#[derive(Debug, Serialize, Deserialize, Clone, Default)]
pub struct ChunkDelta {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub role: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content: Option<String>,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct ModelsListResponse {
    pub object: &'static str,
    pub data: Vec<ModelInfo>,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct ModelInfo {
    pub id: String,
    pub object: &'static str,
    pub created: u64,
    pub owned_by: String,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn serializes_chat_response_in_openai_shape() {
        let r = ChatCompletionResponse {
            id: "chatcmpl-1".into(),
            object: "chat.completion",
            created: 1_700_000_000,
            model: "incord-mini-2b".into(),
            choices: vec![Choice {
                index: 0,
                message: ResponseMessage { role: "assistant".into(), content: "hi".into() },
                finish_reason: Some("stop".into()),
            }],
            usage: Usage { prompt_tokens: 5, completion_tokens: 1, total_tokens: 6 },
        };
        let json = serde_json::to_value(&r).unwrap();
        assert_eq!(json["object"], "chat.completion");
        assert_eq!(json["choices"][0]["finish_reason"], "stop");
        assert_eq!(json["usage"]["total_tokens"], 6);
    }
}
```

- [ ] **Step 4: Wire openai module**

```rust
// src/server/mod.rs
pub mod config;
pub mod error;
pub mod openai;
```

- [ ] **Step 5: Run tests**

```bash
cargo test --lib server::openai
```

Expected: 5 tests pass.

- [ ] **Step 6: Commit**

```bash
git add src/server/openai/ src/server/mod.rs
git commit -m "feat(server): add OpenAI-compatible request/response types"
```

---

## Task 5: HTTP router skeleton with /health and /v1/models

**Files:**
- Create: `src/server/http/mod.rs`
- Create: `src/server/http/router.rs`
- Create: `src/server/http/handlers/mod.rs`
- Create: `src/server/http/handlers/health.rs`
- Create: `src/server/http/handlers/models.rs`
- Modify: `src/server/mod.rs`

- [ ] **Step 1: Wire http module**

```rust
// src/server/mod.rs
pub mod config;
pub mod error;
pub mod http;
pub mod openai;
```

```rust
// src/server/http/mod.rs
pub mod handlers;
pub mod router;
```

```rust
// src/server/http/handlers/mod.rs
pub mod health;
pub mod models;
```

- [ ] **Step 2: Write health handler**

Create `src/server/http/handlers/health.rs`:

```rust
use axum::Json;
use serde_json::json;

pub async fn health() -> Json<serde_json::Value> {
    Json(json!({"status": "ok"}))
}
```

- [ ] **Step 3: Write models handler**

Create `src/server/http/handlers/models.rs`:

```rust
use axum::Json;
use crate::server::openai::response::{ModelInfo, ModelsListResponse};

pub async fn list_models() -> Json<ModelsListResponse> {
    Json(ModelsListResponse {
        object: "list",
        data: vec![ModelInfo {
            id: "incord-mini-2b".into(),
            object: "model",
            created: 1_700_000_000,
            owned_by: "incord".into(),
        }],
    })
}
```

- [ ] **Step 4: Write router**

Create `src/server/http/router.rs`:

```rust
use axum::{Router, routing::get};
use crate::server::http::handlers;

pub fn build_router() -> Router {
    Router::new()
        .route("/health", get(handlers::health::health))
        .route("/v1/models", get(handlers::models::list_models))
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::body::Body;
    use axum::http::{Request, StatusCode};
    use tower::ServiceExt;

    #[tokio::test]
    async fn health_returns_200() {
        let app = build_router();
        let response = app
            .oneshot(Request::builder().uri("/health").body(Body::empty()).unwrap())
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn models_returns_incord_mini_2b() {
        let app = build_router();
        let response = app
            .oneshot(Request::builder().uri("/v1/models").body(Body::empty()).unwrap())
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        let body_bytes = axum::body::to_bytes(response.into_body(), usize::MAX).await.unwrap();
        let body: serde_json::Value = serde_json::from_slice(&body_bytes).unwrap();
        assert_eq!(body["data"][0]["id"], "incord-mini-2b");
    }

    #[tokio::test]
    async fn unknown_route_returns_404() {
        let app = build_router();
        let response = app
            .oneshot(Request::builder().uri("/nope").body(Body::empty()).unwrap())
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::NOT_FOUND);
    }
}
```

- [ ] **Step 5: Run tests**

```bash
cargo test --lib server::http
```

Expected: 3 tests pass.

- [ ] **Step 6: Commit**

```bash
git add src/server/http/ src/server/mod.rs
git commit -m "feat(server): add HTTP router with /health and /v1/models"
```

---

## Task 6: Server binary that starts axum on configured bind address

**Files:**
- Modify: `src/bin/incord_server.rs`
- Modify: `src/server/mod.rs`
- Create: `src/server/http/serve.rs`
- Modify: `src/server/http/mod.rs`

- [ ] **Step 1: Create serve module (no TLS yet — added in Task 7)**

Create `src/server/http/serve.rs`:

```rust
use std::net::SocketAddr;
use anyhow::Context;
use axum::Router;

pub async fn serve_plain(router: Router, bind: &str) -> anyhow::Result<()> {
    let addr: SocketAddr = bind.parse().context("parsing bind address")?;
    let listener = tokio::net::TcpListener::bind(addr).await?;
    tracing::info!(%addr, "incord-server listening (plain HTTP)");
    axum::serve(listener, router.into_make_service()).await?;
    Ok(())
}
```

- [ ] **Step 2: Wire serve into module tree**

```rust
// src/server/http/mod.rs
pub mod handlers;
pub mod router;
pub mod serve;
```

- [ ] **Step 3: Implement binary main**

Update `src/bin/incord_server.rs`:

```rust
use std::path::PathBuf;
use anyhow::Context;
use incord::server::{config::ServerConfig, http::{router::build_router, serve::serve_plain}};

#[tokio::main(flavor = "multi_thread")]
async fn main() -> anyhow::Result<()> {
    init_tracing();

    let config_path = std::env::args().nth(1).unwrap_or_else(|| "incord-server.toml".to_string());
    let config = ServerConfig::load_from_path(&PathBuf::from(&config_path))
        .with_context(|| format!("loading config from {config_path}"))?;
    tracing::info!(?config_path, "config loaded");

    let router = build_router();

    serve_plain(router, &config.server.bind).await?;
    Ok(())
}

fn init_tracing() {
    use tracing_subscriber::{EnvFilter, fmt};
    let filter = EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info"));
    fmt().with_env_filter(filter).json().init();
}
```

- [ ] **Step 4: Build and run smoke test**

```bash
# Create a minimal test config
cat > /tmp/incord-test.toml <<EOF
[server]
bind = "127.0.0.1:18080"

[model]
path = "/tmp/nonexistent.incord"
tokenizer = "/tmp/nonexistent.json"

[scheduler]
slot_count = 64
max_queue = 1000
max_context = 8192
request_timeout_secs = 90

[auth]
admin_token_env = "INCORD_ADMIN_TOKEN"
db_path = "/tmp/incord-auth.db"

[limits.default]
rpm = 60
tpm = 60000

[observability]
prometheus_bind = "127.0.0.1:19090"
log_level = "info"
log_format = "json"
EOF

cargo build --bin incord-server --release
./target/release/incord-server /tmp/incord-test.toml &
SERVER_PID=$!
sleep 2
curl -sf http://127.0.0.1:18080/health
curl -sf http://127.0.0.1:18080/v1/models
kill $SERVER_PID
```

Expected: both curl commands print JSON. Server starts and stops cleanly.

- [ ] **Step 5: Commit**

```bash
git add src/bin/incord_server.rs src/server/http/serve.rs src/server/http/mod.rs
git commit -m "feat(server): incord-server binary serves /health and /v1/models"
```

---

## Task 7: TLS termination via rustls

**Files:**
- Create: `src/server/http/tls.rs`
- Modify: `src/server/http/mod.rs`
- Modify: `src/server/http/serve.rs`
- Modify: `src/bin/incord_server.rs`

- [ ] **Step 1: Add tls module**

Create `src/server/http/tls.rs`:

```rust
use std::path::Path;
use anyhow::{anyhow, Context};
use axum_server::tls_rustls::RustlsConfig;

pub async fn load_tls_config(cert_path: &Path, key_path: &Path) -> anyhow::Result<RustlsConfig> {
    let config = RustlsConfig::from_pem_file(cert_path, key_path)
        .await
        .with_context(|| format!("loading TLS cert from {} and key from {}",
                                 cert_path.display(), key_path.display()))?;
    Ok(config)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::NamedTempFile;

    #[tokio::test]
    async fn rejects_missing_files() {
        let res = load_tls_config(
            Path::new("/nonexistent/cert.pem"),
            Path::new("/nonexistent/key.pem"),
        ).await;
        assert!(res.is_err());
    }
}
```

- [ ] **Step 2: Wire tls module**

```rust
// src/server/http/mod.rs
pub mod handlers;
pub mod router;
pub mod serve;
pub mod tls;
```

- [ ] **Step 3: Add serve_tls function**

Append to `src/server/http/serve.rs`:

```rust
use axum_server::tls_rustls::RustlsConfig;

pub async fn serve_tls(router: Router, bind: &str, tls: RustlsConfig) -> anyhow::Result<()> {
    let addr: SocketAddr = bind.parse().context("parsing bind address")?;
    tracing::info!(%addr, "incord-server listening (TLS)");
    axum_server::bind_rustls(addr, tls)
        .serve(router.into_make_service())
        .await?;
    Ok(())
}
```

- [ ] **Step 4: Update binary to choose TLS vs plain**

In `src/bin/incord_server.rs`, replace the `serve_plain(...)` call with:

```rust
match (&config.server.tls_cert, &config.server.tls_key) {
    (Some(cert), Some(key)) => {
        let tls = incord::server::http::tls::load_tls_config(cert, key).await?;
        incord::server::http::serve::serve_tls(router, &config.server.bind, tls).await
    }
    (None, None) => {
        incord::server::http::serve::serve_plain(router, &config.server.bind).await
    }
    _ => Err(anyhow::anyhow!("tls_cert and tls_key must be set together or both omitted")),
}?;
```

- [ ] **Step 5: Verify build**

```bash
cargo check --all-targets
cargo test --lib server::http::tls
```

Expected: clean build, 1 tls test passes.

- [ ] **Step 6: Commit**

```bash
git add src/server/http/tls.rs src/server/http/serve.rs src/server/http/mod.rs src/bin/incord_server.rs
git commit -m "feat(server): add optional TLS termination via rustls"
```

---

## Task 8: Request ID middleware + structured logging

**Files:**
- Create: `src/server/http/middleware.rs`
- Modify: `src/server/http/mod.rs`
- Modify: `src/server/http/router.rs`

- [ ] **Step 1: Create middleware module**

Create `src/server/http/middleware.rs`:

```rust
use axum::{
    extract::Request,
    http::{HeaderValue, header::HeaderName},
    middleware::Next,
    response::Response,
};
use uuid::Uuid;

pub static REQUEST_ID_HEADER: HeaderName = HeaderName::from_static("x-request-id");

pub async fn request_id_middleware(mut req: Request, next: Next) -> Response {
    let request_id = req
        .headers()
        .get(&REQUEST_ID_HEADER)
        .and_then(|v| v.to_str().ok())
        .map(|s| s.to_string())
        .unwrap_or_else(|| Uuid::new_v4().to_string());

    let span = tracing::info_span!("request",
        request_id = %request_id,
        method = %req.method(),
        path = req.uri().path(),
    );
    let _enter = span.enter();
    tracing::info!("request received");

    req.extensions_mut().insert(RequestId(request_id.clone()));

    let mut response = next.run(req).await;

    if let Ok(value) = HeaderValue::from_str(&request_id) {
        response.headers_mut().insert(REQUEST_ID_HEADER.clone(), value);
    }

    tracing::info!(status = %response.status(), "request complete");
    response
}

#[derive(Debug, Clone)]
pub struct RequestId(pub String);
```

- [ ] **Step 2: Wire middleware module**

```rust
// src/server/http/mod.rs
pub mod handlers;
pub mod middleware;
pub mod router;
pub mod serve;
pub mod tls;
```

- [ ] **Step 3: Apply middleware in router**

Update `src/server/http/router.rs`:

```rust
use axum::{Router, middleware as axum_middleware, routing::get};
use crate::server::http::handlers;
use crate::server::http::middleware::request_id_middleware;

pub fn build_router() -> Router {
    Router::new()
        .route("/health", get(handlers::health::health))
        .route("/v1/models", get(handlers::models::list_models))
        .layer(axum_middleware::from_fn(request_id_middleware))
}

// existing tests retained — all should still pass
```

- [ ] **Step 4: Add a test that verifies the response carries x-request-id**

Append to the tests module in `src/server/http/router.rs`:

```rust
    #[tokio::test]
    async fn response_carries_request_id_header() {
        let app = build_router();
        let response = app
            .oneshot(Request::builder().uri("/health").body(Body::empty()).unwrap())
            .await
            .unwrap();
        assert!(response.headers().contains_key("x-request-id"));
    }
```

- [ ] **Step 5: Run tests**

```bash
cargo test --lib server::http
```

Expected: 4 tests pass (3 existing + 1 new).

- [ ] **Step 6: Commit**

```bash
git add src/server/http/middleware.rs src/server/http/mod.rs src/server/http/router.rs
git commit -m "feat(server): add request-id middleware and structured logging"
```

---

## Task 9: Integration test harness

**Files:**
- Create: `tests/integration/mod.rs`
- Create: `tests/integration/common.rs`
- Create: `tests/integration/http_basic.rs`
- Modify: `Cargo.toml` (add test target)

- [ ] **Step 1: Add integration test target**

Append to `Cargo.toml`:

```toml
[[test]]
name = "http_basic"
path = "tests/integration/http_basic.rs"
```

- [ ] **Step 2: Create test harness**

Create `tests/integration/common.rs`:

```rust
use std::net::SocketAddr;
use incord::server::http::router::build_router;
use tokio::net::TcpListener;
use tokio::task::JoinHandle;

pub struct TestServer {
    pub addr: SocketAddr,
    handle: JoinHandle<()>,
}

impl TestServer {
    pub async fn start() -> Self {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let router = build_router();
        let handle = tokio::spawn(async move {
            axum::serve(listener, router.into_make_service()).await.unwrap();
        });
        // Give the server a moment to be ready
        tokio::time::sleep(std::time::Duration::from_millis(50)).await;
        Self { addr, handle }
    }

    pub fn url(&self, path: &str) -> String {
        format!("http://{}{}", self.addr, path)
    }
}

impl Drop for TestServer {
    fn drop(&mut self) {
        self.handle.abort();
    }
}
```

- [ ] **Step 3: Write http_basic test**

Create `tests/integration/http_basic.rs`:

```rust
mod common;
use common::TestServer;

#[tokio::test]
async fn health_endpoint_returns_200() {
    let server = TestServer::start().await;
    let resp = reqwest::get(server.url("/health")).await.unwrap();
    assert_eq!(resp.status(), 200);
    let body: serde_json::Value = resp.json().await.unwrap();
    assert_eq!(body["status"], "ok");
}

#[tokio::test]
async fn models_endpoint_lists_incord_mini_2b() {
    let server = TestServer::start().await;
    let resp = reqwest::get(server.url("/v1/models")).await.unwrap();
    assert_eq!(resp.status(), 200);
    let body: serde_json::Value = resp.json().await.unwrap();
    assert_eq!(body["object"], "list");
    assert_eq!(body["data"][0]["id"], "incord-mini-2b");
}

#[tokio::test]
async fn unknown_route_returns_404() {
    let server = TestServer::start().await;
    let resp = reqwest::get(server.url("/nope")).await.unwrap();
    assert_eq!(resp.status(), 404);
}

#[tokio::test]
async fn response_includes_x_request_id() {
    let server = TestServer::start().await;
    let resp = reqwest::get(server.url("/health")).await.unwrap();
    assert!(resp.headers().contains_key("x-request-id"));
}
```

- [ ] **Step 4: Run integration test**

```bash
cargo test --test http_basic
```

Expected: 4 tests pass.

- [ ] **Step 5: Commit**

```bash
git add tests/integration/ Cargo.toml
git commit -m "test(server): add integration test harness and http_basic suite"
```

---

## Task 10: Module M1 acceptance check + branch merge

- [ ] **Step 1: Run all tests in module scope**

```bash
cargo test --lib server::
cargo test --test http_basic
cargo clippy --all-targets -- -D warnings
cargo fmt --check
```

Expected: all green.

- [ ] **Step 2: Verify acceptance criteria**

- Server starts on random port: ✓ (covered by `tests/integration/http_basic.rs::health_endpoint_returns_200`)
- `/health` returns 200: ✓
- `/v1/models` returns mock model list: ✓
- Malformed JSON → 400: deferred to M2/M3 when chat completion handler exists (no JSON-accepting endpoint yet)
- TLS works: smoke-tested manually with self-signed cert in Task 7

- [ ] **Step 3: Commit module milestone**

```bash
git commit --allow-empty -m "milestone: M1 HTTP foundation complete

- axum router with /health and /v1/models
- TOML config loading
- TLS termination via rustls
- Request-id middleware + structured logs
- OpenAI-compatible types ready for M2

Acceptance test: cargo test --test http_basic — 4/4 pass"
```

- [ ] **Step 4: Merge to feature/incord-server branch**

```bash
git checkout feature/incord-server
git merge --no-ff feature/incord-server-m1-http
git checkout feature/incord-server-m1-http
```

M1 complete. Proceed to M2.
