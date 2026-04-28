//! Smoke test: registry-driven dispatch via /v1/inference and /v1/chat/completions.
//! Uses a runtime-supplied model id (NOT hardcoded "qora-4b" or "qora-9b") to
//! verify the dispatch logic works for any id.
//!
//! Auth flow: each test that hits /v1/* first POSTs to /v1/admin/keys (with the
//! admin bearer token) to mint a user key, then uses that key for inference.

#![cfg(feature = "mock-weights")]

use std::sync::Arc;
use axum::body::Body;
use axum::http::{Request, StatusCode};
use serde_json::json;
use tower::ServiceExt;
use tempfile::TempDir;

use qor4b::gemv::mock::tiny_weights;
use qor4b::tokenizer::QoraTokenizer;
use qora_server::auth::admin::AdminState;
use qora_server::auth::middleware::AuthState;
use qora_server::auth::ratelimit::RateLimiter;
use qora_server::db;
use qora_server::engines::llm::LlmEngine;
use qora_server::http::handlers::inference::InferenceState;
use qora_server::http::router::build_router;
use qora_server::registry::Registry;

const TEST_ADMIN_TOKEN: &str = "test-admin-token";
const TEST_SECRET: &[u8] = b"test-server-secret";

fn build_test_tokenizer() -> Arc<QoraTokenizer> {
    // Try several candidate paths so the tests work from different working dirs.
    // CARGO_MANIFEST_DIR is crates/qora-server at compile time.
    let candidates = [
        concat!(env!("CARGO_MANIFEST_DIR"), "/../../model/tokenizer.json"),
        concat!(env!("CARGO_MANIFEST_DIR"), "/../qora-llm/model/tokenizer.json"),
        concat!(env!("CARGO_MANIFEST_DIR"), "/../qora-llm/tests/fixtures/tokenizer.json"),
    ];
    for path in candidates {
        let p = std::path::Path::new(path);
        if p.exists() {
            if let Ok(t) = QoraTokenizer::from_file(p) {
                return Arc::new(t);
            }
        }
    }
    panic!("no tokenizer.json found; checked: {candidates:?}");
}

/// Build the full test app with auth wired (returns TempDir to keep DB alive).
fn build_test_app(
    model_id: &str,
) -> (axum::Router, TempDir) {
    let dir = TempDir::new().unwrap();
    let db_path = dir.path().join("test.redb");
    let db = db::open(&db_path).unwrap();

    let auth = Arc::new(AuthState {
        db,
        server_secret: Arc::new(TEST_SECRET.to_vec()),
        limiter: RateLimiter::new(),
    });
    let admin_state = Arc::new(AdminState {
        admin_token: TEST_ADMIN_TOKEN.to_string(),
    });

    let weights = Arc::new(tiny_weights());
    let tokenizer = build_test_tokenizer();
    let engine = Arc::new(LlmEngine::new(
        model_id,
        weights,
        tokenizer,
        format!("Test engine {}", model_id),
    ));
    let registry = Registry::builder().add_chat(engine).build();
    let inference_state = Arc::new(InferenceState { registry });

    let app = build_router(inference_state, auth, admin_state);
    (app, dir)
}

/// Mint a user API key via the admin endpoint. Returns the raw key.
async fn create_test_key(app: axum::Router) -> (String, axum::Router) {
    let body = json!({
        "user_id": "test-user",
        "name": "test-key",
        "rpm_limit": 10000,
        "tpm_limit": 10000000
    });

    let resp = app
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/admin/keys")
                .header("content-type", "application/json")
                .header("authorization", format!("Bearer {TEST_ADMIN_TOKEN}"))
                .body(Body::from(body.to_string()))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(resp.status(), StatusCode::OK, "admin key creation failed");
    let body_bytes = axum::body::to_bytes(resp.into_body(), usize::MAX)
        .await
        .unwrap();
    let body: serde_json::Value = serde_json::from_slice(&body_bytes).unwrap();
    let raw_key = body["raw_key"].as_str().unwrap().to_string();
    (raw_key, app)
}

#[tokio::test]
async fn unified_inference_dispatches_to_registered_chat_model() {
    let (app, _dir) = build_test_app("any-llm-id");
    let (raw_key, app) = create_test_key(app).await;

    let body = json!({
        "task": "chat",
        "model": "any-llm-id",
        "messages": [{"role":"user","content":"hi"}],
        "max_tokens": 2,
        "temperature": 0.0
    });

    let resp = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/inference")
                .header("content-type", "application/json")
                .header("authorization", format!("Bearer {raw_key}"))
                .body(Body::from(body.to_string()))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(resp.status(), StatusCode::OK);
    let body_bytes = axum::body::to_bytes(resp.into_body(), usize::MAX).await.unwrap();
    let body: serde_json::Value = serde_json::from_slice(&body_bytes).unwrap();
    assert_eq!(body["object"], "chat.completion");
    assert!(body["choices"][0]["message"]["content"].is_string());
}

#[tokio::test]
async fn openai_alias_dispatches_through_same_engine() {
    let (app, _dir) = build_test_app("any-other-id");
    let (raw_key, app) = create_test_key(app).await;

    let body = json!({
        "model": "any-other-id",
        "messages": [{"role":"user","content":"hi"}],
        "max_tokens": 2,
        "temperature": 0.0
    });

    let resp = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/chat/completions")
                .header("content-type", "application/json")
                .header("authorization", format!("Bearer {raw_key}"))
                .body(Body::from(body.to_string()))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(resp.status(), StatusCode::OK);
    let body_bytes = axum::body::to_bytes(resp.into_body(), usize::MAX).await.unwrap();
    let body: serde_json::Value = serde_json::from_slice(&body_bytes).unwrap();
    assert_eq!(body["object"], "chat.completion");
}

#[tokio::test]
async fn unknown_model_returns_400() {
    let (app, _dir) = build_test_app("does-exist");
    let (raw_key, app) = create_test_key(app).await;

    let body = json!({
        "task": "chat",
        "model": "does-not-exist",
        "messages": [{"role":"user","content":"hi"}]
    });

    let resp = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/inference")
                .header("content-type", "application/json")
                .header("authorization", format!("Bearer {raw_key}"))
                .body(Body::from(body.to_string()))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
}

#[tokio::test]
async fn list_models_returns_registry_contents() {
    // /v1/models is an open route — no auth needed.
    let (app, _dir) = build_test_app("model-a");

    let resp = app
        .oneshot(
            Request::builder()
                .uri("/v1/models")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(resp.status(), StatusCode::OK);
    let body_bytes = axum::body::to_bytes(resp.into_body(), usize::MAX).await.unwrap();
    let body: serde_json::Value = serde_json::from_slice(&body_bytes).unwrap();
    assert_eq!(body["object"], "list");
    assert_eq!(body["data"][0]["id"], "model-a");
    assert_eq!(body["data"][0]["task"], "chat");
}

#[tokio::test]
async fn inference_without_key_returns_401() {
    let (app, _dir) = build_test_app("some-model");

    let body = json!({
        "task": "chat",
        "model": "some-model",
        "messages": [{"role":"user","content":"hi"}]
    });

    let resp = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/inference")
                .header("content-type", "application/json")
                // deliberately no Authorization header
                .body(Body::from(body.to_string()))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(resp.status(), StatusCode::UNAUTHORIZED);
}

#[tokio::test]
async fn admin_endpoint_rejects_wrong_token() {
    let (app, _dir) = build_test_app("some-model");

    let body = json!({
        "user_id": "u1",
        "name": "k1",
        "rpm_limit": 60,
        "tpm_limit": 60000
    });

    let resp = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/admin/keys")
                .header("content-type", "application/json")
                .header("authorization", "Bearer wrong-token")
                .body(Body::from(body.to_string()))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(resp.status(), StatusCode::UNAUTHORIZED);
}

#[tokio::test]
async fn health_endpoint_open_no_auth() {
    let (app, _dir) = build_test_app("any-model");

    let resp = app
        .oneshot(
            Request::builder()
                .uri("/health")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(resp.status(), StatusCode::OK);
}

#[tokio::test]
async fn metrics_endpoint_open_no_auth() {
    let (app, _dir) = build_test_app("any-model");

    let resp = app
        .oneshot(
            Request::builder()
                .uri("/metrics")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(resp.status(), StatusCode::OK);
}
