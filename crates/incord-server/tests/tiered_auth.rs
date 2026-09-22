//! Integration tests for M20: tiered authentication.
//!
//! Covers:
//! 1. Guest quota is enforced (N ok, N+1 → 402)
//! 2. Register + login return valid session tokens
//! 3. Session token works for inference
//! 4. API key path still works (unchanged)
//! 5. Duplicate email registration returns 400

#![cfg(feature = "mock-weights")]

use axum::body::Body;
use axum::http::{Request, StatusCode};
use serde_json::json;
use std::sync::Arc;
use tempfile::TempDir;
use tower::ServiceExt;

use incord::gemv::mock::tiny_weights;
use incord::tokenizer::IncordTokenizer;
use incord_server::auth::admin::AdminState;
use incord_server::auth::middleware::AuthState;
use incord_server::auth::ratelimit::RateLimiter;
use incord_server::config::TierConfig;
use incord_server::db;
use incord_server::engines::llm::LlmEngine;
use incord_server::http::handlers::inference::InferenceState;
use incord_server::http::router::build_router;
use incord_server::registry::Registry;

const TEST_ADMIN_TOKEN: &str = "test-admin-m20";
const TEST_SECRET: &[u8] = b"test-secret-m20";

fn build_test_tokenizer() -> Arc<IncordTokenizer> {
    let candidates = [
        concat!(env!("CARGO_MANIFEST_DIR"), "/../../model/tokenizer.json"),
        concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../incord-llm/model/tokenizer.json"
        ),
        concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../incord-llm/tests/fixtures/tokenizer.json"
        ),
    ];
    for path in candidates {
        let p = std::path::Path::new(path);
        if p.exists() {
            if let Ok(t) = IncordTokenizer::from_file(p) {
                return Arc::new(t);
            }
        }
    }
    panic!("no tokenizer.json found; checked: {candidates:?}");
}

/// Build a test app with a custom max_messages_per_guest.
fn build_test_app_with_guest_limit(max_msg: u32) -> (axum::Router, TempDir) {
    let dir = TempDir::new().unwrap();
    let db_path = dir.path().join("test.redb");
    let db = db::open(&db_path).unwrap();

    let mut tiers = TierConfig::default();
    tiers.anonymous.max_messages_per_guest = max_msg;

    let auth = Arc::new(AuthState {
        db: db.clone(),
        server_secret: Arc::new(TEST_SECRET.to_vec()),
        limiter: RateLimiter::new(),
        tiers,
        geo: incord_server::geoip::GeoDb::empty(),
    });
    let admin_state = Arc::new(AdminState {
        admin_token: TEST_ADMIN_TOKEN.to_string(),
    });

    let weights = Arc::new(tiny_weights());
    let tokenizer = build_test_tokenizer();
    let engine = Arc::new(LlmEngine::new(
        "test-llm",
        weights,
        tokenizer,
        "Test LLM",
    ));
    let registry = Registry::builder().add_chat(engine).build();
    let inference_state = Arc::new(InferenceState::for_test(registry, db.clone()));
    let app = build_router(inference_state, auth, admin_state);
    (app, dir)
}

/// Helper: hit /v1/chat/completions with an X-Guest-Id header.
async fn chat_as_guest(app: axum::Router, guest_id: &str) -> (StatusCode, axum::Router) {
    let body = json!({
        "model": "test-llm",
        "messages": [{"role":"user","content":"hello"}],
        "max_tokens": 2,
        "temperature": 0.0
    });
    let resp = app
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/chat/completions")
                .header("content-type", "application/json")
                .header("x-guest-id", guest_id)
                .body(Body::from(body.to_string()))
                .unwrap(),
        )
        .await
        .unwrap();
    (resp.status(), app)
}

// ─────────────────────────────────────────────────────────────────────────────
// Test 1: Guest quota enforced
// ─────────────────────────────────────────────────────────────────────────────

#[tokio::test]
async fn guest_quota_enforced() {
    let (app, _dir) = build_test_app_with_guest_limit(3);
    let guest_id = "test-guest-quota-1";

    // First 3 should succeed
    let (s1, app) = chat_as_guest(app, guest_id).await;
    assert_eq!(s1, StatusCode::OK, "1st message should succeed");

    let (s2, app) = chat_as_guest(app, guest_id).await;
    assert_eq!(s2, StatusCode::OK, "2nd message should succeed");

    let (s3, app) = chat_as_guest(app, guest_id).await;
    assert_eq!(s3, StatusCode::OK, "3rd message should succeed");

    // 4th should fail with 402
    let (s4, _app) = chat_as_guest(app, guest_id).await;
    assert_eq!(s4, StatusCode::PAYMENT_REQUIRED, "4th message should return 402");
}

// ─────────────────────────────────────────────────────────────────────────────
// Test 2: Register then login returns session tokens
// ─────────────────────────────────────────────────────────────────────────────

#[tokio::test]
async fn register_then_login_returns_session() {
    let (app, _dir) = build_test_app_with_guest_limit(10);

    let reg_body = json!({"email": "m20test@example.com", "password": "password123"});

    // Register
    let resp = app
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/auth/register")
                .header("content-type", "application/json")
                .body(Body::from(reg_body.to_string()))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::OK, "register should succeed");
    let bytes = axum::body::to_bytes(resp.into_body(), usize::MAX).await.unwrap();
    let reg: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
    let token1 = reg["session_token"].as_str().expect("session_token in register response");
    assert!(token1.starts_with("sess_"), "session token should start with sess_");
    assert_eq!(reg["tier"], "free");

    // Login
    let resp = app
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/auth/login")
                .header("content-type", "application/json")
                .body(Body::from(reg_body.to_string()))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::OK, "login should succeed");
    let bytes = axum::body::to_bytes(resp.into_body(), usize::MAX).await.unwrap();
    let login: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
    let token2 = login["session_token"].as_str().expect("session_token in login response");
    assert!(token2.starts_with("sess_"), "login token should start with sess_");

    // Both tokens are valid (different tokens per session)
    assert_ne!(token1, token2, "each login should produce a fresh token");
}

// ─────────────────────────────────────────────────────────────────────────────
// Test 3: Session token works for inference
// ─────────────────────────────────────────────────────────────────────────────

#[tokio::test]
async fn session_token_works_for_inference() {
    let (app, _dir) = build_test_app_with_guest_limit(10);

    // Register a user
    let reg_body = json!({"email": "sess-inference@example.com", "password": "password123"});
    let resp = app
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/auth/register")
                .header("content-type", "application/json")
                .body(Body::from(reg_body.to_string()))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    let bytes = axum::body::to_bytes(resp.into_body(), usize::MAX).await.unwrap();
    let reg: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
    let session_token = reg["session_token"].as_str().unwrap().to_string();

    // Use session token for inference
    let chat_body = json!({
        "model": "test-llm",
        "messages": [{"role":"user","content":"hello"}],
        "max_tokens": 2,
        "temperature": 0.0
    });
    let resp = app
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/chat/completions")
                .header("content-type", "application/json")
                .header("authorization", format!("Bearer {session_token}"))
                .body(Body::from(chat_body.to_string()))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::OK, "session token should allow inference");
}

// ─────────────────────────────────────────────────────────────────────────────
// Test 4: API key path still works
// ─────────────────────────────────────────────────────────────────────────────

#[tokio::test]
async fn api_key_path_still_works() {
    let (app, _dir) = build_test_app_with_guest_limit(10);

    // Create API key via admin
    let key_body = json!({
        "user_id": "api-key-user",
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
                .body(Body::from(key_body.to_string()))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    let bytes = axum::body::to_bytes(resp.into_body(), usize::MAX).await.unwrap();
    let key_resp: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
    let raw_key = key_resp["raw_key"].as_str().unwrap().to_string();
    assert!(raw_key.starts_with("incord_"));

    // Use API key for inference
    let chat_body = json!({
        "model": "test-llm",
        "messages": [{"role":"user","content":"hello"}],
        "max_tokens": 2,
        "temperature": 0.0
    });
    let resp = app
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/chat/completions")
                .header("content-type", "application/json")
                .header("authorization", format!("Bearer {raw_key}"))
                .body(Body::from(chat_body.to_string()))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::OK, "API key should still work");
}

// ─────────────────────────────────────────────────────────────────────────────
// Test 5: Duplicate email registration fails with 400
// ─────────────────────────────────────────────────────────────────────────────

#[tokio::test]
async fn duplicate_email_register_fails() {
    let (app, _dir) = build_test_app_with_guest_limit(10);
    let body = json!({"email": "dup@example.com", "password": "password123"});

    // First registration should succeed
    let resp = app
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/auth/register")
                .header("content-type", "application/json")
                .body(Body::from(body.to_string()))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::OK, "first register should succeed");

    // Second registration with same email should fail
    let resp = app
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/auth/register")
                .header("content-type", "application/json")
                .body(Body::from(body.to_string()))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::BAD_REQUEST, "duplicate email should return 400");
}

// ─────────────────────────────────────────────────────────────────────────────
// Test 6: No auth → 401
// ─────────────────────────────────────────────────────────────────────────────

#[tokio::test]
async fn no_auth_returns_401() {
    let (app, _dir) = build_test_app_with_guest_limit(10);

    let chat_body = json!({
        "model": "test-llm",
        "messages": [{"role":"user","content":"hello"}],
        "max_tokens": 2
    });
    let resp = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/chat/completions")
                .header("content-type", "application/json")
                .body(Body::from(chat_body.to_string()))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::UNAUTHORIZED);
}

// ─────────────────────────────────────────────────────────────────────────────
// Test 7: /v1/auth/me returns correct identity for each auth type
// ─────────────────────────────────────────────────────────────────────────────

#[tokio::test]
async fn me_endpoint_returns_correct_identity() {
    let (app, _dir) = build_test_app_with_guest_limit(10);

    // Register and get session token
    let reg_body = json!({"email": "me-test@example.com", "password": "password123"});
    let resp = app
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/auth/register")
                .header("content-type", "application/json")
                .body(Body::from(reg_body.to_string()))
                .unwrap(),
        )
        .await
        .unwrap();
    let bytes = axum::body::to_bytes(resp.into_body(), usize::MAX).await.unwrap();
    let reg: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
    let session_token = reg["session_token"].as_str().unwrap().to_string();

    // GET /v1/auth/me with session token
    let resp = app
        .clone()
        .oneshot(
            Request::builder()
                .method("GET")
                .uri("/v1/auth/me")
                .header("authorization", format!("Bearer {session_token}"))
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    let bytes = axum::body::to_bytes(resp.into_body(), usize::MAX).await.unwrap();
    let me: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(me["type"], "user");
    assert_eq!(me["email"], "me-test@example.com");

    // GET /v1/auth/me with guest header
    let resp = app
        .clone()
        .oneshot(
            Request::builder()
                .method("GET")
                .uri("/v1/auth/me")
                .header("x-guest-id", "guest-me-test")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    let bytes = axum::body::to_bytes(resp.into_body(), usize::MAX).await.unwrap();
    let me: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(me["type"], "guest");
    assert_eq!(me["guest_id"], "guest-me-test");
}
