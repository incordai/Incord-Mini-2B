//! Integration tests for chat history endpoints (M17).
//!
//! Tests: create / list / get / export / delete conversations,
//! plus privacy check (user A cannot see user B's conversations).

#![cfg(feature = "mock-weights")]

use axum::body::Body;
use axum::http::{Request, StatusCode};
use serde_json::json;
use std::sync::Arc;
use tempfile::TempDir;
use tower::ServiceExt;

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

const TEST_ADMIN_TOKEN: &str = "test-admin-conv";
const TEST_SECRET: &[u8] = b"test-conv-secret";

fn build_test_tokenizer() -> Arc<QoraTokenizer> {
    let candidates = [
        concat!(env!("CARGO_MANIFEST_DIR"), "/../../model/tokenizer.json"),
        concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../qora-llm/model/tokenizer.json"
        ),
        concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../qora-llm/tests/fixtures/tokenizer.json"
        ),
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

/// Build a test app + TempDir (kept alive to prevent DB cleanup).
fn build_test_app() -> (axum::Router, TempDir) {
    let dir = TempDir::new().unwrap();
    let db_path = dir.path().join("test.redb");
    let db = db::open(&db_path).unwrap();

    let auth = Arc::new(AuthState {
        db: db.clone(),
        server_secret: Arc::new(TEST_SECRET.to_vec()),
        limiter: RateLimiter::new(),
    });
    let admin_state = Arc::new(AdminState {
        admin_token: TEST_ADMIN_TOKEN.to_string(),
    });

    let weights = Arc::new(tiny_weights());
    let tokenizer = build_test_tokenizer();
    let engine = Arc::new(LlmEngine::new(
        "conv-test-llm",
        weights,
        tokenizer,
        "Conv test engine",
    ));
    let registry = Registry::builder().add_chat(engine).build();
    let inference_state = Arc::new(InferenceState::for_test(registry, db.clone()));

    let app = build_router(inference_state, auth, admin_state);
    (app, dir)
}

/// Mint a user key. Returns raw key string.
async fn mint_key_for(app: axum::Router, user_id: &str) -> (String, axum::Router) {
    let body = json!({
        "user_id": user_id,
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
    assert_eq!(resp.status(), StatusCode::OK, "key creation failed");
    let bytes = axum::body::to_bytes(resp.into_body(), usize::MAX)
        .await
        .unwrap();
    let v: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
    (v["raw_key"].as_str().unwrap().to_string(), app)
}

// ---------------------------------------------------------------------------

#[tokio::test]
async fn create_then_list_then_get_then_delete() {
    let (app, _dir) = build_test_app();
    let (key, app) = mint_key_for(app, "user-a").await;

    // --- Create ---
    let resp = app
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/conversations")
                .header("content-type", "application/json")
                .header("authorization", format!("Bearer {key}"))
                .body(Body::from(json!({"title": "my first chat"}).to_string()))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::OK, "create conversation failed");
    let bytes = axum::body::to_bytes(resp.into_body(), usize::MAX)
        .await
        .unwrap();
    let created: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
    let conv_id = created["id"].as_str().expect("id missing").to_string();
    assert!(!conv_id.is_empty());

    // --- List ---
    let resp = app
        .clone()
        .oneshot(
            Request::builder()
                .method("GET")
                .uri("/v1/conversations")
                .header("authorization", format!("Bearer {key}"))
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::OK, "list conversations failed");
    let bytes = axum::body::to_bytes(resp.into_body(), usize::MAX)
        .await
        .unwrap();
    let list: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(list["object"], "list");
    let data = list["data"].as_array().unwrap();
    assert_eq!(data.len(), 1);
    assert_eq!(data[0]["id"], conv_id);

    // --- Get ---
    let resp = app
        .clone()
        .oneshot(
            Request::builder()
                .method("GET")
                .uri(format!("/v1/conversations/{conv_id}"))
                .header("authorization", format!("Bearer {key}"))
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::OK, "get conversation failed");
    let bytes = axum::body::to_bytes(resp.into_body(), usize::MAX)
        .await
        .unwrap();
    let got: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(got["id"], conv_id);
    assert_eq!(got["title"], "my first chat");
    assert!(got["messages"].as_array().unwrap().is_empty());

    // --- Delete ---
    let resp = app
        .clone()
        .oneshot(
            Request::builder()
                .method("DELETE")
                .uri(format!("/v1/conversations/{conv_id}"))
                .header("authorization", format!("Bearer {key}"))
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::OK, "delete conversation failed");
    let bytes = axum::body::to_bytes(resp.into_body(), usize::MAX)
        .await
        .unwrap();
    let del: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(del["deleted"], true);

    // --- Get after delete → 400 ---
    let resp = app
        .clone()
        .oneshot(
            Request::builder()
                .method("GET")
                .uri(format!("/v1/conversations/{conv_id}"))
                .header("authorization", format!("Bearer {key}"))
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(
        resp.status(),
        StatusCode::BAD_REQUEST,
        "should be 400 after deletion"
    );
}

#[tokio::test]
async fn user_a_cannot_see_user_b_conversation() {
    let (app, _dir) = build_test_app();
    let (key_a, app) = mint_key_for(app, "privacy-user-a").await;
    let (key_b, app) = mint_key_for(app, "privacy-user-b").await;

    // User A creates a conversation
    let resp = app
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/conversations")
                .header("content-type", "application/json")
                .header("authorization", format!("Bearer {key_a}"))
                .body(Body::from(json!({"title": "secret"}).to_string()))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    let bytes = axum::body::to_bytes(resp.into_body(), usize::MAX)
        .await
        .unwrap();
    let created: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
    let conv_id = created["id"].as_str().unwrap().to_string();

    // User B tries to GET it → 400 (not found / privacy)
    let resp = app
        .clone()
        .oneshot(
            Request::builder()
                .method("GET")
                .uri(format!("/v1/conversations/{conv_id}"))
                .header("authorization", format!("Bearer {key_b}"))
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(
        resp.status(),
        StatusCode::BAD_REQUEST,
        "user B should not be able to see user A's conversation"
    );

    // User B tries to DELETE it → 400
    let resp = app
        .clone()
        .oneshot(
            Request::builder()
                .method("DELETE")
                .uri(format!("/v1/conversations/{conv_id}"))
                .header("authorization", format!("Bearer {key_b}"))
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(
        resp.status(),
        StatusCode::BAD_REQUEST,
        "user B should not be able to delete user A's conversation"
    );

    // User B's list is empty
    let resp = app
        .clone()
        .oneshot(
            Request::builder()
                .method("GET")
                .uri("/v1/conversations")
                .header("authorization", format!("Bearer {key_b}"))
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    let bytes = axum::body::to_bytes(resp.into_body(), usize::MAX)
        .await
        .unwrap();
    let list: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(
        list["data"].as_array().unwrap().len(),
        0,
        "user B should have no conversations"
    );
}

#[tokio::test]
async fn export_returns_json_with_correct_disposition() {
    let (app, _dir) = build_test_app();
    let (key, app) = mint_key_for(app, "export-user").await;

    // Create conversation
    let resp = app
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/conversations")
                .header("content-type", "application/json")
                .header("authorization", format!("Bearer {key}"))
                .body(Body::from(json!({"title": "export test"}).to_string()))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    let bytes = axum::body::to_bytes(resp.into_body(), usize::MAX)
        .await
        .unwrap();
    let created: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
    let conv_id = created["id"].as_str().unwrap().to_string();

    // Export
    let resp = app
        .clone()
        .oneshot(
            Request::builder()
                .method("GET")
                .uri(format!("/v1/conversations/{conv_id}/export"))
                .header("authorization", format!("Bearer {key}"))
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::OK, "export failed");

    // Check Content-Disposition header
    let disposition = resp
        .headers()
        .get("content-disposition")
        .and_then(|v| v.to_str().ok())
        .unwrap_or("");
    assert!(
        disposition.contains(&format!("qora-chat-{conv_id}.json")),
        "content-disposition should contain filename, got: {disposition}"
    );

    // Check body is valid JSON with required fields
    let bytes = axum::body::to_bytes(resp.into_body(), usize::MAX)
        .await
        .unwrap();
    let body: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(body["id"], conv_id);
    assert!(body["messages"].is_array(), "messages should be an array");
}

#[tokio::test]
async fn unauthenticated_request_returns_401() {
    let (app, _dir) = build_test_app();

    let resp = app
        .oneshot(
            Request::builder()
                .method("GET")
                .uri("/v1/conversations")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::UNAUTHORIZED);
}
