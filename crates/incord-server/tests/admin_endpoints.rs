//! Integration tests for admin backend endpoints (M18).
//!
//! Covers:
//! 1. GET /v1/admin/keys requires admin token (401 without)
//! 2. After creating 2 keys, list returns 2 entries
//! 3. GET /v1/admin/stats returns totals + models
//! 4. GET /v1/admin/users aggregates correctly
//! 5. GET /v1/admin/usage returns records with filters

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
use incord_server::db;
use incord_server::engines::llm::LlmEngine;
use incord_server::http::handlers::inference::InferenceState;
use incord_server::http::router::build_router;
use incord_server::registry::Registry;

const TEST_ADMIN_TOKEN: &str = "test-admin-m18";
const TEST_SECRET: &[u8] = b"test-admin-secret-m18";

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

fn build_test_app() -> (axum::Router, TempDir) {
    let dir = TempDir::new().unwrap();
    let db_path = dir.path().join("test.redb");
    let db = db::open(&db_path).unwrap();

    let auth = Arc::new(AuthState {
        db: db.clone(),
        server_secret: Arc::new(TEST_SECRET.to_vec()),
        limiter: RateLimiter::new(),
        tiers: Default::default(),
        geo: incord_server::geoip::GeoDb::empty(),
    });
    let admin_state = Arc::new(AdminState {
        admin_token: TEST_ADMIN_TOKEN.to_string(),
    });

    let weights = Arc::new(tiny_weights());
    let tokenizer = build_test_tokenizer();
    let engine = Arc::new(LlmEngine::new(
        "admin-test-llm",
        weights,
        tokenizer,
        "Admin test engine",
    ));
    let registry = Registry::builder().add_chat(engine).build();
    let inference_state = Arc::new(InferenceState::for_test(registry, db.clone()));

    let app = build_router(inference_state, auth, admin_state);
    (app, dir)
}

/// Create a key via the admin API. Returns raw key string.
async fn admin_create_key(app: axum::Router, user_id: &str, name: &str) -> (String, axum::Router) {
    let body = json!({
        "user_id": user_id,
        "name": name,
        "rpm_limit": 1000,
        "tpm_limit": 100000
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

// ─────────────────────────────────────────────────────────────────────────────
// Test 1: GET /v1/admin/keys requires admin token (401 without)
// ─────────────────────────────────────────────────────────────────────────────

#[tokio::test]
async fn list_keys_requires_admin_token() {
    let (app, _dir) = build_test_app();

    // No auth header → 401
    let resp = app
        .clone()
        .oneshot(
            Request::builder()
                .method("GET")
                .uri("/v1/admin/keys")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::UNAUTHORIZED, "expected 401 with no auth");

    // Wrong token → 401
    let resp = app
        .clone()
        .oneshot(
            Request::builder()
                .method("GET")
                .uri("/v1/admin/keys")
                .header("authorization", "Bearer wrong-token")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::UNAUTHORIZED, "expected 401 with wrong token");

    // Correct token → 200
    let resp = app
        .oneshot(
            Request::builder()
                .method("GET")
                .uri("/v1/admin/keys")
                .header("authorization", format!("Bearer {TEST_ADMIN_TOKEN}"))
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::OK, "expected 200 with correct admin token");
}

// ─────────────────────────────────────────────────────────────────────────────
// Test 2: After creating 2 keys, list returns 2 entries
// ─────────────────────────────────────────────────────────────────────────────

#[tokio::test]
async fn list_keys_returns_all_keys() {
    let (app, _dir) = build_test_app();

    let (_, app) = admin_create_key(app, "user-alpha", "key-1").await;
    let (_, app) = admin_create_key(app, "user-beta", "key-2").await;

    let resp = app
        .oneshot(
            Request::builder()
                .method("GET")
                .uri("/v1/admin/keys")
                .header("authorization", format!("Bearer {TEST_ADMIN_TOKEN}"))
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::OK);

    let bytes = axum::body::to_bytes(resp.into_body(), usize::MAX)
        .await
        .unwrap();
    let body: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(body["object"], "list");
    let data = body["data"].as_array().expect("data should be array");
    assert_eq!(data.len(), 2, "expected 2 keys, got {}", data.len());

    // Verify no hash fields are exposed
    for item in data {
        assert!(item.get("hmac").is_none(), "hmac should not be in response");
        assert!(item.get("argon2_hash").is_none(), "argon2_hash should not be in response");
        assert!(item.get("key_hash").is_none(), "key_hash should not be in response");
        // Required fields present
        assert!(item["id"].is_string());
        assert!(item["user_id"].is_string());
        assert!(item["name"].is_string());
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Test 3: GET /v1/admin/stats returns totals + models
// ─────────────────────────────────────────────────────────────────────────────

#[tokio::test]
async fn stats_returns_totals_and_models() {
    let (app, _dir) = build_test_app();

    let (_, app) = admin_create_key(app, "stats-user-1", "key-s1").await;
    let (_, app) = admin_create_key(app, "stats-user-2", "key-s2").await;

    let resp = app
        .oneshot(
            Request::builder()
                .method("GET")
                .uri("/v1/admin/stats")
                .header("authorization", format!("Bearer {TEST_ADMIN_TOKEN}"))
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::OK);

    let bytes = axum::body::to_bytes(resp.into_body(), usize::MAX)
        .await
        .unwrap();
    let body: serde_json::Value = serde_json::from_slice(&bytes).unwrap();

    let totals = &body["totals"];
    assert!(totals["users"].as_u64().unwrap() >= 2, "expected >= 2 users");
    assert!(totals["active_keys"].as_u64().unwrap() >= 2, "expected >= 2 active keys");
    assert!(totals["requests_today"].is_number());
    assert!(totals["tokens_today"].is_number());
    assert!(totals["conversations"].is_number());

    let models = body["models"].as_array().expect("models should be array");
    assert!(!models.is_empty(), "models list should not be empty");
    let first = &models[0];
    assert!(first["id"].is_string());
    assert!(first["kind"].is_string());
    assert!(first["description"].is_string());

    assert!(body["active_slots"].is_number());
    assert!(body["queue_depth"].is_number());
}

// ─────────────────────────────────────────────────────────────────────────────
// Test 4: GET /v1/admin/users aggregates correctly
// ─────────────────────────────────────────────────────────────────────────────

#[tokio::test]
async fn users_aggregates_correctly() {
    let (app, _dir) = build_test_app();

    // Create 2 keys for user-x and 1 for user-y
    let (_, app) = admin_create_key(app, "user-x", "key-x1").await;
    let (_, app) = admin_create_key(app, "user-x", "key-x2").await;
    let (_, app) = admin_create_key(app, "user-y", "key-y1").await;

    let resp = app
        .oneshot(
            Request::builder()
                .method("GET")
                .uri("/v1/admin/users")
                .header("authorization", format!("Bearer {TEST_ADMIN_TOKEN}"))
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::OK);

    let bytes = axum::body::to_bytes(resp.into_body(), usize::MAX)
        .await
        .unwrap();
    let body: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(body["object"], "list");
    let data = body["data"].as_array().expect("data should be array");

    // Find user-x in the list
    let user_x = data
        .iter()
        .find(|u| u["user_id"] == "user-x")
        .expect("user-x should be in list");
    assert_eq!(
        user_x["key_count"].as_u64().unwrap(),
        2,
        "user-x should have 2 keys"
    );
    assert_eq!(
        user_x["active_key_count"].as_u64().unwrap(),
        2,
        "user-x should have 2 active keys"
    );

    let user_y = data
        .iter()
        .find(|u| u["user_id"] == "user-y")
        .expect("user-y should be in list");
    assert_eq!(
        user_y["key_count"].as_u64().unwrap(),
        1,
        "user-y should have 1 key"
    );
}

// ─────────────────────────────────────────────────────────────────────────────
// Test 5: GET /v1/admin/usage returns records with filters
// ─────────────────────────────────────────────────────────────────────────────

#[tokio::test]
async fn usage_endpoint_returns_filtered_records() {
    let (app, _dir) = build_test_app();

    // Usage table starts empty → should return empty list with no errors
    let resp = app
        .clone()
        .oneshot(
            Request::builder()
                .method("GET")
                .uri("/v1/admin/usage?limit=10")
                .header("authorization", format!("Bearer {TEST_ADMIN_TOKEN}"))
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::OK);

    let bytes = axum::body::to_bytes(resp.into_body(), usize::MAX)
        .await
        .unwrap();
    let body: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(body["object"], "list");
    assert!(body["data"].as_array().is_some());
    assert!(body["count"].is_number());

    // With since=0 and user_id filter — also should return 200 (just empty)
    let resp = app
        .oneshot(
            Request::builder()
                .method("GET")
                .uri("/v1/admin/usage?since=0&user_id=nobody&limit=5")
                .header("authorization", format!("Bearer {TEST_ADMIN_TOKEN}"))
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    let bytes = axum::body::to_bytes(resp.into_body(), usize::MAX)
        .await
        .unwrap();
    let body: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(body["count"].as_u64().unwrap(), 0);
}

// ─────────────────────────────────────────────────────────────────────────────
// Test 6: Export CSV returns correct content-type and disposition
// ─────────────────────────────────────────────────────────────────────────────

#[tokio::test]
async fn export_csv_correct_headers() {
    let (app, _dir) = build_test_app();

    let resp = app
        .oneshot(
            Request::builder()
                .method("GET")
                .uri("/v1/admin/usage/export.csv")
                .header("authorization", format!("Bearer {TEST_ADMIN_TOKEN}"))
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::OK);

    let ct = resp
        .headers()
        .get("content-type")
        .and_then(|v| v.to_str().ok())
        .unwrap_or("");
    assert!(ct.contains("text/csv"), "content-type should be text/csv, got: {ct}");

    let cd = resp
        .headers()
        .get("content-disposition")
        .and_then(|v| v.to_str().ok())
        .unwrap_or("");
    assert!(
        cd.contains("incord-usage-"),
        "content-disposition should contain incord-usage-, got: {cd}"
    );
    assert!(
        cd.contains(".csv"),
        "content-disposition should contain .csv, got: {cd}"
    );

    let bytes = axum::body::to_bytes(resp.into_body(), usize::MAX)
        .await
        .unwrap();
    let csv = String::from_utf8(bytes.to_vec()).unwrap();
    // Should have header row
    assert!(
        csv.contains("request_id"),
        "CSV should contain header row, got: {csv}"
    );
}

// ─────────────────────────────────────────────────────────────────────────────
// Test 7: Admin conversations list + get (admin override)
// ─────────────────────────────────────────────────────────────────────────────

#[tokio::test]
async fn admin_conversations_list_and_get() {
    let (app, _dir) = build_test_app();

    // Create a user key and a conversation
    let (user_key, app) = admin_create_key(app, "conv-admin-user", "key-conv").await;

    let resp = app
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/conversations")
                .header("content-type", "application/json")
                .header("authorization", format!("Bearer {user_key}"))
                .body(Body::from(json!({"title": "admin test conv"}).to_string()))
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

    // Admin can list all conversations
    let resp = app
        .clone()
        .oneshot(
            Request::builder()
                .method("GET")
                .uri("/v1/admin/conversations")
                .header("authorization", format!("Bearer {TEST_ADMIN_TOKEN}"))
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
    assert_eq!(list["object"], "list");
    let data = list["data"].as_array().unwrap();
    assert!(
        data.iter().any(|c| c["id"] == conv_id),
        "admin list should contain the conversation"
    );

    // Admin can get full detail (bypasses user_id check)
    let resp = app
        .oneshot(
            Request::builder()
                .method("GET")
                .uri(format!("/v1/admin/conversations/{conv_id}"))
                .header("authorization", format!("Bearer {TEST_ADMIN_TOKEN}"))
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    let bytes = axum::body::to_bytes(resp.into_body(), usize::MAX)
        .await
        .unwrap();
    let detail: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(detail["id"], conv_id);
    assert_eq!(detail["title"], "admin test conv");
    assert!(detail["messages"].is_array());
}

// ─────────────────────────────────────────────────────────────────────────────
// Test 8: Admin can list guests and see IP
// ─────────────────────────────────────────────────────────────────────────────

#[tokio::test]
async fn admin_can_list_guests() {
    let (app, _dir) = build_test_app();

    // Send a chat as a guest with X-Forwarded-For
    let chat_body = json!({
        "model": "admin-test-llm",
        "messages": [{"role": "user", "content": "hello from guest"}],
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
                .header("x-guest-id", "guest-m21-list")
                .header("x-forwarded-for", "1.2.3.4")
                .body(Body::from(chat_body.to_string()))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::OK, "guest chat should succeed");

    // GET /v1/admin/guests
    let resp = app
        .clone()
        .oneshot(
            Request::builder()
                .method("GET")
                .uri("/v1/admin/guests")
                .header("authorization", format!("Bearer {TEST_ADMIN_TOKEN}"))
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::OK);

    let bytes = axum::body::to_bytes(resp.into_body(), usize::MAX)
        .await
        .unwrap();
    let body: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(body["object"], "list");
    let data = body["data"].as_array().expect("data should be array");

    let guest = data
        .iter()
        .find(|g| g["guest_id"] == "guest-m21-list")
        .expect("guest-m21-list should appear in list");
    assert!(
        guest["message_count"].as_u64().unwrap() >= 2,
        "should have at least 2 messages (user + assistant)"
    );
    assert_eq!(
        guest["last_ip"].as_str().unwrap_or(""),
        "1.2.3.4",
        "last_ip should be 1.2.3.4 from X-Forwarded-For"
    );
}

// ─────────────────────────────────────────────────────────────────────────────
// Test 9: Admin can view guest messages
// ─────────────────────────────────────────────────────────────────────────────

#[tokio::test]
async fn admin_can_view_guest_messages() {
    let (app, _dir) = build_test_app();

    // Send a chat as guest
    let chat_body = json!({
        "model": "admin-test-llm",
        "messages": [{"role": "user", "content": "guest message content"}],
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
                .header("x-guest-id", "guest-m21-msgs")
                .header("x-forwarded-for", "9.8.7.6")
                .body(Body::from(chat_body.to_string()))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::OK, "guest chat should succeed");

    // GET /v1/admin/guests/guest-m21-msgs/messages
    let resp = app
        .clone()
        .oneshot(
            Request::builder()
                .method("GET")
                .uri("/v1/admin/guests/guest-m21-msgs/messages")
                .header("authorization", format!("Bearer {TEST_ADMIN_TOKEN}"))
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::OK);

    let bytes = axum::body::to_bytes(resp.into_body(), usize::MAX)
        .await
        .unwrap();
    let body: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(body["object"], "list");
    let msgs = body["data"].as_array().expect("data should be array");

    assert!(
        msgs.len() >= 2,
        "should have at least 2 messages (user + assistant), got {}",
        msgs.len()
    );

    // Verify user message
    let user_msg = msgs
        .iter()
        .find(|m| m["role"] == "user")
        .expect("should have a user message");
    assert_eq!(user_msg["content"], "guest message content");
    assert_eq!(user_msg["ip"].as_str().unwrap_or(""), "9.8.7.6");
    assert_eq!(user_msg["guest_id"], "guest-m21-msgs");

    // Verify assistant message
    let asst_msg = msgs
        .iter()
        .find(|m| m["role"] == "assistant")
        .expect("should have an assistant message");
    assert_eq!(asst_msg["guest_id"], "guest-m21-msgs");
}

// ─────────────────────────────────────────────────────────────────────────────
// Test 10: User-supplied geo headers take priority; timezone is persisted
// ─────────────────────────────────────────────────────────────────────────────

#[tokio::test]
async fn user_supplied_geo_headers_take_priority() {
    let (app, _dir) = build_test_app();

    // Send chat as guest with all four geo headers set by the frontend.
    let chat_body = json!({
        "model": "admin-test-llm",
        "messages": [{"role": "user", "content": "hi from Berlin"}],
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
                .header("x-guest-id", "g-geo")
                .header("x-user-country", "DE")
                .header("x-user-city", "Berlin")
                .header("x-user-region", "Berlin")
                .header("x-user-timezone", "Europe/Berlin")
                .body(Body::from(chat_body.to_string()))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::OK, "guest chat should succeed");

    // Retrieve messages via admin endpoint and verify geo fields were persisted.
    let resp = app
        .clone()
        .oneshot(
            Request::builder()
                .method("GET")
                .uri("/v1/admin/guests/g-geo/messages")
                .header("authorization", format!("Bearer {TEST_ADMIN_TOKEN}"))
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::OK);

    let bytes = axum::body::to_bytes(resp.into_body(), usize::MAX)
        .await
        .unwrap();
    let body: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
    let msgs = body["data"].as_array().expect("data should be an array");

    let user_msg = msgs
        .iter()
        .find(|m| m["role"] == "user")
        .expect("should have a user message");

    assert_eq!(user_msg["country"].as_str().unwrap_or(""), "DE");
    assert_eq!(user_msg["city"].as_str().unwrap_or(""), "Berlin");
    assert_eq!(user_msg["timezone"].as_str().unwrap_or(""), "Europe/Berlin");
}
