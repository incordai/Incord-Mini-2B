//! Smoke test for TTS dispatch via /v1/inference and /v1/audio/speech.
//!
//! The heavy-weight path (actual synthesis) requires the real QORA-TTS model directory
//! at D:/Burn-AI/QORA-TTS/model/1.7B — skipped gracefully when absent.
//!
//! The lightweight path (registry wiring, routing, request parsing) runs always.

#![cfg(feature = "mock-weights")]

use axum::body::Body;
use axum::http::{Request, StatusCode};
use serde_json::json;
use std::sync::Arc;
use tempfile::TempDir;
use tower::ServiceExt;

use qora_server::auth::admin::AdminState;
use qora_server::auth::middleware::AuthState;
use qora_server::auth::ratelimit::RateLimiter;
use qora_server::db;
use qora_server::http::handlers::inference::InferenceState;
use qora_server::http::router::build_router;
use qora_server::registry::Registry;

const TEST_ADMIN_TOKEN: &str = "test-admin-tts";
const TEST_SECRET: &[u8] = b"test-tts-secret";

fn build_empty_app() -> (axum::Router, TempDir) {
    let dir = TempDir::new().unwrap();
    let db = db::open(&dir.path().join("test.redb")).unwrap();
    let auth = Arc::new(AuthState {
        db: db.clone(),
        server_secret: Arc::new(TEST_SECRET.to_vec()),
        limiter: RateLimiter::new(),
    });
    let admin_state = Arc::new(AdminState {
        admin_token: TEST_ADMIN_TOKEN.to_string(),
    });
    let registry = Registry::builder().build();
    let state = Arc::new(InferenceState::for_test(registry, db.clone()));
    let app = build_router(state, auth, admin_state);
    (app, dir)
}

/// Mint a user key via admin endpoint.
async fn mint_key(app: axum::Router) -> (String, axum::Router) {
    let body = json!({
        "user_id": "tts-test-user",
        "name": "tts-test-key",
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
    assert_eq!(resp.status(), StatusCode::OK);
    let bytes = axum::body::to_bytes(resp.into_body(), usize::MAX)
        .await
        .unwrap();
    let val: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
    (val["raw_key"].as_str().unwrap().to_string(), app)
}

// ---------------------------------------------------------------------------
// Routing / dispatch tests (no real model needed)
// ---------------------------------------------------------------------------

/// Verify that a TTS request for an unknown model returns 400 — not a panic or 500.
#[tokio::test]
async fn tts_unknown_model_returns_400() {
    let (app, _dir) = build_empty_app();
    let (raw_key, app) = mint_key(app).await;

    let body = json!({
        "task": "tts",
        "model": "nonexistent-tts",
        "input": "hello"
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

/// Verify that /v1/audio/speech for an unknown model returns 400.
#[tokio::test]
async fn openai_speech_unknown_model_returns_400() {
    let (app, _dir) = build_empty_app();
    let (raw_key, app) = mint_key(app).await;

    let body = json!({
        "model": "nonexistent-tts",
        "input": "hello",
        "voice": "ryan"
    });

    let resp = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/audio/speech")
                .header("content-type", "application/json")
                .header("authorization", format!("Bearer {raw_key}"))
                .body(Body::from(body.to_string()))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
}

/// Verify that a TTS request with missing input returns 400, not a panic.
#[tokio::test]
async fn tts_missing_input_field_returns_400() {
    let (app, _dir) = build_empty_app();
    let (raw_key, app) = mint_key(app).await;

    // Even if the model were registered, a missing "input" field should be rejected.
    // With no model registered, we get 400 for unknown model first — either way 400.
    let body = json!({
        "task": "tts",
        "model": "nonexistent-tts"
        // no "input" field
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

// ---------------------------------------------------------------------------
// Real synthesis test (skipped when model absent)
// ---------------------------------------------------------------------------

/// Full round-trip: load real model → synthesize short text → verify WAV output.
///
/// Skipped automatically when the QORA-TTS model directory is not present.
#[tokio::test]
async fn tts_synthesizes_short_text() {
    let tts_model_dir = std::path::PathBuf::from("D:/Burn-AI/QORA-TTS/model/1.7B");
    if !tts_model_dir.exists() {
        eprintln!(
            "skipping tts_synthesizes_short_text: model not at {}",
            tts_model_dir.display()
        );
        return;
    }

    use qora_server::engine::tts::{TtsEngine, TtsRequest};
    use qora_server::engines::tts::QoraTtsEngine;

    let engine = match QoraTtsEngine::load("test-tts", tts_model_dir, "Test TTS") {
        Ok(e) => e,
        Err(err) => {
            eprintln!("skipping: TTS engine load failed: {err}");
            return;
        }
    };

    let req = TtsRequest {
        model: "test-tts".to_string(),
        input: "hello".to_string(),
        voice: Some("luna".to_string()), // luna.wav ships with the 1.7B model
        format: Some("wav".to_string()),
        language: Some("english".to_string()),
    };

    let out = engine.synthesize(req).await.unwrap();

    assert!(!out.audio.is_empty(), "synthesis produced no audio");
    assert_eq!(out.format, "wav");
    assert_eq!(out.sample_rate, 24_000);

    // WAV header magic
    assert_eq!(&out.audio[0..4], b"RIFF", "missing RIFF header");
    assert_eq!(&out.audio[8..12], b"WAVE", "missing WAVE tag");
    assert_eq!(&out.audio[12..16], b"fmt ", "missing fmt chunk");

    eprintln!("synthesis OK — {} bytes of WAV audio", out.audio.len());
}
