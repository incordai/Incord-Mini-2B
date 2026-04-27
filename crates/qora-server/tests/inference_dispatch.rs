//! Smoke test: registry-driven dispatch via /v1/inference and /v1/chat/completions.
//! Uses a runtime-supplied model id (NOT hardcoded "qora-4b" or "qora-9b") to
//! verify the dispatch logic works for any id.

#![cfg(feature = "mock-weights")]

use std::sync::Arc;
use axum::body::Body;
use axum::http::{Request, StatusCode};
use serde_json::json;
use tower::ServiceExt;

use qor4b::gemv::mock::tiny_weights;
use qor4b::tokenizer::QoraTokenizer;
use qora_server::engines::llm::LlmEngine;
use qora_server::http::handlers::inference::InferenceState;
use qora_server::http::router::build_router;
use qora_server::registry::Registry;

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

fn build_test_state(model_id: &str) -> Arc<InferenceState> {
    let weights = Arc::new(tiny_weights());
    let tokenizer = build_test_tokenizer();
    let engine = Arc::new(LlmEngine::new(
        model_id,
        weights,
        tokenizer,
        format!("Test engine {}", model_id),
    ));
    let registry = Registry::builder().add_chat(engine).build();
    Arc::new(InferenceState { registry })
}

#[tokio::test]
async fn unified_inference_dispatches_to_registered_chat_model() {
    // Use a runtime model id that has nothing to do with "4b" — proves no hardcoding
    let state = build_test_state("any-llm-id");
    let app = build_router(state);

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
    let state = build_test_state("any-other-id");
    let app = build_router(state);

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
    let state = build_test_state("does-exist");
    let app = build_router(state);

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
                .body(Body::from(body.to_string()))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
}

#[tokio::test]
async fn list_models_returns_registry_contents() {
    let state = build_test_state("model-a");
    let app = build_router(state);

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
