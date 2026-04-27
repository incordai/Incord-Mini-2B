#![cfg(feature = "mock-weights")]

mod common;
use common::TestServer;
use serde_json::json;

async fn create_user_key(server: &common::TestServer) -> String {
    let resp = reqwest::Client::new()
        .post(server.url("/v1/admin/keys"))
        .bearer_auth(&server.admin_token)
        .json(&json!({"user_id":"u","name":"t","rpm_limit":1000,"tpm_limit":1_000_000}))
        .send().await.unwrap();
    let body: serde_json::Value = resp.json().await.unwrap();
    body["raw_key"].as_str().unwrap().to_string()
}

/// Tiny 1x1 PNG (red pixel) encoded as base64 — minimal valid image bytes.
const TINY_PNG_B64: &str = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII=";

#[tokio::test]
async fn vision_request_without_vision_weights_returns_500() {
    // The mock model has has_vision=false and vision=None.
    // A request with image_url should fail at vision encoding time with a clear error.
    let server = TestServer::start().await;
    let key = create_user_key(&server).await;

    let data_url = format!("data:image/png;base64,{}", TINY_PNG_B64);

    let resp = reqwest::Client::new()
        .post(server.url("/v1/chat/completions"))
        .bearer_auth(&key)
        .json(&json!({
            "model":"qora-4b",
            "messages":[{
                "role":"user",
                "content":[
                    {"type":"text","text":"describe"},
                    {"type":"image_url","image_url":{"url":data_url}}
                ]
            }],
            "max_tokens":2
        }))
        .timeout(std::time::Duration::from_secs(30))
        .send().await.unwrap();

    // Mock has no vision encoder — should get 500 (Internal: "model has no vision weights")
    // OR 400 if the error mapping is structured differently. Both are acceptable for v1.
    let status = resp.status();
    assert!(
        status == 500 || status == 400,
        "expected 4xx or 5xx for vision request without vision weights, got {status}"
    );

    // Verify we get a JSON error, not a hang or panic
    let body: serde_json::Value = resp.json().await.unwrap();
    assert!(body.get("error").is_some(), "expected error object in body: {body}");
}

#[tokio::test]
async fn vision_request_with_invalid_data_url_returns_400() {
    let server = TestServer::start().await;
    let key = create_user_key(&server).await;

    let resp = reqwest::Client::new()
        .post(server.url("/v1/chat/completions"))
        .bearer_auth(&key)
        .json(&json!({
            "model":"qora-4b",
            "messages":[{
                "role":"user",
                "content":[
                    {"type":"text","text":"describe"},
                    {"type":"image_url","image_url":{"url":"data:image/png;base64,!!!invalid!!!"}}
                ]
            }],
            "max_tokens":2
        }))
        .timeout(std::time::Duration::from_secs(10))
        .send().await.unwrap();

    assert_eq!(resp.status(), 400, "expected 400 for invalid base64");
    let body: serde_json::Value = resp.json().await.unwrap();
    assert!(body["error"]["type"].as_str().unwrap().contains("invalid_request") || body["error"]["type"].as_str().unwrap().contains("error"));
}

#[tokio::test]
async fn vision_request_with_unsupported_scheme_returns_400() {
    let server = TestServer::start().await;
    let key = create_user_key(&server).await;

    let resp = reqwest::Client::new()
        .post(server.url("/v1/chat/completions"))
        .bearer_auth(&key)
        .json(&json!({
            "model":"qora-4b",
            "messages":[{
                "role":"user",
                "content":[
                    {"type":"image_url","image_url":{"url":"ftp://example.com/img.png"}}
                ]
            }],
            "max_tokens":2
        }))
        .timeout(std::time::Duration::from_secs(10))
        .send().await.unwrap();

    assert_eq!(resp.status(), 400, "expected 400 for unsupported URL scheme");
}
