#![cfg(feature = "mock-weights")]

mod common;
use common::TestServer;
use serde_json::json;

async fn create_user_key(server: &common::TestServer) -> String {
    let resp = reqwest::Client::new()
        .post(server.url("/v1/admin/keys"))
        .bearer_auth(&server.admin_token)
        .json(&json!({"user_id":"u1","name":"t","rpm_limit":1000,"tpm_limit":1_000_000}))
        .send()
        .await
        .unwrap();
    let body: serde_json::Value = resp.json().await.unwrap();
    body["raw_key"].as_str().unwrap().to_string()
}

#[tokio::test]
async fn unknown_model_returns_400() {
    let server = TestServer::start().await;
    let key = create_user_key(&server).await;
    let resp = reqwest::Client::new()
        .post(server.url("/v1/chat/completions"))
        .bearer_auth(&key)
        .json(&json!({"model":"gpt-99","messages":[{"role":"user","content":"hi"}]}))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 400);
}

#[tokio::test]
async fn missing_auth_returns_401() {
    let server = TestServer::start().await;
    let resp = reqwest::Client::new()
        .post(server.url("/v1/chat/completions"))
        .json(&json!({"model":"qora-4b","messages":[{"role":"user","content":"hi"}]}))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 401);
}

#[tokio::test]
async fn non_streaming_returns_completion() {
    let server = TestServer::start().await;
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
        .timeout(std::time::Duration::from_secs(30))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 200);
    let body: serde_json::Value = resp.json().await.unwrap();
    assert_eq!(body["object"], "chat.completion");
    assert!(body["choices"][0]["message"]["content"].is_string());
    assert!(body["usage"]["completion_tokens"].as_u64().unwrap() > 0);
}

#[tokio::test]
async fn streaming_returns_sse_chunks() {
    let server = TestServer::start().await;
    let key = create_user_key(&server).await;

    let resp = reqwest::Client::new()
        .post(server.url("/v1/chat/completions"))
        .bearer_auth(&key)
        .json(&json!({
            "model": "qora-4b",
            "messages": [{"role":"user","content":"hi"}],
            "stream": true,
            "max_tokens": 4,
            "temperature": 0.0
        }))
        .timeout(std::time::Duration::from_secs(30))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 200);

    let body_text = resp.text().await.unwrap();
    assert!(
        body_text.contains("data: "),
        "expected SSE data: prefix in body, got: {body_text}"
    );
    assert!(
        body_text.contains("[DONE]"),
        "expected [DONE] terminator, got: {body_text}"
    );
}

#[tokio::test]
async fn chat_completions_accepts_incord_model_names() {
    let server = TestServer::start().await;
    let key = create_user_key(&server).await;

    for model_name in &["incord", "incord-mini-2b", "incord-2b", "qora-4b"] {
        let resp = reqwest::Client::new()
            .post(server.url("/v1/chat/completions"))
            .bearer_auth(&key)
            .json(&json!({
                "model": model_name,
                "messages": [{"role":"user","content":"hi"}],
                "max_tokens": 2,
                "temperature": 0.0
            }))
            .timeout(std::time::Duration::from_secs(30))
            .send()
            .await
            .unwrap();
        assert_eq!(
            resp.status(),
            200,
            "expected model `{model_name}` to be accepted with 200 OK"
        );
    }
}
