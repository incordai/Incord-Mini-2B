#![cfg(feature = "mock-weights")]

mod common;
use common::TestServer;
use futures::stream::{self, StreamExt};
use serde_json::json;

async fn create_key(server: &common::TestServer) -> String {
    let resp = reqwest::Client::new()
        .post(server.url("/v1/admin/keys"))
        .bearer_auth(&server.admin_token)
        .json(&json!({"user_id":"u","name":"t","rpm_limit":10_000,"tpm_limit":100_000_000}))
        .send()
        .await
        .unwrap();
    let body: serde_json::Value = resp.json().await.unwrap();
    body["raw_key"].as_str().unwrap().to_string()
}

#[tokio::test]
async fn sixty_four_concurrent_requests_all_complete() {
    let server = TestServer::start().await;
    let key = create_key(&server).await;

    let client = reqwest::Client::new();
    let url = server.url("/v1/chat/completions");

    // Issue 64 concurrent requests via buffer_unordered for true parallelism.
    let futs = (0..64u32).map(|i| {
        let client = client.clone();
        let key = key.clone();
        let url = url.clone();
        async move {
            let resp = client
                .post(&url)
                .bearer_auth(&key)
                .json(&json!({
                    "model":"qora-4b",
                    "messages":[{"role":"user","content":format!("hi {i}")}],
                    "max_tokens":2,
                    "temperature":0.0
                }))
                .timeout(std::time::Duration::from_secs(60))
                .send()
                .await
                .expect("send failed");
            assert_eq!(resp.status(), 200, "req {i} status {}", resp.status());
            let body: serde_json::Value = resp.json().await.expect("json failed");
            assert!(
                body["choices"][0]["message"]["content"].is_string(),
                "req {i} missing content: {body}"
            );
            i
        }
    });

    let results: Vec<u32> = stream::iter(futs).buffer_unordered(64).collect().await;
    assert_eq!(results.len(), 64, "lost some requests in flight");

    // Verify every i in 0..64 was processed exactly once
    let mut sorted = results.clone();
    sorted.sort_unstable();
    for (idx, val) in sorted.iter().enumerate() {
        assert_eq!(*val, idx as u32, "request gap at index {idx}");
    }
}

#[tokio::test]
async fn sequential_requests_after_burst_work() {
    // Verify slot pool fully recovers after a burst — issue burst, then more requests.
    let server = TestServer::start().await;
    let key = create_key(&server).await;

    // Burst of 32
    let client = reqwest::Client::new();
    let url = server.url("/v1/chat/completions");

    let burst: Vec<u32> = stream::iter((0..32u32).map(|i| {
        let c = client.clone();
        let k = key.clone();
        let u = url.clone();
        async move {
            let resp = c
                .post(&u)
                .bearer_auth(&k)
                .json(&json!({
                    "model":"qora-4b",
                    "messages":[{"role":"user","content":format!("hi {i}")}],
                    "max_tokens":2,
                    "temperature":0.0
                }))
                .timeout(std::time::Duration::from_secs(60))
                .send()
                .await
                .unwrap();
            assert_eq!(resp.status(), 200);
            i
        }
    }))
    .buffer_unordered(32)
    .collect()
    .await;
    assert_eq!(burst.len(), 32);

    // Then 5 sequential requests — should still work.
    for i in 0..5 {
        let resp = client
            .post(&url)
            .bearer_auth(&key)
            .json(&json!({
                "model":"qora-4b",
                "messages":[{"role":"user","content":format!("seq {i}")}],
                "max_tokens":2,
                "temperature":0.0
            }))
            .timeout(std::time::Duration::from_secs(30))
            .send()
            .await
            .unwrap();
        assert_eq!(resp.status(), 200, "sequential req {i} failed");
    }
}
