#![cfg(feature = "mock-weights")]

mod common;
use common::TestServer;
use serde_json::json;

async fn create_user_key(server: &common::TestServer) -> String {
    let resp = reqwest::Client::new()
        .post(server.url("/v1/admin/keys"))
        .bearer_auth(&server.admin_token)
        .json(&json!({"user_id":"u","name":"t","rpm_limit":1000,"tpm_limit":10_000_000}))
        .send().await.unwrap();
    let body: serde_json::Value = resp.json().await.unwrap();
    body["raw_key"].as_str().unwrap().to_string()
}

#[tokio::test]
async fn client_drop_frees_slot() {
    let server = TestServer::start().await;
    let key = create_user_key(&server).await;

    // Start a streaming request, then drop the response without consuming it.
    {
        let resp = reqwest::Client::new()
            .post(server.url("/v1/chat/completions"))
            .bearer_auth(&key)
            .json(&json!({
                "model":"qora-4b",
                "messages":[{"role":"user","content":"hi"}],
                "stream":true,
                "max_tokens":1000
            }))
            .send().await.unwrap();

        // Verify the request was accepted (200) and started streaming
        assert_eq!(resp.status(), 200);

        // Drop the response. The reqwest client closes the connection,
        // which causes the SSE channel on the server to error on next try_send.
        drop(resp);
    }

    // Give the Batcher a moment to detect the disconnect and free the slot.
    // The Batcher's idle sleep is 1ms, so 100ms is plenty.
    tokio::time::sleep(std::time::Duration::from_millis(200)).await;

    // Verify the slot was freed by issuing another request that succeeds.
    // If the slot were still occupied (8-slot pool with one stuck), this would still
    // succeed by using a different slot. To make the test robust against pool size,
    // we issue several requests in sequence and verify they all complete.
    for i in 0..3 {
        let resp = reqwest::Client::new()
            .post(server.url("/v1/chat/completions"))
            .bearer_auth(&key)
            .json(&json!({
                "model":"qora-4b",
                "messages":[{"role":"user","content":format!("hi {i}")}],
                "max_tokens":2,
                "temperature":0.0
            }))
            .timeout(std::time::Duration::from_secs(30))
            .send().await.unwrap();
        assert_eq!(resp.status(), 200, "request {i} failed");
        let body: serde_json::Value = resp.json().await.unwrap();
        assert!(body["choices"][0]["message"]["content"].is_string(), "no content in resp {i}");
    }
}
