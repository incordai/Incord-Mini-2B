#![cfg(feature = "mock-weights")]

mod common;
use common::TestServer;
use serde_json::json;

#[tokio::test]
async fn admin_can_create_key() {
    let server = TestServer::start().await;
    let resp = reqwest::Client::new()
        .post(server.url("/v1/admin/keys"))
        .bearer_auth(&server.admin_token)
        .json(&json!({
            "user_id":"user1","name":"test","rpm_limit":60,"tpm_limit":60_000
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 200);
    let body: serde_json::Value = resp.json().await.unwrap();
    assert!(body["raw_key"].as_str().unwrap().starts_with("incord_"));
}

#[tokio::test]
async fn missing_admin_token_returns_401() {
    let server = TestServer::start().await;
    let resp = reqwest::Client::new()
        .post(server.url("/v1/admin/keys"))
        .json(&json!({"user_id":"u","name":"n","rpm_limit":1,"tpm_limit":1}))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 401);
}

#[tokio::test]
async fn wrong_admin_token_returns_401() {
    let server = TestServer::start().await;
    let resp = reqwest::Client::new()
        .post(server.url("/v1/admin/keys"))
        .bearer_auth("wrong-token")
        .json(&json!({"user_id":"u","name":"n","rpm_limit":1,"tpm_limit":1}))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 401);
}

#[tokio::test]
async fn revoke_key_works() {
    let server = TestServer::start().await;
    let resp = reqwest::Client::new()
        .post(server.url("/v1/admin/keys"))
        .bearer_auth(&server.admin_token)
        .json(&json!({"user_id":"u","name":"n","rpm_limit":1,"tpm_limit":1}))
        .send()
        .await
        .unwrap();
    let body: serde_json::Value = resp.json().await.unwrap();
    let id = body["id"].as_str().unwrap().to_string();

    let resp = reqwest::Client::new()
        .delete(server.url(&format!("/v1/admin/keys/{id}")))
        .bearer_auth(&server.admin_token)
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 200);

    // v1 behavior: revoke is idempotent (re-revoking an already-revoked key
    // still finds the record in redb and sets is_active=false again, returning 200).
    let resp = reqwest::Client::new()
        .delete(server.url(&format!("/v1/admin/keys/{id}")))
        .bearer_auth(&server.admin_token)
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 200);
}
