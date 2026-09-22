#![cfg(feature = "mock-weights")]

mod common;
use common::TestServer;

#[tokio::test]
async fn health_endpoint_returns_200() {
    let server = TestServer::start().await;
    let resp = reqwest::get(server.url("/health")).await.unwrap();
    assert_eq!(resp.status(), 200);
    let body: serde_json::Value = resp.json().await.unwrap();
    assert_eq!(body["status"], "ok");
}

#[tokio::test]
async fn models_endpoint_lists_incord_models() {
    let server = TestServer::start().await;
    let resp = reqwest::get(server.url("/v1/models")).await.unwrap();
    assert_eq!(resp.status(), 200);
    let body: serde_json::Value = resp.json().await.unwrap();
    assert_eq!(body["object"], "list");
    assert_eq!(body["data"][0]["id"], "incord-mini-2b");
    assert_eq!(body["data"][1]["id"], "incord");
    assert_eq!(body["data"][2]["id"], "qora-4b");
}

#[tokio::test]
async fn unknown_route_returns_404() {
    let server = TestServer::start().await;
    let resp = reqwest::get(server.url("/nope")).await.unwrap();
    assert_eq!(resp.status(), 404);
}

#[tokio::test]
async fn response_includes_x_request_id() {
    let server = TestServer::start().await;
    let resp = reqwest::get(server.url("/health")).await.unwrap();
    assert!(resp.headers().contains_key("x-request-id"));
}
