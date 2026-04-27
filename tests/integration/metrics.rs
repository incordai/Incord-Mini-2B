#![cfg(feature = "mock-weights")]

mod common;
use common::TestServer;

#[tokio::test]
async fn metrics_endpoint_returns_prometheus_text() {
    let server = TestServer::start().await;
    let resp = reqwest::get(server.url("/metrics")).await.unwrap();
    assert_eq!(resp.status(), 200);
    let ct = resp
        .headers()
        .get("content-type")
        .unwrap()
        .to_str()
        .unwrap();
    assert!(ct.contains("text/plain"));
    let body = resp.text().await.unwrap();
    // Verify at least one of our custom metrics appears
    assert!(
        body.contains("qora_requests_total")
            || body.contains("qora_active_slots")
            || body.contains("qora_queue_depth"),
        "expected qora_* metric in body: {body}"
    );
}

#[tokio::test]
async fn metrics_endpoint_does_not_require_auth() {
    // /metrics is a system endpoint scraped by Prometheus — no API key needed.
    let server = TestServer::start().await;
    let resp = reqwest::get(server.url("/metrics")).await.unwrap();
    assert_eq!(resp.status(), 200);
}
