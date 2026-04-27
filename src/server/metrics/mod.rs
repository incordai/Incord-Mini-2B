//! Prometheus metrics for QORA API server.

use once_cell::sync::Lazy;
use prometheus::{
    Counter, CounterVec, Encoder, Gauge, HistogramVec, TextEncoder, register_counter,
    register_counter_vec, register_gauge, register_histogram_vec,
};

pub static REQUESTS_TOTAL: Lazy<CounterVec> = Lazy::new(|| {
    register_counter_vec!(
        "qora_requests_total",
        "Total HTTP requests by endpoint and status",
        &["endpoint", "status"]
    )
    .unwrap()
});

pub static TOKENS_EMITTED: Lazy<Counter> = Lazy::new(|| {
    register_counter!("qora_tokens_emitted_total", "Total output tokens emitted").unwrap()
});

pub static REQUEST_DURATION: Lazy<HistogramVec> = Lazy::new(|| {
    register_histogram_vec!(
        "qora_request_duration_seconds",
        "Request duration in seconds",
        &["endpoint"],
        vec![0.1, 0.5, 1.0, 5.0, 10.0, 30.0, 60.0, 120.0, 300.0]
    )
    .unwrap()
});

pub static ACTIVE_SLOTS: Lazy<Gauge> =
    Lazy::new(|| register_gauge!("qora_active_slots", "Currently active slots").unwrap());

pub static QUEUE_DEPTH: Lazy<Gauge> =
    Lazy::new(|| register_gauge!("qora_queue_depth", "Pending requests in queue").unwrap());

pub static ERRORS_TOTAL: Lazy<CounterVec> = Lazy::new(|| {
    register_counter_vec!("qora_errors_total", "Total errors by type", &["error_type"]).unwrap()
});

pub fn render_metrics() -> String {
    let mut buf = Vec::new();
    let encoder = TextEncoder::new();
    let metrics = prometheus::gather();
    encoder.encode(&metrics, &mut buf).unwrap();
    String::from_utf8(buf).unwrap()
}
