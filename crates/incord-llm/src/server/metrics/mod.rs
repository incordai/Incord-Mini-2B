//! Prometheus metrics for Incord API server.

use once_cell::sync::Lazy;
use prometheus::{
    Counter, CounterVec, Encoder, Gauge, HistogramVec, TextEncoder, register_counter,
    register_counter_vec, register_gauge, register_histogram_vec,
};

pub static REQUESTS_TOTAL: Lazy<CounterVec> = Lazy::new(|| {
    register_counter_vec!(
        "incord_requests_total",
        "Total HTTP requests by endpoint and status",
        &["endpoint", "status"]
    )
    .unwrap()
});

pub static TOKENS_EMITTED: Lazy<Counter> = Lazy::new(|| {
    register_counter!("incord_tokens_emitted_total", "Total output tokens emitted").unwrap()
});

pub static REQUEST_DURATION: Lazy<HistogramVec> = Lazy::new(|| {
    register_histogram_vec!(
        "incord_request_duration_seconds",
        "Request duration in seconds",
        &["endpoint"],
        vec![0.1, 0.5, 1.0, 5.0, 10.0, 30.0, 60.0, 120.0, 300.0]
    )
    .unwrap()
});

pub static ACTIVE_SLOTS: Lazy<Gauge> =
    Lazy::new(|| register_gauge!("incord_active_slots", "Currently active slots").unwrap());

pub static QUEUE_DEPTH: Lazy<Gauge> =
    Lazy::new(|| register_gauge!("incord_queue_depth", "Pending requests in queue").unwrap());

pub static ERRORS_TOTAL: Lazy<CounterVec> = Lazy::new(|| {
    register_counter_vec!("incord_errors_total", "Total errors by type", &["error_type"]).unwrap()
});

pub fn render_metrics() -> String {
    let mut buf = Vec::new();
    let encoder = TextEncoder::new();
    let metrics = prometheus::gather();
    encoder.encode(&metrics, &mut buf).unwrap();
    String::from_utf8(buf).unwrap()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every metric registers and renders in the Prometheus text format that
    /// `/metrics` serves.
    #[test]
    fn metrics_render_as_prometheus_text() {
        REQUESTS_TOTAL.with_label_values(&["chat", "200"]).inc();
        TOKENS_EMITTED.inc_by(3.0);
        REQUEST_DURATION.with_label_values(&["chat"]).observe(0.2);
        ACTIVE_SLOTS.set(2.0);
        QUEUE_DEPTH.set(1.0);
        ERRORS_TOTAL.with_label_values(&["timeout"]).inc();

        let text = render_metrics();
        for line in [
            "# TYPE incord_requests_total counter",
            "incord_requests_total{endpoint=\"chat\",status=\"200\"}",
            "incord_tokens_emitted_total",
            "incord_request_duration_seconds_bucket{endpoint=\"chat\",le=\"0.5\"}",
            "incord_active_slots 2",
            "incord_queue_depth 1",
            "incord_errors_total{error_type=\"timeout\"}",
        ] {
            assert!(text.contains(line), "missing {line:?} in:\n{text}");
        }
    }
}
