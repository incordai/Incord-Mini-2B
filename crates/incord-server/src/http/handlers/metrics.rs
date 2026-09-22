use axum::http::header;
use axum::response::IntoResponse;

pub async fn metrics_handler() -> impl IntoResponse {
    (
        [(header::CONTENT_TYPE, "text/plain; version=0.0.4")],
        crate::metrics::render_metrics(),
    )
}
