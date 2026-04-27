use axum::{Router, middleware as axum_mw, routing::{get, post}};
use std::sync::Arc;

use crate::http::handlers::{health, inference, models};
use crate::http::handlers::inference::InferenceState;
use crate::http::middleware::request_id_middleware;

pub fn build_router(state: Arc<InferenceState>) -> Router {
    Router::new()
        .route("/health", get(health::health))
        .route("/v1/models", get(models::list_models))
        .route("/v1/inference", post(inference::unified_inference))
        .route("/v1/chat/completions", post(inference::openai_chat_completions))
        .route("/v1/audio/speech", post(inference::openai_audio_speech))
        .with_state(state)
        .layer(axum_mw::from_fn(request_id_middleware))
}
