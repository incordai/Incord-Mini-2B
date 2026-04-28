use std::sync::Arc;
use axum::{Router, middleware as axum_mw, routing::{delete, get, post}};

use crate::auth::admin::{admin_guard, AdminState};
use crate::auth::middleware::{auth_middleware, AuthState};
use crate::http::handlers::{admin, health, inference, metrics, models};
use crate::http::handlers::inference::InferenceState;
use crate::http::middleware::request_id_middleware;

pub fn build_router(
    inference_state: Arc<InferenceState>,
    auth: Arc<AuthState>,
    admin_state: Arc<AdminState>,
) -> Router {
    // Admin routes — protected by admin_guard (separate token from user keys)
    let admin_routes = Router::new()
        .route("/v1/admin/keys", post(admin::create_key_handler))
        .route("/v1/admin/keys/{id}", delete(admin::revoke_key_handler))
        .with_state(auth.clone())
        .layer(axum_mw::from_fn_with_state(admin_state.clone(), admin_guard));

    // Inference routes — protected by user-key auth_middleware
    let inference_routes = Router::new()
        .route("/v1/inference", post(inference::unified_inference))
        .route("/v1/chat/completions", post(inference::openai_chat_completions))
        .route("/v1/audio/speech", post(inference::openai_audio_speech))
        .route("/v1/audio/transcriptions", post(inference::openai_audio_transcriptions))
        .route("/v1/embeddings", post(inference::openai_embeddings))
        .route("/v1/rerank", post(inference::custom_rerank))
        .with_state(inference_state.clone())
        .route_layer(axum_mw::from_fn_with_state(auth.clone(), auth_middleware));

    // Open routes (no auth) — health, models, metrics
    let open_routes = Router::new()
        .route("/health", get(health::health))
        .route("/v1/models", get(models::list_models))
        .route("/metrics", get(metrics::metrics_handler))
        .with_state(inference_state);

    Router::new()
        .merge(open_routes)
        .merge(admin_routes)
        .merge(inference_routes)
        .layer(axum_mw::from_fn(request_id_middleware))
}
