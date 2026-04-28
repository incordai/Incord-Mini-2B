use axum::{
    Router, middleware as axum_mw,
    routing::{delete, get, post},
};
use std::sync::Arc;

use crate::auth::admin::{AdminState, admin_guard};
use crate::auth::middleware::{AuthState, auth_middleware};
use crate::http::handlers::admin::AdminFullState;
use crate::http::handlers::inference::InferenceState;
use crate::http::handlers::{admin, conversations, health, inference, metrics, models};
use crate::http::middleware::request_id_middleware;

pub fn build_router(
    inference_state: Arc<InferenceState>,
    auth: Arc<AuthState>,
    admin_state: Arc<AdminState>,
) -> Router {
    // Build combined state for handlers that need both auth DB and inference/registry/history.
    let admin_full = Arc::new(AdminFullState {
        auth: auth.clone(),
        inference: inference_state.clone(),
    });

    // Admin routes that only need AuthState (key create/revoke/list).
    let admin_auth_routes = Router::new()
        .route("/v1/admin/keys", post(admin::create_key_handler))
        .route("/v1/admin/keys", get(admin::list_keys_handler))
        .route("/v1/admin/keys/{id}", delete(admin::revoke_key_handler))
        .with_state(auth.clone());

    // Admin routes that need AdminFullState (full DB + registry access).
    let admin_full_routes = Router::new()
        .route("/v1/admin/users", get(admin::list_users_handler))
        .route("/v1/admin/usage", get(admin::list_usage_handler))
        .route("/v1/admin/usage/export.csv", get(admin::export_usage_csv_handler))
        .route("/v1/admin/stats", get(admin::stats_handler))
        .route("/v1/admin/conversations", get(admin::list_all_conversations_handler))
        .route(
            "/v1/admin/conversations/{id}",
            get(admin::get_conversation_admin_handler),
        )
        .with_state(admin_full);

    // Merge both admin sub-routers and apply the admin_guard layer to both.
    let admin_routes = Router::new()
        .merge(admin_auth_routes)
        .merge(admin_full_routes)
        .layer(axum_mw::from_fn_with_state(
            admin_state.clone(),
            admin_guard,
        ));

    // Inference routes — protected by user-key auth_middleware
    let inference_routes = Router::new()
        .route("/v1/inference", post(inference::unified_inference))
        .route(
            "/v1/chat/completions",
            post(inference::openai_chat_completions),
        )
        .route("/v1/audio/speech", post(inference::openai_audio_speech))
        .route(
            "/v1/audio/transcriptions",
            post(inference::openai_audio_transcriptions),
        )
        .route("/v1/embeddings", post(inference::openai_embeddings))
        .route("/v1/rerank", post(inference::custom_rerank))
        .with_state(inference_state.clone())
        .route_layer(axum_mw::from_fn_with_state(auth.clone(), auth_middleware));

    // Chat history routes — protected by user-key auth_middleware
    let history_routes = Router::new()
        .route(
            "/v1/conversations",
            post(conversations::create_conversation_handler),
        )
        .route(
            "/v1/conversations",
            get(conversations::list_conversations_handler),
        )
        .route(
            "/v1/conversations/{id}",
            get(conversations::get_conversation_handler),
        )
        .route(
            "/v1/conversations/{id}/export",
            get(conversations::export_conversation_handler),
        )
        .route(
            "/v1/conversations/{id}",
            delete(conversations::delete_conversation_handler),
        )
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
        .merge(history_routes)
        .layer(axum_mw::from_fn(request_id_middleware))
}
