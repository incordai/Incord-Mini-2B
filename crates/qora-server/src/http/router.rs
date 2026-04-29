use axum::{
    Router, middleware as axum_mw,
    routing::{delete, get, post},
    http::{HeaderName, Method, header},
};
use std::sync::Arc;
use tower_http::cors::{Any, CorsLayer};

use crate::auth::admin::{AdminState, admin_guard};
use crate::auth::middleware::{AuthState, auth_middleware};
use crate::http::handlers::admin::AdminFullState;
use crate::http::handlers::inference::InferenceState;
use crate::http::handlers::{admin, auth, conversations, health, inference, metrics, models};
use crate::http::middleware::request_id_middleware;

pub fn build_router(
    inference_state: Arc<InferenceState>,
    auth_state: Arc<AuthState>,
    admin_state: Arc<AdminState>,
) -> Router {
    // CORS layer — allow_origin(Any) is intentionally permissive for local dev.
    // In production, the security boundary is enforced by auth_middleware.
    let cors = CorsLayer::new()
        .allow_origin(Any)
        .allow_methods([Method::GET, Method::POST, Method::DELETE, Method::OPTIONS])
        .allow_headers([
            header::AUTHORIZATION,
            header::CONTENT_TYPE,
            header::ACCEPT,
            // Allow the guest-id header from browser clients.
            HeaderName::from_static("x-guest-id"),
        ])
        .expose_headers([header::CONTENT_DISPOSITION]);

    // Build combined state for handlers that need both auth DB and inference/registry/history.
    let admin_full = Arc::new(AdminFullState {
        auth: auth_state.clone(),
        inference: inference_state.clone(),
    });

    // Admin routes that only need AuthState (key create/revoke/list).
    let admin_auth_routes = Router::new()
        .route("/v1/admin/keys", post(admin::create_key_handler))
        .route("/v1/admin/keys", get(admin::list_keys_handler))
        .route("/v1/admin/keys/{id}", delete(admin::revoke_key_handler))
        .with_state(auth_state.clone());

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

    // Open auth routes (register + login — no auth required).
    let auth_open_routes = Router::new()
        .route("/v1/auth/register", post(auth::register_handler))
        .route("/v1/auth/login", post(auth::login_handler))
        .with_state(auth_state.clone());

    // Protected auth routes (me + logout — any valid identity).
    let auth_protected_routes = Router::new()
        .route("/v1/auth/me", get(auth::me_handler))
        .route("/v1/auth/logout", post(auth::logout_handler))
        .with_state(auth_state.clone())
        .route_layer(axum_mw::from_fn_with_state(
            auth_state.clone(),
            auth_middleware,
        ));

    // Inference routes — protected by tiered auth_middleware
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
        .route_layer(axum_mw::from_fn_with_state(
            auth_state.clone(),
            auth_middleware,
        ));

    // Chat history routes — protected by tiered auth_middleware
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
        .route_layer(axum_mw::from_fn_with_state(
            auth_state.clone(),
            auth_middleware,
        ));

    // Open routes (no auth) — health, models, metrics
    let open_routes = Router::new()
        .route("/health", get(health::health))
        .route("/v1/models", get(models::list_models))
        .route("/metrics", get(metrics::metrics_handler))
        .with_state(inference_state);

    Router::new()
        .merge(open_routes)
        .merge(admin_routes)
        .merge(auth_open_routes)
        .merge(auth_protected_routes)
        .merge(inference_routes)
        .merge(history_routes)
        .layer(axum_mw::from_fn(request_id_middleware))
        // CORS layer is outermost so it wraps every route including OPTIONS pre-flight
        .layer(cors)
}
