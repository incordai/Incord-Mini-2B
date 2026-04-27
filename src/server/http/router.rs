use crate::server::auth::admin::{AdminState, admin_guard};
use crate::server::auth::middleware::{AuthState, auth_middleware};
use crate::server::http::handlers::{self, chat::ChatState};
use crate::server::http::middleware::request_id_middleware;
use axum::{
    Router, middleware as axum_mw,
    routing::{delete, get, post},
};
use std::sync::Arc;

pub fn build_router(
    auth: Arc<AuthState>,
    admin: Arc<AdminState>,
    chat_state: Arc<ChatState>,
) -> Router {
    let admin_routes = Router::new()
        .route("/v1/admin/keys", post(handlers::admin::create_key_handler))
        .route(
            "/v1/admin/keys/{id}",
            delete(handlers::admin::revoke_key_handler),
        )
        .with_state(auth.clone())
        .layer(axum_mw::from_fn_with_state(admin.clone(), admin_guard));

    let chat_routes = Router::new()
        .route(
            "/v1/chat/completions",
            post(handlers::chat::chat_completions),
        )
        .route_layer(axum_mw::from_fn_with_state(auth.clone(), auth_middleware))
        .with_state(chat_state);

    Router::new()
        .route("/health", get(handlers::health::health))
        .route("/v1/models", get(handlers::models::list_models))
        .merge(admin_routes)
        .merge(chat_routes)
        .layer(axum_mw::from_fn(request_id_middleware))
}

#[cfg(all(test, feature = "mock-weights"))]
mod tests {
    use super::*;
    use crate::server::auth::ratelimit::RateLimiter;
    use crate::server::db::open;
    use crate::server::scheduler::queue::Queue;
    use crate::server::usage::tracker::UsageTracker;
    use axum::body::Body;
    use axum::http::{Request, StatusCode};
    use tempfile::tempdir;
    use tower::ServiceExt;

    fn test_states() -> (
        Arc<AuthState>,
        Arc<AdminState>,
        Arc<ChatState>,
        tempfile::TempDir,
    ) {
        let dir = tempdir().unwrap();
        let db = open(&dir.path().join("test.redb")).unwrap();
        let auth = Arc::new(AuthState {
            db: db.clone(),
            server_secret: Arc::new(b"test-secret".to_vec()),
            limiter: RateLimiter::new(),
        });
        let admin = Arc::new(AdminState {
            admin_token: "admin-token".to_string(),
        });

        // Build a tiny ChatState. We don't actually exercise it in router tests —
        // those tests only hit /health, /v1/models, /admin/* and unauth chat (returns 401 before touching state).
        let queue = Queue::new(10);
        let weights = Arc::new(crate::gemv::mock::tiny_weights());
        let tokenizer = build_test_tokenizer();
        let usage = UsageTracker::new(db.clone());
        let vision_semaphore = Arc::new(tokio::sync::Semaphore::new(1));
        let chat_state = Arc::new(ChatState {
            queue,
            weights,
            tokenizer,
            usage,
            vision_semaphore,
        });

        (auth, admin, chat_state, dir)
    }

    /// Construct a tokenizer for tests. The real tokenizer.json is not always available;
    /// fall back to whatever the codebase exposes for tests. If the production tokenizer
    /// requires a tokenizer.json file, this helper should panic with a clear message
    /// directing the engineer to set QORA_TEST_TOKENIZER env var or similar.
    fn build_test_tokenizer() -> Arc<crate::tokenizer::QoraTokenizer> {
        // For mock-weights builds, look for the tokenizer at a well-known test fixture path
        // OR the model dir alongside model.qor4b.
        let candidates = ["model/tokenizer.json", "tests/fixtures/tokenizer.json"];
        for path in candidates {
            if std::path::Path::new(path).exists()
                && let Ok(t) =
                    crate::tokenizer::QoraTokenizer::from_file(std::path::Path::new(path))
            {
                return Arc::new(t);
            }
        }
        panic!("router tests need a tokenizer.json file; expected at one of: {candidates:?}");
    }

    #[tokio::test]
    async fn health_returns_200() {
        let (auth, admin, chat, _dir) = test_states();
        let app = build_router(auth, admin, chat);
        let response = app
            .oneshot(
                Request::builder()
                    .uri("/health")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn models_returns_qora_4b() {
        let (auth, admin, chat, _dir) = test_states();
        let app = build_router(auth, admin, chat);
        let response = app
            .oneshot(
                Request::builder()
                    .uri("/v1/models")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn unknown_route_returns_404() {
        let (auth, admin, chat, _dir) = test_states();
        let app = build_router(auth, admin, chat);
        let response = app
            .oneshot(Request::builder().uri("/nope").body(Body::empty()).unwrap())
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::NOT_FOUND);
    }

    #[tokio::test]
    async fn response_carries_request_id_header() {
        let (auth, admin, chat, _dir) = test_states();
        let app = build_router(auth, admin, chat);
        let response = app
            .oneshot(
                Request::builder()
                    .uri("/health")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert!(response.headers().contains_key("x-request-id"));
    }

    #[tokio::test]
    async fn chat_without_auth_returns_401() {
        let (auth, admin, chat, _dir) = test_states();
        let app = build_router(auth, admin, chat);
        let body = serde_json::json!({"model":"qora-4b","messages":[]});
        let response = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/v1/chat/completions")
                    .header("content-type", "application/json")
                    .body(Body::from(body.to_string()))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::UNAUTHORIZED);
    }

    #[tokio::test]
    async fn admin_create_without_token_returns_401() {
        let (auth, admin, chat, _dir) = test_states();
        let app = build_router(auth, admin, chat);
        let body = serde_json::json!({"user_id":"u","name":"n","rpm_limit":1,"tpm_limit":1});
        let response = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/v1/admin/keys")
                    .header("content-type", "application/json")
                    .body(Body::from(body.to_string()))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::UNAUTHORIZED);
    }
}
