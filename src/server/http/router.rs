use crate::server::auth::admin::{AdminState, admin_guard};
use crate::server::auth::middleware::AuthState;
use crate::server::http::handlers;
use crate::server::http::middleware::request_id_middleware;
use axum::{
    Router, middleware as axum_mw,
    routing::{delete, get, post},
};
use std::sync::Arc;

pub fn build_router(auth: Arc<AuthState>, admin: Arc<AdminState>) -> Router {
    Router::new()
        .route("/health", get(handlers::health::health))
        .route("/v1/models", get(handlers::models::list_models))
        .route(
            "/v1/admin/keys",
            post(handlers::admin::create_key_handler)
                .layer(axum_mw::from_fn_with_state(admin.clone(), admin_guard)),
        )
        .route(
            "/v1/admin/keys/{id}",
            delete(handlers::admin::revoke_key_handler)
                .layer(axum_mw::from_fn_with_state(admin.clone(), admin_guard)),
        )
        .with_state(auth.clone())
        .layer(axum_mw::from_fn(request_id_middleware))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::server::auth::ratelimit::RateLimiter;
    use crate::server::db::open;
    use axum::body::Body;
    use axum::http::{Request, StatusCode};
    use tempfile::tempdir;
    use tower::ServiceExt;

    fn test_states() -> (Arc<AuthState>, Arc<AdminState>, tempfile::TempDir) {
        let dir = tempdir().unwrap();
        let db = open(&dir.path().join("test.redb")).unwrap();
        let auth = Arc::new(AuthState {
            db,
            server_secret: Arc::new(b"test-secret".to_vec()),
            limiter: RateLimiter::new(),
        });
        let admin = Arc::new(AdminState {
            admin_token: "admin-token".to_string(),
        });
        (auth, admin, dir)
    }

    #[tokio::test]
    async fn health_returns_200() {
        let (auth, admin, _dir) = test_states();
        let app = build_router(auth, admin);
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
        let (auth, admin, _dir) = test_states();
        let app = build_router(auth, admin);
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
        let body_bytes = axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .unwrap();
        let body: serde_json::Value = serde_json::from_slice(&body_bytes).unwrap();
        assert_eq!(body["data"][0]["id"], "qora-4b");
    }

    #[tokio::test]
    async fn unknown_route_returns_404() {
        let (auth, admin, _dir) = test_states();
        let app = build_router(auth, admin);
        let response = app
            .oneshot(Request::builder().uri("/nope").body(Body::empty()).unwrap())
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::NOT_FOUND);
    }

    #[tokio::test]
    async fn response_carries_request_id_header() {
        let (auth, admin, _dir) = test_states();
        let app = build_router(auth, admin);
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
    async fn admin_create_without_token_returns_401() {
        let (auth, admin, _dir) = test_states();
        let app = build_router(auth, admin);
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
