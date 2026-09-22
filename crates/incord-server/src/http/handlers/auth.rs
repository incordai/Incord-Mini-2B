use std::sync::Arc;

use axum::{
    extract::{Request, State},
    http::HeaderMap,
    Extension, Json,
};
use serde::{Deserialize, Serialize};
use serde_json::json;

use crate::auth::middleware::{AuthIdentity, AuthState};
use crate::auth::{sessions, users};
use crate::db::{is_safe_tenant, DEFAULT_TENANT};
use crate::error::ServerError;

#[derive(Deserialize)]
pub struct RegisterRequest {
    pub email: String,
    pub password: String,
}

#[derive(Serialize)]
pub struct AuthResponse {
    pub user_id: String,
    pub email: String,
    pub tier: String,
    pub session_token: String,
    pub expires_at: i64,
}

/// Read the tenant for register/login flows. These are pre-auth so the
/// tenant comes from the `X-Tenant` header, defaulting to [`DEFAULT_TENANT`].
fn tenant_from_headers(headers: &HeaderMap) -> String {
    headers
        .get("x-tenant")
        .and_then(|v| v.to_str().ok())
        .filter(|s| is_safe_tenant(s))
        .map(|s| s.to_string())
        .unwrap_or_else(|| DEFAULT_TENANT.to_string())
}

pub async fn register_handler(
    State(state): State<Arc<AuthState>>,
    headers: HeaderMap,
    Json(req): Json<RegisterRequest>,
) -> Result<Json<AuthResponse>, ServerError> {
    let tenant = tenant_from_headers(&headers);
    let user = users::register_user(&state.db, &tenant, &req.email, &req.password)
        .map_err(|e| ServerError::BadRequest(e.to_string()))?;
    let session = sessions::create_session(&state.db, &tenant, &user.id)
        .map_err(|e| ServerError::Db(e.to_string()))?;
    Ok(Json(AuthResponse {
        user_id: user.id,
        email: user.email,
        tier: user.tier,
        session_token: session.token,
        expires_at: session.expires_at,
    }))
}

pub async fn login_handler(
    State(state): State<Arc<AuthState>>,
    headers: HeaderMap,
    Json(req): Json<RegisterRequest>,
) -> Result<Json<AuthResponse>, ServerError> {
    let tenant = tenant_from_headers(&headers);
    let user = users::login(&state.db, &tenant, &req.email, &req.password)
        .map_err(|e| ServerError::Db(e.to_string()))?
        .ok_or_else(|| ServerError::BadRequest("invalid email or password".into()))?;
    let session = sessions::create_session(&state.db, &tenant, &user.id)
        .map_err(|e| ServerError::Db(e.to_string()))?;
    Ok(Json(AuthResponse {
        user_id: user.id,
        email: user.email,
        tier: user.tier,
        session_token: session.token,
        expires_at: session.expires_at,
    }))
}

pub async fn logout_handler(
    State(_state): State<Arc<AuthState>>,
    Extension(_identity): Extension<AuthIdentity>,
) -> Result<Json<serde_json::Value>, ServerError> {
    Ok(Json(json!({"logged_out": true})))
}

pub async fn me_handler(
    Extension(identity): Extension<AuthIdentity>,
) -> Json<serde_json::Value> {
    Json(match &identity {
        AuthIdentity::User { user } => json!({
            "type": "user",
            "user_id": user.id,
            "tenant_id": user.tenant_id,
            "email": user.email,
            "tier": user.tier,
        }),
        AuthIdentity::ApiKey { key } => json!({
            "type": "api_key",
            "key_id": key.id,
            "tenant_id": key.tenant_id,
            "user_id": key.user_id,
        }),
        AuthIdentity::Guest {
            tenant_id,
            guest_id,
            msg_count,
            max_messages,
        } => json!({
            "type": "guest",
            "tenant_id": tenant_id,
            "guest_id": guest_id,
            "messages_used": msg_count,
            "messages_max": max_messages,
        }),
    })
}

// Suppress unused-import lint when Request isn't directly referenced.
#[allow(dead_code)]
fn _unused_request_marker(_: Request) {}
