use std::sync::Arc;

use axum::{Extension, Json, extract::State};
use serde::{Deserialize, Serialize};
use serde_json::json;

use crate::auth::middleware::{AuthIdentity, AuthState};
use crate::auth::{sessions, users};
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

pub async fn register_handler(
    State(state): State<Arc<AuthState>>,
    Json(req): Json<RegisterRequest>,
) -> Result<Json<AuthResponse>, ServerError> {
    let user = users::register_user(&state.db, &req.email, &req.password)
        .map_err(|e| ServerError::BadRequest(e.to_string()))?;
    let session = sessions::create_session(&state.db, &user.id)
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
    Json(req): Json<RegisterRequest>,
) -> Result<Json<AuthResponse>, ServerError> {
    let user = users::login(&state.db, &req.email, &req.password)
        .map_err(|e| ServerError::Db(e.to_string()))?
        .ok_or_else(|| ServerError::BadRequest("invalid email or password".into()))?;
    let session = sessions::create_session(&state.db, &user.id)
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
    // For session-based auth, revoking the token would require storing it in the
    // extension — a future enhancement. For now, returning logged_out: true is
    // sufficient; the frontend drops the stored token on its end.
    Ok(Json(json!({"logged_out": true})))
}

pub async fn me_handler(
    Extension(identity): Extension<AuthIdentity>,
) -> Json<serde_json::Value> {
    Json(match &identity {
        AuthIdentity::User { user } => json!({
            "type": "user",
            "user_id": user.id,
            "email": user.email,
            "tier": user.tier,
        }),
        AuthIdentity::ApiKey { key } => json!({
            "type": "api_key",
            "key_id": key.id,
            "user_id": key.user_id,
        }),
        AuthIdentity::Guest {
            guest_id,
            msg_count,
            max_messages,
        } => json!({
            "type": "guest",
            "guest_id": guest_id,
            "messages_used": msg_count,
            "messages_max": max_messages,
        }),
    })
}
