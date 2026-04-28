use crate::auth::keys::{create_key, revoke_key};
use crate::auth::middleware::AuthState;
use crate::error::ServerError;
use axum::{
    Json,
    extract::{Path, State},
};
use serde::{Deserialize, Serialize};
use std::sync::Arc;

#[derive(Deserialize)]
pub struct CreateKeyRequest {
    pub user_id: String,
    pub name: String,
    pub rpm_limit: u32,
    pub tpm_limit: u32,
}

#[derive(Serialize)]
pub struct CreateKeyResponse {
    pub id: String,
    pub raw_key: String,
    pub user_id: String,
    pub created_at: i64,
}

pub async fn create_key_handler(
    State(state): State<Arc<AuthState>>,
    Json(req): Json<CreateKeyRequest>,
) -> Result<Json<CreateKeyResponse>, ServerError> {
    let nk = create_key(
        &state.db,
        &state.server_secret,
        &req.user_id,
        &req.name,
        req.rpm_limit,
        req.tpm_limit,
    )
    .map_err(|e| ServerError::Db(e.to_string()))?;
    Ok(Json(CreateKeyResponse {
        id: nk.key.id,
        raw_key: nk.raw_key,
        user_id: nk.key.user_id,
        created_at: nk.key.created_at,
    }))
}

pub async fn revoke_key_handler(
    State(state): State<Arc<AuthState>>,
    Path(id): Path<String>,
) -> Result<Json<serde_json::Value>, ServerError> {
    let ok = revoke_key(&state.db, &id).map_err(|e| ServerError::Db(e.to_string()))?;
    if !ok {
        return Err(ServerError::BadRequest(format!("no key with id {id}")));
    }
    Ok(Json(serde_json::json!({"id": id, "revoked": true})))
}
