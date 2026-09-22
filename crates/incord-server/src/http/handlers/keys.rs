//! Self-service API key management — endpoints a logged-in user calls to
//! create/list/revoke their OWN keys (no admin token required).
//!
//! Hard rules:
//!   - Endpoint requires `AuthIdentity::User` (session token). API-key-bound
//!     callers cannot create more keys, and guests cannot create any.
//!   - All limits (rpm/tpm/key_count_max) come from the user's tier in
//!     `auth::tiers::limits_for_tier`. Untrusted client input is ignored.
//!   - Revoke requires the key to belong to the calling user. Trying to
//!     revoke someone else's key returns 404, not 403, so we don't leak
//!     existence of other users' keys.

use std::sync::Arc;

use axum::{Extension, Json, extract::{Path, State}};
use serde::{Deserialize, Serialize};
use serde_json::json;

use crate::auth::keys::{
    create_key, get_by_id, list_for_user, revoke_key,
};
use crate::auth::middleware::{AuthIdentity, AuthState};
use crate::auth::tiers::limits_for_tier;
use crate::error::ServerError;

#[derive(Debug, Deserialize)]
pub struct CreateMyKeyRequest {
    pub name: String,
}

#[derive(Debug, Serialize)]
pub struct CreateMyKeyResponse {
    pub id: String,
    pub raw_key: String,
    pub name: String,
    pub user_id: String,
    pub created_at: i64,
    pub rpm_limit: u32,
    pub tpm_limit: u32,
}

/// POST /v1/keys — create a new API key for the current user.
/// Limits come from the user's tier; the request body only carries `name`.
pub async fn create_my_key(
    State(state): State<Arc<AuthState>>,
    Extension(identity): Extension<AuthIdentity>,
    Json(req): Json<CreateMyKeyRequest>,
) -> Result<Json<CreateMyKeyResponse>, ServerError> {
    let user = match identity {
        AuthIdentity::User { user } => user,
        _ => return Err(ServerError::Unauthorized),
    };

    if req.name.trim().is_empty() {
        return Err(ServerError::BadRequest("name is required".into()));
    }
    if req.name.len() > 80 {
        return Err(ServerError::BadRequest("name too long (max 80 chars)".into()));
    }

    let limits = limits_for_tier(&user.tier);

    // Hard subscription gate: free tier has key_count_max=0, so this branch
    // ALSO catches every "no subscription, no key" attempt with a clear
    // message the frontend can route to the upgrade flow.
    if limits.key_count_max == 0 {
        return Err(ServerError::BadRequest(format!(
            "Your current plan ({}) does not allow API keys. Upgrade to Starter, Pro, or Max to create programmatic keys.",
            user.tier
        )));
    }

    // Enforce key-count quota: count ACTIVE keys for this user. Revoked keys
    // don't count against the limit.
    let existing = list_for_user(&state.db, &user.tenant_id, &user.id)
        .map_err(|e| ServerError::Db(e.to_string()))?;
    let active = existing.iter().filter(|k| k.is_active).count() as u32;
    if active >= limits.key_count_max {
        return Err(ServerError::BadRequest(format!(
            "Key limit reached ({}/{} active keys on {} tier). Revoke an existing key or upgrade your plan.",
            active, limits.key_count_max, user.tier
        )));
    }

    let nk = create_key(
        &state.db,
        &state.server_secret,
        &user.tenant_id,
        &user.id,
        req.name.trim(),
        limits.rpm,
        limits.tpm,
    )
    .map_err(|e| ServerError::Db(e.to_string()))?;

    Ok(Json(CreateMyKeyResponse {
        id: nk.key.id,
        raw_key: nk.raw_key,
        name: nk.key.name,
        user_id: nk.key.user_id,
        created_at: nk.key.created_at,
        rpm_limit: nk.key.rpm_limit,
        tpm_limit: nk.key.tpm_limit,
    }))
}

/// GET /v1/keys — list ONLY the calling user's keys.
/// Returns the public ApiKey shape (no hash material, no raw_key).
pub async fn list_my_keys(
    State(state): State<Arc<AuthState>>,
    Extension(identity): Extension<AuthIdentity>,
) -> Result<Json<serde_json::Value>, ServerError> {
    let user = match identity {
        AuthIdentity::User { user } => user,
        _ => return Err(ServerError::Unauthorized),
    };

    let keys = list_for_user(&state.db, &user.tenant_id, &user.id)
        .map_err(|e| ServerError::Db(e.to_string()))?;

    let limits = limits_for_tier(&user.tier);
    let active = keys.iter().filter(|k| k.is_active).count() as u32;

    let data: Vec<_> = keys.iter().map(|k| json!({
        "id": k.id,
        "name": k.name,
        "created_at": k.created_at,
        "last_used_at": k.last_used_at,
        "is_active": k.is_active,
        "rpm_limit": k.rpm_limit,
        "tpm_limit": k.tpm_limit,
    })).collect();

    Ok(Json(json!({
        "object": "list",
        "data": data,
        "tier": user.tier,
        "key_count_active": active,
        "key_count_max": limits.key_count_max,
    })))
}

/// DELETE /v1/keys/{id} — revoke ONE of the calling user's keys.
/// 404 (not 403) if the key belongs to someone else, to avoid leaking
/// existence of other users' keys.
pub async fn revoke_my_key(
    State(state): State<Arc<AuthState>>,
    Extension(identity): Extension<AuthIdentity>,
    Path(id): Path<String>,
) -> Result<Json<serde_json::Value>, ServerError> {
    let user = match identity {
        AuthIdentity::User { user } => user,
        _ => return Err(ServerError::Unauthorized),
    };

    let key = get_by_id(&state.db, &user.tenant_id, &id)
        .map_err(|e| ServerError::Db(e.to_string()))?;

    // Treat "not yours" identically to "doesn't exist".
    let key = match key {
        Some(k) if k.user_id == user.id => k,
        _ => return Err(ServerError::BadRequest(format!("no key with id {id}"))),
    };

    if !key.is_active {
        return Ok(Json(json!({"id": id, "revoked": true, "already_inactive": true})));
    }

    let ok = revoke_key(&state.db, &user.tenant_id, &id)
        .map_err(|e| ServerError::Db(e.to_string()))?;
    if !ok {
        return Err(ServerError::BadRequest(format!("no key with id {id}")));
    }
    Ok(Json(json!({"id": id, "revoked": true})))
}
