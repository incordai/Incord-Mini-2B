//! Subscription / billing endpoints (placeholder, no Stripe yet).
//!
//! - `POST /v1/billing/subscribe { tier, cycle? }` — change my subscription tier.
//!   Free is instant. Paid tiers are CURRENTLY also instant (placeholder for
//!   manual testing); when Stripe is wired in, this endpoint will redirect the
//!   user to a Stripe Checkout session and the tier will be applied after the
//!   webhook fires.
//!
//! - `GET /v1/billing/me` — return my tier + the limits derived from it
//!   (rpm/tpm/key_count_max/daily_messages). The frontend "API keys" panel
//!   uses this to show "you have 1/3 active keys on Starter tier".
//!
//! - `POST /v1/billing/cancel` — downgrade me to free. Stripe later will
//!   schedule the downgrade for the end of the billing cycle.
//!
//! Authentication: every endpoint requires `AuthIdentity::User` — API keys
//! and guests cannot subscribe (they don't have user records to attach to).

use std::sync::Arc;

use axum::{Extension, Json, extract::State};
use serde::Deserialize;
use serde_json::json;

use crate::auth::middleware::{AuthIdentity, AuthState};
use crate::auth::tiers::{TIER_NAMES, limits_for_tier, tier_exists};
use crate::auth::users::update_user_tier;
use crate::error::ServerError;

#[derive(Debug, Deserialize)]
pub struct SubscribeRequest {
    pub tier: String,
    /// Optional: "monthly" | "quarterly" | "yearly". Stored as a hint for the
    /// future Stripe integration; doesn't affect functional limits today.
    #[serde(default)]
    pub cycle: Option<String>,
}

/// POST /v1/billing/subscribe
pub async fn subscribe(
    State(state): State<Arc<AuthState>>,
    Extension(identity): Extension<AuthIdentity>,
    Json(req): Json<SubscribeRequest>,
) -> Result<Json<serde_json::Value>, ServerError> {
    let user = match identity {
        AuthIdentity::User { user } => user,
        _ => return Err(ServerError::Unauthorized),
    };

    if !tier_exists(&req.tier) {
        return Err(ServerError::BadRequest(format!(
            "unknown tier '{}'. Valid: {:?}",
            req.tier, TIER_NAMES
        )));
    }

    // Enterprise must be granted by an admin, not self-service.
    if req.tier == "enterprise" {
        return Err(ServerError::BadRequest(
            "Enterprise tier requires admin provisioning. Contact support.".into(),
        ));
    }

    let updated = update_user_tier(&state.db, &user.tenant_id, &user.id, &req.tier)
        .map_err(|e| ServerError::Db(e.to_string()))?
        .ok_or_else(|| ServerError::BadRequest("user record vanished mid-update".into()))?;

    let limits = limits_for_tier(&updated.tier);

    Ok(Json(json!({
        "ok": true,
        "tier": updated.tier,
        "cycle": req.cycle,
        "limits": {
            "rpm": limits.rpm,
            "tpm": limits.tpm,
            "key_count_max": limits.key_count_max,
            "daily_messages": limits.daily_messages,
        },
        // Until Stripe is wired, paid tiers don't actually charge anything.
        // Surface that fact in the response so the frontend can render
        // "trial" / "preview" badges.
        "payment_required": false,
        "note": "Stripe integration pending — paid tier applied immediately as a preview. Real billing activates when Stripe is wired up.",
    })))
}

/// GET /v1/billing/me
pub async fn billing_me(
    Extension(identity): Extension<AuthIdentity>,
) -> Result<Json<serde_json::Value>, ServerError> {
    let user = match identity {
        AuthIdentity::User { user } => user,
        _ => return Err(ServerError::Unauthorized),
    };
    let limits = limits_for_tier(&user.tier);
    Ok(Json(json!({
        "user_id": user.id,
        "email": user.email,
        "tier": user.tier,
        "limits": {
            "rpm": limits.rpm,
            "tpm": limits.tpm,
            "key_count_max": limits.key_count_max,
            "daily_messages": limits.daily_messages,
        },
    })))
}

/// POST /v1/billing/cancel — downgrade to free immediately.
pub async fn cancel(
    State(state): State<Arc<AuthState>>,
    Extension(identity): Extension<AuthIdentity>,
) -> Result<Json<serde_json::Value>, ServerError> {
    let user = match identity {
        AuthIdentity::User { user } => user,
        _ => return Err(ServerError::Unauthorized),
    };

    let updated = update_user_tier(&state.db, &user.tenant_id, &user.id, "free")
        .map_err(|e| ServerError::Db(e.to_string()))?
        .ok_or_else(|| ServerError::BadRequest("user record vanished mid-update".into()))?;

    let limits = limits_for_tier(&updated.tier);
    Ok(Json(json!({
        "ok": true,
        "tier": updated.tier,
        "limits": {
            "rpm": limits.rpm,
            "tpm": limits.tpm,
            "key_count_max": limits.key_count_max,
            "daily_messages": limits.daily_messages,
        },
    })))
}
