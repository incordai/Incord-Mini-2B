use crate::auth::keys::ApiKey;
use crate::auth::ratelimit::RateLimiter;
use crate::auth::users::User;
use crate::error::ServerError;
use axum::{
    extract::{Request, State},
    http::header::AUTHORIZATION,
    middleware::Next,
    response::Response,
};
use redb::Database;
use std::sync::Arc;

#[derive(Clone)]
pub struct AuthState {
    pub db: Arc<Database>,
    pub server_secret: Arc<Vec<u8>>,
    pub limiter: RateLimiter,
    pub tiers: crate::config::TierConfig,
}

/// The resolved identity of a request — inserted as an Axum extension.
#[derive(Debug, Clone)]
pub enum AuthIdentity {
    /// API key holder (programmatic / legacy).
    ApiKey { key: ApiKey },
    /// Logged-in user (free or paid web session).
    User { user: User },
    /// Anonymous guest with per-ID quota.
    Guest {
        guest_id: String,
        msg_count: u32,
        max_messages: u32,
    },
}

pub async fn auth_middleware(
    State(state): State<Arc<AuthState>>,
    mut req: Request,
    next: Next,
) -> Result<Response, ServerError> {
    let auth_header = req
        .headers()
        .get(AUTHORIZATION)
        .and_then(|v| v.to_str().ok())
        .map(|s| s.to_string());
    let guest_header = req
        .headers()
        .get("x-guest-id")
        .and_then(|v| v.to_str().ok())
        .map(|s| s.to_string());

    let identity = if let Some(ref h) = auth_header {
        let token = h
            .strip_prefix("Bearer ")
            .ok_or(ServerError::Unauthorized)?;
        if token.starts_with(crate::auth::keys::KEY_PREFIX) {
            // API key path (existing behavior)
            let key =
                crate::auth::keys::lookup_by_raw(&state.db, &state.server_secret, token)
                    .map_err(|e| ServerError::Db(e.to_string()))?
                    .ok_or(ServerError::Unauthorized)?;
            if !key.is_active {
                return Err(ServerError::Unauthorized);
            }
            if let Err(retry) =
                state.limiter.check_request(&format!("k:{}", key.id), key.rpm_limit)
            {
                return Err(ServerError::RateLimited {
                    retry_after_secs: retry,
                });
            }
            AuthIdentity::ApiKey { key }
        } else if token.starts_with("sess_") {
            // Session token path
            let session =
                crate::auth::sessions::lookup_session(&state.db, token)
                    .map_err(|e| ServerError::Db(e.to_string()))?
                    .ok_or(ServerError::Unauthorized)?;
            let user = crate::auth::users::get_user(&state.db, &session.user_id)
                .map_err(|e| ServerError::Db(e.to_string()))?
                .ok_or(ServerError::Unauthorized)?;
            let tier_limits = match user.tier.as_str() {
                "paid" => &state.tiers.paid,
                _ => &state.tiers.free,
            };
            if let Err(retry) =
                state.limiter.check_request(&format!("u:{}", user.id), tier_limits.rpm)
            {
                return Err(ServerError::RateLimited {
                    retry_after_secs: retry,
                });
            }
            AuthIdentity::User { user }
        } else {
            return Err(ServerError::Unauthorized);
        }
    } else if let Some(guest_id) = guest_header {
        // Guest path — check + increment quota
        let max = state.tiers.anonymous.max_messages_per_guest;
        match crate::auth::guest::check_and_increment(&state.db, &guest_id, max)
            .map_err(|e| ServerError::Db(e.to_string()))?
        {
            Ok(q) => AuthIdentity::Guest {
                guest_id,
                msg_count: q.msg_count,
                max_messages: max,
            },
            Err(_q) => {
                return Err(ServerError::PaymentRequired {
                    message: format!(
                        "guest message limit reached ({max} messages). Sign up for free to continue."
                    ),
                })
            }
        }
    } else {
        return Err(ServerError::Unauthorized);
    };

    req.extensions_mut().insert(identity);
    Ok(next.run(req).await)
}

/// Convenience: extract the API key from extensions (for handlers that only care about API key path).
pub fn current_key(req: &Request) -> Option<&ApiKey> {
    req.extensions().get::<ApiKey>()
}
