use std::sync::Arc;
use axum::{extract::{Request, State}, http::header::AUTHORIZATION, middleware::Next, response::Response};
use redb::Database;
use crate::server::auth::keys::{lookup_by_raw, ApiKey};
use crate::server::auth::ratelimit::RateLimiter;
use crate::server::error::ServerError;

#[derive(Clone)]
pub struct AuthState {
    pub db: Arc<Database>,
    pub server_secret: Arc<Vec<u8>>,
    pub limiter: RateLimiter,
}

pub async fn auth_middleware(
    State(state): State<Arc<AuthState>>,
    mut req: Request,
    next: Next,
) -> Result<Response, ServerError> {
    let raw_key = req
        .headers()
        .get(AUTHORIZATION)
        .and_then(|v| v.to_str().ok())
        .and_then(|s| s.strip_prefix("Bearer "))
        .map(|s| s.to_string())
        .ok_or(ServerError::Unauthorized)?;

    let key = lookup_by_raw(&state.db, &state.server_secret, &raw_key)
        .map_err(|e| ServerError::Db(e.to_string()))?
        .ok_or(ServerError::Unauthorized)?;

    if !key.is_active {
        return Err(ServerError::Unauthorized);
    }

    if let Err(retry_after_secs) = state.limiter.check_request(&key.id, key.rpm_limit) {
        return Err(ServerError::RateLimited { retry_after_secs });
    }

    req.extensions_mut().insert(key);
    Ok(next.run(req).await)
}

pub fn current_key(req: &Request) -> Option<&ApiKey> {
    req.extensions().get::<ApiKey>()
}
