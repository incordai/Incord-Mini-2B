use crate::error::ServerError;
use axum::{
    extract::{Request, State},
    http::header::AUTHORIZATION,
    middleware::Next,
    response::Response,
};
use std::sync::Arc;

#[derive(Clone)]
pub struct AdminState {
    pub admin_token: String,
}

pub async fn admin_guard(
    State(state): State<Arc<AdminState>>,
    req: Request,
    next: Next,
) -> Result<Response, ServerError> {
    let token = req
        .headers()
        .get(AUTHORIZATION)
        .and_then(|v| v.to_str().ok())
        .and_then(|s| s.strip_prefix("Bearer "))
        .ok_or(ServerError::Unauthorized)?;
    if !constant_time_eq(token.as_bytes(), state.admin_token.as_bytes()) {
        return Err(ServerError::Unauthorized);
    }
    Ok(next.run(req).await)
}

fn constant_time_eq(a: &[u8], b: &[u8]) -> bool {
    if a.len() != b.len() {
        return false;
    }
    let mut diff = 0u8;
    for (x, y) in a.iter().zip(b.iter()) {
        diff |= x ^ y;
    }
    diff == 0
}
