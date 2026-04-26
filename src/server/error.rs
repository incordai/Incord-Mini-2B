use axum::Json;
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use serde_json::json;
use thiserror::Error;
use uuid::Uuid;

#[derive(Debug, Error)]
pub enum ServerError {
    #[error("invalid request: {0}")]
    BadRequest(String),
    #[error("authentication failed")]
    Unauthorized,
    #[error("rate limit exceeded")]
    RateLimited { retry_after_secs: u64 },
    #[error("server queue full")]
    QueueFull { retry_after_secs: u64 },
    #[error("request timed out")]
    Timeout,
    #[error("inference error: {0}")]
    Inference(String),
    #[error("database error: {0}")]
    Db(String),
    #[error("internal error: {0}")]
    Internal(String),
}

impl IntoResponse for ServerError {
    fn into_response(self) -> Response {
        let request_id = Uuid::new_v4();
        let (status, error_type, message) = match &self {
            Self::BadRequest(m) => (StatusCode::BAD_REQUEST, "invalid_request_error", m.clone()),
            Self::Unauthorized => (
                StatusCode::UNAUTHORIZED,
                "authentication_error",
                self.to_string(),
            ),
            Self::RateLimited { .. } => (
                StatusCode::TOO_MANY_REQUESTS,
                "rate_limit_error",
                self.to_string(),
            ),
            Self::QueueFull { .. } => (
                StatusCode::SERVICE_UNAVAILABLE,
                "server_error",
                self.to_string(),
            ),
            Self::Timeout => (
                StatusCode::REQUEST_TIMEOUT,
                "timeout_error",
                self.to_string(),
            ),
            Self::Inference(m) => (
                StatusCode::INTERNAL_SERVER_ERROR,
                "inference_error",
                m.clone(),
            ),
            Self::Db(m) => (
                StatusCode::INTERNAL_SERVER_ERROR,
                "database_error",
                m.clone(),
            ),
            Self::Internal(m) => (
                StatusCode::INTERNAL_SERVER_ERROR,
                "internal_error",
                m.clone(),
            ),
        };

        let body = Json(json!({
            "error": {
                "message": message,
                "type": error_type,
                "request_id": request_id,
            }
        }));

        let mut response = (status, body).into_response();

        if let Self::RateLimited { retry_after_secs } | Self::QueueFull { retry_after_secs } = &self
        {
            response
                .headers_mut()
                .insert("retry-after", retry_after_secs.to_string().parse().unwrap());
        }
        response
    }
}

pub type ServerResult<T> = Result<T, ServerError>;
