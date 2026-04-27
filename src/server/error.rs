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

#[cfg(test)]
mod tests {
    use super::*;
    use axum::body::to_bytes;
    use axum::http::StatusCode;

    #[tokio::test]
    async fn queue_full_returns_503_with_retry_after() {
        let err = ServerError::QueueFull {
            retry_after_secs: 7,
        };
        let resp = err.into_response();
        assert_eq!(resp.status(), StatusCode::SERVICE_UNAVAILABLE);
        assert_eq!(resp.headers().get("retry-after").unwrap(), "7");
    }

    #[tokio::test]
    async fn rate_limited_returns_429_with_retry_after() {
        let err = ServerError::RateLimited {
            retry_after_secs: 30,
        };
        let resp = err.into_response();
        assert_eq!(resp.status(), StatusCode::TOO_MANY_REQUESTS);
        assert_eq!(resp.headers().get("retry-after").unwrap(), "30");
    }

    #[tokio::test]
    async fn unauthorized_returns_401_no_retry_after() {
        let err = ServerError::Unauthorized;
        let resp = err.into_response();
        assert_eq!(resp.status(), StatusCode::UNAUTHORIZED);
        assert!(resp.headers().get("retry-after").is_none());
    }

    #[tokio::test]
    async fn bad_request_returns_400() {
        let err = ServerError::BadRequest("invalid input".into());
        let resp = err.into_response();
        assert_eq!(resp.status(), StatusCode::BAD_REQUEST);

        // Verify the error body contains the message
        let body = to_bytes(resp.into_body(), usize::MAX).await.unwrap();
        let body: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(body["error"]["type"], "invalid_request_error");
        assert_eq!(body["error"]["message"], "invalid input");
    }
}
