use axum::{
    extract::Request,
    http::{HeaderValue, header::HeaderName},
    middleware::Next,
    response::Response,
};
use uuid::Uuid;

pub static REQUEST_ID_HEADER: HeaderName = HeaderName::from_static("x-request-id");

pub async fn request_id_middleware(mut req: Request, next: Next) -> Response {
    let request_id = req
        .headers()
        .get(&REQUEST_ID_HEADER)
        .and_then(|v| v.to_str().ok())
        .map(|s| s.to_string())
        .unwrap_or_else(|| Uuid::new_v4().to_string());

    let span = tracing::info_span!("request",
        request_id = %request_id,
        method = %req.method(),
        path = req.uri().path(),
    );
    let _enter = span.enter();
    tracing::info!("request received");

    req.extensions_mut().insert(RequestId(request_id.clone()));

    let mut response = next.run(req).await;

    if let Ok(value) = HeaderValue::from_str(&request_id) {
        response
            .headers_mut()
            .insert(REQUEST_ID_HEADER.clone(), value);
    }

    tracing::info!(status = %response.status(), "request complete");
    response
}

#[derive(Debug, Clone)]
pub struct RequestId(pub String);
