//! Client IP + User-Agent extraction (proxy-aware).
//!
//! Priority order for IP:
//!   1. CF-Connecting-IP  (Cloudflare — single, trusted value)
//!   2. X-Forwarded-For   (nginx / AWS ALB — first comma-separated entry)
//!   3. X-Real-IP         (nginx single-value variant)
//!   4. TCP peer address  (ConnectInfo — direct connection, no proxy)

use axum::extract::ConnectInfo;
use axum::http::request::Parts;
use std::net::SocketAddr;

/// Extract client IP from headers (proxy-aware) or fall back to peer address.
pub fn extract_client_ip(parts: &Parts) -> Option<String> {
    // 1. CF-Connecting-IP (Cloudflare sets this to the true client IP)
    if let Some(v) = parts
        .headers
        .get("cf-connecting-ip")
        .and_then(|v| v.to_str().ok())
    {
        let v = v.trim();
        if !v.is_empty() {
            return Some(v.to_string());
        }
    }
    // 2. X-Forwarded-For: "client, proxy1, proxy2" — take the first entry
    if let Some(v) = parts
        .headers
        .get("x-forwarded-for")
        .and_then(|v| v.to_str().ok())
    {
        let first = v.split(',').next().map(|s| s.trim()).unwrap_or("");
        if !first.is_empty() {
            return Some(first.to_string());
        }
    }
    // 3. X-Real-IP: single value
    if let Some(v) = parts
        .headers
        .get("x-real-ip")
        .and_then(|v| v.to_str().ok())
    {
        let v = v.trim();
        if !v.is_empty() {
            return Some(v.to_string());
        }
    }
    // 4. Fallback: TCP peer address (set by into_make_service_with_connect_info)
    parts
        .extensions
        .get::<ConnectInfo<SocketAddr>>()
        .map(|ConnectInfo(addr)| addr.ip().to_string())
}

/// Extract user-agent, truncated to 256 chars to bound storage.
pub fn extract_user_agent(parts: &Parts) -> Option<String> {
    parts
        .headers
        .get("user-agent")
        .and_then(|v| v.to_str().ok())
        .map(|s| s.chars().take(256).collect())
}

/// Extract proxy-injected country code (Cloudflare CF-IPCountry, AWS CloudFront X-Country-Code).
/// Returns None if not present or if the value is "XX" (Cloudflare's placeholder for unknown).
pub fn extract_proxy_country(parts: &Parts) -> Option<String> {
    parts
        .headers
        .get("cf-ipcountry")
        .or_else(|| parts.headers.get("x-country-code"))
        .and_then(|v| v.to_str().ok())
        .map(|s| s.trim())
        .filter(|s| !s.is_empty() && *s != "XX")
        .map(|s| s.to_string())
}
