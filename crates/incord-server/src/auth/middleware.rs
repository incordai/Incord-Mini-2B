use crate::auth::keys::ApiKey;
use crate::auth::ratelimit::RateLimiter;
use crate::auth::users::User;
use crate::db::{is_safe_tenant, DEFAULT_TENANT};
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
    /// Offline GeoIP database. GeoDb::empty() when no .mmdb file is configured.
    pub geo: crate::geoip::GeoDb,
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
        tenant_id: String,
        guest_id: String,
        msg_count: u32,
        max_messages: u32,
    },
}

impl AuthIdentity {
    /// Tenant this caller belongs to. Every persistence call goes through
    /// this — never hardcode a tenant string in handlers.
    pub fn tenant_id(&self) -> &str {
        match self {
            AuthIdentity::ApiKey { key } => &key.tenant_id,
            AuthIdentity::User { user } => &user.tenant_id,
            AuthIdentity::Guest { tenant_id, .. } => tenant_id,
        }
    }
}

/// IP, User-Agent, and geo data extracted from the incoming request.
#[derive(Debug, Clone, Default)]
pub struct ClientInfo {
    pub ip: Option<String>,
    pub user_agent: Option<String>,
    pub country: Option<String>,
    pub country_name: Option<String>,
    pub city: Option<String>,
    pub region: Option<String>,
    pub timezone: Option<String>,
    pub latitude: Option<f64>,
    pub longitude: Option<f64>,
}

/// Read the request's tenant from the `X-Tenant` header. Falls back to
/// [`DEFAULT_TENANT`] for single-tenant deployments. Invalid tenant IDs
/// (not matching `[A-Za-z0-9_-]{1,64}`) fall back too — we don't 401 on
/// the header alone since the auth layer is the gate.
fn extract_tenant(req: &Request) -> String {
    req.headers()
        .get("x-tenant")
        .and_then(|v| v.to_str().ok())
        .filter(|s| is_safe_tenant(s))
        .map(|s| s.to_string())
        .unwrap_or_else(|| DEFAULT_TENANT.to_string())
}

pub async fn auth_middleware(
    State(state): State<Arc<AuthState>>,
    req: Request,
    next: Next,
) -> Result<Response, ServerError> {
    let (parts, body) = req.into_parts();
    let ip = crate::http::clientinfo::extract_client_ip(&parts);
    let user_agent = crate::http::clientinfo::extract_user_agent(&parts);
    let cdn_country = crate::http::clientinfo::extract_proxy_country(&parts);

    let user_country = crate::http::clientinfo::extract_user_country(&parts);
    let user_city = crate::http::clientinfo::extract_user_city(&parts);
    let user_region = crate::http::clientinfo::extract_user_region(&parts);
    let user_timezone = crate::http::clientinfo::extract_user_timezone(&parts);

    let geo = ip
        .as_deref()
        .and_then(|addr| state.geo.lookup(addr))
        .unwrap_or_default();
    let client = ClientInfo {
        ip,
        user_agent,
        country: user_country.or(cdn_country).or(geo.country),
        country_name: geo.country_name,
        city: user_city.or(geo.city),
        region: user_region.or(geo.region),
        timezone: user_timezone,
        latitude: geo.latitude,
        longitude: geo.longitude,
    };
    let mut req = Request::from_parts(parts, body);
    req.extensions_mut().insert(client.clone());

    let tenant = extract_tenant(&req);

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
            let key =
                crate::auth::keys::lookup_by_raw(&state.db, &state.server_secret, &tenant, token)
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
            let session = crate::auth::sessions::lookup_session(&state.db, &tenant, token)
                .map_err(|e| ServerError::Db(e.to_string()))?
                .ok_or(ServerError::Unauthorized)?;
            let user = crate::auth::users::get_user(&state.db, &tenant, &session.user_id)
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
        let max = state.tiers.anonymous.max_messages_per_guest;
        match crate::auth::guest::check_and_increment(
            &state.db,
            &tenant,
            &guest_id,
            max,
            client.ip.as_deref(),
            client.user_agent.as_deref(),
            client.country.as_deref(),
            client.city.as_deref(),
            client.timezone.as_deref(),
        )
        .map_err(|e| ServerError::Db(e.to_string()))?
        {
            Ok(q) => AuthIdentity::Guest {
                tenant_id: tenant.clone(),
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
