use anyhow::Context;
use std::path::PathBuf;
use std::sync::Arc;

use incord_server::auth::admin::AdminState;
use incord_server::auth::middleware::AuthState;
use incord_server::auth::ratelimit::RateLimiter;
use incord_server::config::ServerConfig;
use incord_server::db;
use incord_server::http::handlers::inference::InferenceState;
use incord_server::http::router::build_router;
use incord_server::loader::build_registry;

#[tokio::main(flavor = "multi_thread")]
async fn main() -> anyhow::Result<()> {
    init_tracing();

    let config_path = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "incord-server.toml".to_string());
    let config = ServerConfig::load_from_path(&PathBuf::from(&config_path))
        .with_context(|| format!("loading config from {config_path}"))?;
    tracing::info!(?config_path, "config loaded");

    // Auth DB
    let db = db::open(&config.auth.db_path)?;

    // Server secret (for HMAC key index)
    let server_secret = std::env::var("INCORD_SERVER_SECRET")
        .context("INCORD_SERVER_SECRET env var must be set")?
        .into_bytes();

    // Admin token
    let admin_token = std::env::var(&config.auth.admin_token_env).with_context(|| {
        format!(
            "admin token env var {} not set",
            config.auth.admin_token_env
        )
    })?;

    // GeoIP database (optional — graceful if absent)
    let geo = incord_server::geoip::GeoDb::try_load(config.geoip.db_path.as_deref());

    let auth = Arc::new(AuthState {
        db,
        server_secret: Arc::new(server_secret),
        limiter: RateLimiter::new(),
        tiers: config.tiers.clone(),
        geo,
    });
    let admin_state = Arc::new(AdminState { admin_token });

    // Build the model registry — eagerly loads all models from config
    tracing::info!("building model registry...");
    let registry = build_registry(&config)?;
    tracing::info!(model_count = registry.len(), "registry ready");

    let inference_state = Arc::new(InferenceState {
        registry,
        vision_semaphore: Arc::new(tokio::sync::Semaphore::new(1)), // serial — vision encode is ~90 s
        tts_semaphore: Arc::new(tokio::sync::Semaphore::new(2)), // 2 concurrent (bandwidth-heavy)
        stt_semaphore: Arc::new(tokio::sync::Semaphore::new(2)),
        embed_semaphore: Arc::new(tokio::sync::Semaphore::new(8)), // lighter, allow more
        rerank_semaphore: Arc::new(tokio::sync::Semaphore::new(4)),
        // Chat history shares the same redb file as auth keys (separate logical tables).
        history_db: auth.db.clone(),
        history_config: config.history.clone(),
    });
    let router = build_router(inference_state, auth, admin_state);

    // Serve via TLS or plain HTTP based on config
    match (&config.server.tls_cert, &config.server.tls_key) {
        (Some(cert), Some(key)) => {
            let tls = incord_server::http::tls::load_tls_config(cert, key).await?;
            incord_server::http::server::serve_tls(router, &config.server.bind, tls).await
        }
        (None, None) => incord_server::http::server::serve_plain(router, &config.server.bind).await,
        _ => Err(anyhow::anyhow!(
            "tls_cert and tls_key must be set together or both omitted"
        )),
    }
}

fn init_tracing() {
    use tracing_subscriber::{EnvFilter, fmt};
    let filter = EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info"));
    fmt().with_env_filter(filter).json().init();
}
