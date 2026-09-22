use anyhow::Context;
use incord::server::{
    config::ServerConfig,
    http::{
        router::build_router,
        serve::{serve_plain, serve_tls},
        tls::load_tls_config,
    },
};
use std::path::PathBuf;
use std::sync::Arc;

#[tokio::main(flavor = "multi_thread")]
async fn main() -> anyhow::Result<()> {
    init_tracing();

    let config_path = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "incord-server.toml".to_string());
    let config = ServerConfig::load_from_path(&PathBuf::from(&config_path))
        .with_context(|| format!("loading config from {config_path}"))?;
    tracing::info!(?config_path, "config loaded");

    let db = incord::server::db::open(&config.auth.db_path)?;
    let admin_token = std::env::var(&config.auth.admin_token_env).with_context(|| {
        format!(
            "admin token env var {} not set",
            config.auth.admin_token_env
        )
    })?;
    let server_secret = std::env::var("INCORD_SERVER_SECRET")
        .or_else(|_| std::env::var("QORA_SERVER_SECRET"))
        .context("INCORD_SERVER_SECRET (or QORA_SERVER_SECRET) env var must be set (used for HMAC key index)")?
        .into_bytes();

    let auth = Arc::new(incord::server::auth::middleware::AuthState {
        db: db.clone(),
        server_secret: Arc::new(server_secret),
        limiter: incord::server::auth::ratelimit::RateLimiter::new(),
    });
    let admin_state = Arc::new(incord::server::auth::admin::AdminState { admin_token });

    let weights = Arc::new(
        incord::save::load_model(&config.model.path)
            .with_context(|| format!("loading model from {}", config.model.path.display()))?,
    );
    let tokenizer = Arc::new(
        incord::tokenizer::QoraTokenizer::from_file(&config.model.tokenizer)
            .map_err(|e| anyhow::anyhow!("{}", e))
            .with_context(|| {
                format!(
                    "loading tokenizer from {}",
                    config.model.tokenizer.display()
                )
            })?,
    );

    let queue = incord::server::scheduler::queue::Queue::new(config.scheduler.max_queue);

    // Spawn the slot pool
    let pool = incord::server::scheduler::pool::SlotPool::new(config.scheduler.slot_count);

    // Build the batcher and run it on a dedicated OS thread with its own multi_thread runtime
    // (same pattern as M3 Task 6 — Worker had the same tokio deadlock issue, Batcher avoids
    // the issue by NOT calling block_on inside spawn_blocking; it's a pure synchronous loop.)
    let batcher = incord::server::scheduler::batcher::Batcher {
        pool: pool.clone(),
        queue: queue.clone(),
        weights: weights.clone(),
        tokenizer: tokenizer.clone(),
    };
    std::thread::Builder::new()
        .name("incord-batcher".into())
        .spawn(move || batcher.run())
        .expect("failed to spawn batcher thread");
    tracing::info!(slot_count = config.scheduler.slot_count, "batcher started");

    let usage = incord::server::usage::tracker::UsageTracker::new(db.clone());

    // One permit: serialise concurrent vision encodes to cap peak memory usage.
    let vision_semaphore = Arc::new(tokio::sync::Semaphore::new(1));

    let chat_state = Arc::new(incord::server::http::handlers::chat::ChatState {
        queue,
        weights,
        tokenizer,
        usage,
        vision_semaphore,
    });

    let router = build_router(auth, admin_state, chat_state);

    match (&config.server.tls_cert, &config.server.tls_key) {
        (Some(cert), Some(key)) => {
            let tls = load_tls_config(cert, key).await?;
            serve_tls(router, &config.server.bind, tls).await
        }
        (None, None) => serve_plain(router, &config.server.bind).await,
        _ => Err(anyhow::anyhow!(
            "tls_cert and tls_key must be set together or both omitted"
        )),
    }?;
    Ok(())
}

fn init_tracing() {
    use tracing_subscriber::{EnvFilter, fmt};
    let filter = EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info"));
    fmt().with_env_filter(filter).json().init();
}
