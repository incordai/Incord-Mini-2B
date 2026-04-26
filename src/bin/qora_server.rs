use anyhow::Context;
use qor4b::server::{
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
        .unwrap_or_else(|| "qora-server.toml".to_string());
    let config = ServerConfig::load_from_path(&PathBuf::from(&config_path))
        .with_context(|| format!("loading config from {config_path}"))?;
    tracing::info!(?config_path, "config loaded");

    let db = qor4b::server::db::open(&config.auth.db_path)?;
    let admin_token = std::env::var(&config.auth.admin_token_env).with_context(|| {
        format!(
            "admin token env var {} not set",
            config.auth.admin_token_env
        )
    })?;
    let server_secret = std::env::var("QORA_SERVER_SECRET")
        .context("QORA_SERVER_SECRET env var must be set (used for HMAC key index)")?
        .into_bytes();

    let auth = Arc::new(qor4b::server::auth::middleware::AuthState {
        db: db.clone(),
        server_secret: Arc::new(server_secret),
        limiter: qor4b::server::auth::ratelimit::RateLimiter::new(),
    });
    let admin_state = Arc::new(qor4b::server::auth::admin::AdminState { admin_token });

    let weights = Arc::new(
        qor4b::save::load_model(&config.model.path)
            .with_context(|| format!("loading model from {}", config.model.path.display()))?,
    );
    let tokenizer = Arc::new(
        qor4b::tokenizer::QoraTokenizer::from_file(&config.model.tokenizer)
            .map_err(|e| anyhow::anyhow!("{}", e))
            .with_context(|| {
                format!(
                    "loading tokenizer from {}",
                    config.model.tokenizer.display()
                )
            })?,
    );

    let queue = qor4b::server::scheduler::queue::Queue::new(config.scheduler.max_queue);

    // Spawn N worker threads. Conservative default (kernel work is internally rayon-parallelized,
    // so too many workers contend for memory bandwidth). Cap at min(cpus, 8) for v1.
    let num_workers = std::thread::available_parallelism()
        .map(|p| p.get().min(8))
        .unwrap_or(2);
    for id in 0..num_workers {
        qor4b::server::scheduler::worker::Worker {
            id,
            weights: weights.clone(),
            tokenizer: tokenizer.clone(),
            queue: queue.clone(),
        }
        .spawn();
    }
    tracing::info!(num_workers, "spawned worker threads");

    let usage = qor4b::server::usage::tracker::UsageTracker::new(db.clone());

    let chat_state = Arc::new(qor4b::server::http::handlers::chat::ChatState {
        queue,
        weights,
        tokenizer,
        usage,
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
