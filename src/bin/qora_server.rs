use std::path::PathBuf;
use anyhow::Context;
use qor4b::server::{config::ServerConfig, http::{router::build_router, serve::serve_plain}};

#[tokio::main(flavor = "multi_thread")]
async fn main() -> anyhow::Result<()> {
    init_tracing();

    let config_path = std::env::args().nth(1).unwrap_or_else(|| "qora-server.toml".to_string());
    let config = ServerConfig::load_from_path(&PathBuf::from(&config_path))
        .with_context(|| format!("loading config from {config_path}"))?;
    tracing::info!(?config_path, "config loaded");

    let router = build_router();

    serve_plain(router, &config.server.bind).await?;
    Ok(())
}

fn init_tracing() {
    use tracing_subscriber::{EnvFilter, fmt};
    let filter = EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info"));
    fmt().with_env_filter(filter).json().init();
}
