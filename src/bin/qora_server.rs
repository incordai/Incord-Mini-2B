use std::path::PathBuf;
use anyhow::Context;
use qor4b::server::{config::ServerConfig, http::{router::build_router, serve::{serve_plain, serve_tls}, tls::load_tls_config}};

#[tokio::main(flavor = "multi_thread")]
async fn main() -> anyhow::Result<()> {
    init_tracing();

    let config_path = std::env::args().nth(1).unwrap_or_else(|| "qora-server.toml".to_string());
    let config = ServerConfig::load_from_path(&PathBuf::from(&config_path))
        .with_context(|| format!("loading config from {config_path}"))?;
    tracing::info!(?config_path, "config loaded");

    let router = build_router();

    match (&config.server.tls_cert, &config.server.tls_key) {
        (Some(cert), Some(key)) => {
            let tls = load_tls_config(cert, key).await?;
            serve_tls(router, &config.server.bind, tls).await
        }
        (None, None) => {
            serve_plain(router, &config.server.bind).await
        }
        _ => Err(anyhow::anyhow!("tls_cert and tls_key must be set together or both omitted")),
    }?;
    Ok(())
}

fn init_tracing() {
    use tracing_subscriber::{EnvFilter, fmt};
    let filter = EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info"));
    fmt().with_env_filter(filter).json().init();
}
