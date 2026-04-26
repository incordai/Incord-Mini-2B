use std::net::SocketAddr;
use anyhow::Context;
use axum::Router;

pub async fn serve_plain(router: Router, bind: &str) -> anyhow::Result<()> {
    let addr: SocketAddr = bind.parse().context("parsing bind address")?;
    let listener = tokio::net::TcpListener::bind(addr).await?;
    tracing::info!(%addr, "qora-server listening (plain HTTP)");
    axum::serve(listener, router.into_make_service()).await?;
    Ok(())
}
