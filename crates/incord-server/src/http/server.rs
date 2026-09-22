use anyhow::Context;
use axum::Router;
use axum_server::tls_rustls::RustlsConfig;
use std::net::SocketAddr;

pub async fn serve_plain(router: Router, bind: &str) -> anyhow::Result<()> {
    let addr: SocketAddr = bind.parse().context("parsing bind address")?;
    let listener = tokio::net::TcpListener::bind(addr).await?;
    tracing::info!(%addr, "qora-server listening (plain HTTP)");
    axum::serve(
        listener,
        router.into_make_service_with_connect_info::<SocketAddr>(),
    )
    .await?;
    Ok(())
}

pub async fn serve_tls(router: Router, bind: &str, tls: RustlsConfig) -> anyhow::Result<()> {
    let addr: SocketAddr = bind.parse().context("parsing bind address")?;
    tracing::info!(%addr, "qora-server listening (TLS)");
    axum_server::bind_rustls(addr, tls)
        .serve(router.into_make_service_with_connect_info::<SocketAddr>())
        .await?;
    Ok(())
}
