use std::path::Path;
use anyhow::Context;
use axum_server::tls_rustls::RustlsConfig;

pub async fn load_tls_config(cert_path: &Path, key_path: &Path) -> anyhow::Result<RustlsConfig> {
    let config = RustlsConfig::from_pem_file(cert_path, key_path)
        .await
        .with_context(|| format!("loading TLS cert from {} and key from {}",
                                 cert_path.display(), key_path.display()))?;
    Ok(config)
}
