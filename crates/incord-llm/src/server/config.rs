use serde::Deserialize;
use std::path::PathBuf;

#[derive(Debug, Deserialize, Clone)]
pub struct ServerConfig {
    pub server: HttpConfig,
    pub model: ModelConfig,
    pub scheduler: SchedulerConfig,
    pub auth: AuthConfig,
    pub limits: LimitsConfig,
    pub observability: ObservabilityConfig,
}

#[derive(Debug, Deserialize, Clone)]
pub struct HttpConfig {
    pub bind: String,
    pub tls_cert: Option<PathBuf>,
    pub tls_key: Option<PathBuf>,
}

#[derive(Debug, Deserialize, Clone)]
pub struct ModelConfig {
    pub path: PathBuf,
    pub tokenizer: PathBuf,
}

#[derive(Debug, Deserialize, Clone)]
pub struct SchedulerConfig {
    pub slot_count: usize,
    pub max_queue: usize,
    pub max_context: usize,
    pub request_timeout_secs: u64,
}

#[derive(Debug, Deserialize, Clone)]
pub struct AuthConfig {
    pub admin_token_env: String,
    pub db_path: PathBuf,
}

#[derive(Debug, Deserialize, Clone)]
pub struct LimitsConfig {
    pub default: PerKeyLimits,
}

#[derive(Debug, Deserialize, Clone)]
pub struct PerKeyLimits {
    pub rpm: u32,
    pub tpm: u32,
}

#[derive(Debug, Deserialize, Clone)]
pub struct ObservabilityConfig {
    pub prometheus_bind: String,
    pub log_level: String,
    pub log_format: String,
}

impl ServerConfig {
    pub fn load_from_path(path: &std::path::Path) -> anyhow::Result<Self> {
        let contents = std::fs::read_to_string(path)?;
        let config: Self = toml::from_str(&contents)?;
        Ok(config)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::NamedTempFile;

    #[test]
    fn loads_minimal_valid_config() {
        let mut tmp = NamedTempFile::new().unwrap();
        writeln!(
            tmp,
            r#"
[server]
bind = "127.0.0.1:8080"

[model]
path = "/opt/incord/model.incord"
tokenizer = "/opt/incord/tokenizer.json"

[scheduler]
slot_count = 64
max_queue = 1000
max_context = 8192
request_timeout_secs = 90

[auth]
admin_token_env = "INCORD_ADMIN_TOKEN"
db_path = "/opt/incord/auth.db"

[limits.default]
rpm = 60
tpm = 60000

[observability]
prometheus_bind = "127.0.0.1:9090"
log_level = "info"
log_format = "json"
"#
        )
        .unwrap();
        let cfg = ServerConfig::load_from_path(tmp.path()).unwrap();
        assert_eq!(cfg.server.bind, "127.0.0.1:8080");
        assert_eq!(cfg.scheduler.slot_count, 64);
        assert_eq!(cfg.limits.default.rpm, 60);
    }

    #[test]
    fn missing_required_field_errors() {
        let mut tmp = NamedTempFile::new().unwrap();
        writeln!(
            tmp,
            r#"
[server]
bind = "127.0.0.1:8080"
"#
        )
        .unwrap();
        assert!(ServerConfig::load_from_path(tmp.path()).is_err());
    }
}
