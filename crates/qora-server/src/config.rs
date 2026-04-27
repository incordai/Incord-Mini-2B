//! Multi-model server config (TOML).

use std::path::{Path, PathBuf};
use serde::Deserialize;

#[derive(Debug, Clone, Deserialize)]
pub struct ServerConfig {
    pub server: HttpConfig,
    pub auth: AuthConfig,
    pub limits: LimitsConfig,
    pub observability: ObservabilityConfig,
    pub scheduler: SchedulerConfig,
    /// Each model registered with the server.
    #[serde(default)]
    pub models: Vec<ModelEntry>,
    #[serde(default)]
    pub loading: LoadingConfig,
}

#[derive(Debug, Clone, Deserialize)]
pub struct HttpConfig {
    pub bind: String,
    pub tls_cert: Option<PathBuf>,
    pub tls_key: Option<PathBuf>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct AuthConfig {
    pub admin_token_env: String,
    pub db_path: PathBuf,
}

#[derive(Debug, Clone, Deserialize)]
pub struct LimitsConfig {
    pub default: PerKeyLimits,
}

#[derive(Debug, Clone, Deserialize)]
pub struct PerKeyLimits {
    pub rpm: u32,
    pub tpm: u32,
}

#[derive(Debug, Clone, Deserialize)]
pub struct ObservabilityConfig {
    pub prometheus_bind: String,
    pub log_level: String,
    pub log_format: String,
}

#[derive(Debug, Clone, Deserialize)]
pub struct SchedulerConfig {
    pub slot_count: usize,
    pub max_queue: usize,
    pub max_context: usize,
    pub request_timeout_secs: u64,
}

/// One model in the registry.
#[derive(Debug, Clone, Deserialize)]
pub struct ModelEntry {
    pub id: String,
    pub kind: ModelKind,
    pub path: PathBuf,
    pub tokenizer: Option<PathBuf>,
    pub description: Option<String>,
    /// Optional per-engine extras (TTS voices_dir, etc.). Loosely typed so
    /// each engine kind can read what it needs without the schema knowing.
    #[serde(default)]
    pub extras: toml::Table,
}

#[derive(Debug, Clone, Copy, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum ModelKind {
    Chat,
    Tts,
    Stt,
    Embedding,
    Rerank,
}

#[derive(Debug, Clone, Deserialize)]
pub struct LoadingConfig {
    /// "eager" loads all models at startup; "lazy" loads on first request.
    #[serde(default = "default_loading_mode")]
    pub mode: String,
}

impl Default for LoadingConfig {
    fn default() -> Self {
        Self { mode: default_loading_mode() }
    }
}

fn default_loading_mode() -> String { "eager".to_string() }

impl ServerConfig {
    pub fn load_from_path(path: &Path) -> anyhow::Result<Self> {
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
    fn loads_config_with_multiple_models() {
        let mut tmp = NamedTempFile::new().unwrap();
        writeln!(tmp, r#"
[server]
bind = "127.0.0.1:8080"

[scheduler]
slot_count = 64
max_queue = 1000
max_context = 8192
request_timeout_secs = 90

[auth]
admin_token_env = "QORA_ADMIN_TOKEN"
db_path = "/var/lib/qora/auth.redb"

[limits.default]
rpm = 60
tpm = 60000

[observability]
prometheus_bind = "127.0.0.1:9090"
log_level = "info"
log_format = "json"

[[models]]
id = "qora-llm"
kind = "chat"
path = "/var/lib/qora/llm/model.qor4b"
tokenizer = "/var/lib/qora/llm/tokenizer.json"
description = "Default LLM"

[[models]]
id = "qora-tts"
kind = "tts"
path = "/var/lib/qora/tts/model.qora-tts"
tokenizer = "/var/lib/qora/tts/tokenizer.json"
description = "TTS engine"

[[models]]
id = "qora-stt"
kind = "stt"
path = "/var/lib/qora/stt/model.qora-stt-small"

[[models]]
id = "qora-embed"
kind = "embedding"
path = "/var/lib/qora/rag/embed.bin"

[[models]]
id = "qora-rerank"
kind = "rerank"
path = "/var/lib/qora/rag/rerank.bin"
"#).unwrap();

        let cfg = ServerConfig::load_from_path(tmp.path()).unwrap();
        assert_eq!(cfg.models.len(), 5);
        assert_eq!(cfg.models[0].id, "qora-llm");
        assert_eq!(cfg.models[0].kind, ModelKind::Chat);
        assert_eq!(cfg.models[1].kind, ModelKind::Tts);
        assert_eq!(cfg.models[2].kind, ModelKind::Stt);
        assert_eq!(cfg.models[3].kind, ModelKind::Embedding);
        assert_eq!(cfg.models[4].kind, ModelKind::Rerank);
    }

    #[test]
    fn loads_with_no_models() {
        let mut tmp = NamedTempFile::new().unwrap();
        writeln!(tmp, r#"
[server]
bind = "127.0.0.1:8080"

[scheduler]
slot_count = 64
max_queue = 1000
max_context = 8192
request_timeout_secs = 90

[auth]
admin_token_env = "QORA_ADMIN_TOKEN"
db_path = "/var/lib/qora/auth.redb"

[limits.default]
rpm = 60
tpm = 60000

[observability]
prometheus_bind = "127.0.0.1:9090"
log_level = "info"
log_format = "json"
"#).unwrap();
        let cfg = ServerConfig::load_from_path(tmp.path()).unwrap();
        assert!(cfg.models.is_empty());
    }

    #[test]
    fn loading_mode_defaults_to_eager() {
        let mut tmp = NamedTempFile::new().unwrap();
        writeln!(tmp, r#"
[server]
bind = "127.0.0.1:8080"

[scheduler]
slot_count = 64
max_queue = 1000
max_context = 8192
request_timeout_secs = 90

[auth]
admin_token_env = "QORA_ADMIN_TOKEN"
db_path = "/var/lib/qora/auth.redb"

[limits.default]
rpm = 60
tpm = 60000

[observability]
prometheus_bind = "127.0.0.1:9090"
log_level = "info"
log_format = "json"
"#).unwrap();
        let cfg = ServerConfig::load_from_path(tmp.path()).unwrap();
        assert_eq!(cfg.loading.mode, "eager");
    }
}
