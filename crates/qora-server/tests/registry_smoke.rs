//! Smoke test: registry can be built from a TOML config with multiple model kinds.
//! TTS/STT/Embedding/Rerank entries are accepted but produce warnings (no concrete impl yet).

#![cfg(feature = "mock-weights")]

use qora_server::config::{ModelKind, ServerConfig};
use qora_server::registry::Registry;

#[test]
fn empty_registry_is_buildable() {
    let r = Registry::builder().build();
    assert!(r.is_empty());
    assert_eq!(r.len(), 0);
    assert!(r.ids().is_empty());
}

#[test]
fn config_supports_5_model_kinds() {
    // Verify the TOML schema accepts all 5 kinds (chat/tts/stt/embedding/rerank)
    // without requiring implementations to exist yet.
    use std::io::Write;
    let mut tmp = tempfile::NamedTempFile::new().unwrap();
    writeln!(
        tmp,
        r#"
[server]
bind = "127.0.0.1:8080"

[scheduler]
slot_count = 64
max_queue = 1000
max_context = 8192
request_timeout_secs = 90

[auth]
admin_token_env = "QORA_ADMIN_TOKEN"
db_path = "/tmp/auth.redb"

[limits.default]
rpm = 60
tpm = 60000

[observability]
prometheus_bind = "127.0.0.1:9090"
log_level = "info"
log_format = "json"

[[models]]
id = "any-llm"
kind = "chat"
path = "/x"
tokenizer = "/y"

[[models]]
id = "any-tts"
kind = "tts"
path = "/x"

[[models]]
id = "any-stt"
kind = "stt"
path = "/x"

[[models]]
id = "any-embed"
kind = "embedding"
path = "/x"

[[models]]
id = "any-rerank"
kind = "rerank"
path = "/x"
"#
    )
    .unwrap();

    let cfg = ServerConfig::load_from_path(tmp.path()).unwrap();
    assert_eq!(cfg.models.len(), 5);
    assert_eq!(
        cfg.models
            .iter()
            .filter(|m| m.kind == ModelKind::Chat)
            .count(),
        1
    );
    assert_eq!(
        cfg.models
            .iter()
            .filter(|m| m.kind == ModelKind::Tts)
            .count(),
        1
    );
    assert_eq!(
        cfg.models
            .iter()
            .filter(|m| m.kind == ModelKind::Stt)
            .count(),
        1
    );
    assert_eq!(
        cfg.models
            .iter()
            .filter(|m| m.kind == ModelKind::Embedding)
            .count(),
        1
    );
    assert_eq!(
        cfg.models
            .iter()
            .filter(|m| m.kind == ModelKind::Rerank)
            .count(),
        1
    );
}

#[tokio::test]
async fn registry_with_mock_chat_engine_dispatches_correctly() {
    use qora_server::engines::llm::LlmEngine;
    use std::sync::Arc;

    // Build a mock chat engine using qor4b::gemv::mock::tiny_weights
    let weights = Arc::new(qor4b::gemv::mock::tiny_weights());
    let tokenizer_path = std::path::PathBuf::from(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../qora-llm/model/tokenizer.json"
    ));
    let tokenizer = if tokenizer_path.exists() {
        Arc::new(qor4b::tokenizer::QoraTokenizer::from_file(&tokenizer_path).unwrap())
    } else {
        // Skip this test if no tokenizer fixture is available
        eprintln!(
            "skipping: tokenizer fixture not found at {}",
            tokenizer_path.display()
        );
        return;
    };

    // Use a non-hardcoded id (could be anything)
    let engine = Arc::new(LlmEngine::new(
        "test-llm-runtime-id",
        weights,
        tokenizer,
        "Mock chat engine for tests",
    ));

    let registry = Registry::builder().add_chat(engine).build();

    assert!(registry.contains("test-llm-runtime-id"));
    assert!(!registry.contains("does-not-exist"));
    assert!(registry.get_chat("test-llm-runtime-id").is_some());
    assert!(registry.get_chat("does-not-exist").is_none());
    // The same id is NOT a tts/stt/embed/rerank engine
    assert!(registry.get_tts("test-llm-runtime-id").is_none());
    assert_eq!(registry.len(), 1);
}
