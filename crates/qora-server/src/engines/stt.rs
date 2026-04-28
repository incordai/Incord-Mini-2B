//! Concrete SttEngine wrapping qora-stt-small.
//!
//! Integration approach: lib linkage (path dep).
//! We added `[lib]` to D:/Burn-AI/QORA-STT-Small/Cargo.toml.
//!
//! Loading order (mirrors main.rs in qora-stt-small):
//!   1. config.json  → WhisperConfig::from_file
//!   2. weights      → loader::load_weights (safetensors) or save::load_model (binary)
//!   3. tokenizer.json → WhisperTokenizer::from_file
//!
//! Audio: the qora-stt-small transcribe() API takes a Path to a WAV file.
//! For v1 we write the incoming bytes to a temp file and pass that path;
//! the temp file is cleaned up after transcription.

use std::path::PathBuf;
use std::sync::Arc;
use anyhow::Context;
use async_trait::async_trait;

use crate::engine::stt::{SttEngine, SttOutput, SttRequest};
use crate::engine::{ModelEngine, Task};

// ---------------------------------------------------------------------------
// Model handle
// ---------------------------------------------------------------------------

/// All loaded STT model state. All fields are plain heap data (Vec<f32>)
/// plus the HuggingFace tokenizer (which is Arc-backed and Send+Sync),
/// so Send + Sync hold without unsafe.
pub struct SttModelHandle {
    pub weights: qora_stt_small::weights::WhisperWeights,
    pub config: qora_stt_small::config::WhisperConfig,
    pub tokenizer: qora_stt_small::tokenizer::WhisperTokenizer,
}

// Safety: weights are all Vec<f32> (plain heap), config is all primitives/Vecs,
// tokenizer wraps an Arc-backed HuggingFace tokenizer.
unsafe impl Send for SttModelHandle {}
unsafe impl Sync for SttModelHandle {}

// ---------------------------------------------------------------------------
// Engine struct
// ---------------------------------------------------------------------------

/// Wrapper around the qora-stt-small crate's transcription API.
///
/// On creation (`load`), reads config + weights + tokenizer from disk.
/// On `transcribe`, runs inference on a Tokio blocking thread (STT is heavy CPU).
pub struct QoraSttEngine {
    pub id: String,
    pub description: String,
    pub model: Arc<SttModelHandle>,
}

impl QoraSttEngine {
    /// Load the STT model from `model_dir`.
    ///
    /// Expected layout (mirrors QORA-STT-Small binary invocation):
    /// ```text
    /// model_dir/
    ///   config.json          ← WhisperConfig
    ///   model.safetensors    ← weights (safetensors, preferred)
    ///   model.qora-stt-small ← binary weights (optional, used if safetensors absent)
    ///   tokenizer.json       ← HuggingFace tokenizer
    /// ```
    pub fn load(
        id: impl Into<String>,
        model_dir: PathBuf,
        description: impl Into<String>,
    ) -> anyhow::Result<Self> {
        let id = id.into();
        let description = description.into();

        // 1. Load config
        let config_path = model_dir.join("config.json");
        tracing::info!(path = ?config_path, "loading STT config...");
        let config = qora_stt_small::config::WhisperConfig::from_file(&config_path)
            .map_err(|e| anyhow::anyhow!("loading STT config from {}: {e}", config_path.display()))?;

        // 2. Load weights — prefer safetensors; fall back to binary .qora-stt-small
        let binary_path = model_dir.join("model.qora-stt-small");
        let st_path = model_dir.join("model.safetensors");

        let weights = if binary_path.exists() {
            tracing::info!(path = ?binary_path, "loading STT binary weights...");
            qora_stt_small::save::load_model(&binary_path)
                .map_err(|e| anyhow::anyhow!("load STT binary model: {e}"))?
        } else if st_path.exists() {
            tracing::info!(path = ?st_path, "loading STT safetensors weights...");
            qora_stt_small::loader::load_weights(&model_dir, &config)
                .map_err(|e| anyhow::anyhow!("load STT safetensors: {e}"))?
        } else {
            anyhow::bail!(
                "STT model weights not found in {} (expected model.qora-stt-small or model.safetensors)",
                model_dir.display()
            );
        };

        // 3. Load tokenizer
        let tokenizer_path = model_dir.join("tokenizer.json");
        tracing::info!(path = ?tokenizer_path, "loading STT tokenizer...");
        let tokenizer = qora_stt_small::tokenizer::WhisperTokenizer::from_file(&tokenizer_path)
            .map_err(|e| anyhow::anyhow!("loading STT tokenizer from {}: {e}", tokenizer_path.display()))?;

        tracing::info!(
            model_id = %id,
            d_model = config.d_model,
            vocab_size = config.vocab_size,
            "STT model loaded"
        );

        Ok(Self {
            id,
            description,
            model: Arc::new(SttModelHandle { weights, config, tokenizer }),
        })
    }
}

// ---------------------------------------------------------------------------
// Trait impls
// ---------------------------------------------------------------------------

impl ModelEngine for QoraSttEngine {
    fn id(&self) -> &str { &self.id }
    fn task(&self) -> Task { Task::Stt }
    fn description(&self) -> &str { &self.description }
}

#[async_trait]
impl SttEngine for QoraSttEngine {
    async fn transcribe(&self, req: SttRequest) -> anyhow::Result<SttOutput> {
        let model = self.model.clone();
        let language = req.language.unwrap_or_else(|| "english".to_string());

        // Write incoming audio bytes to a temp WAV file — qora-stt-small's
        // transcribe() API takes a &Path, so we materialise the bytes to disk.
        let temp_path = tokio::task::spawn_blocking(move || -> anyhow::Result<PathBuf> {
            let path = std::env::temp_dir()
                .join(format!("qora-stt-{}.wav", uuid::Uuid::new_v4()));
            std::fs::write(&path, &req.audio)
                .context("writing temp audio file for STT")?;
            Ok(path)
        })
        .await??;

        let temp_path_for_inf = temp_path.clone();

        let text = tokio::task::spawn_blocking(move || -> anyhow::Result<String> {
            let result = qora_stt_small::inference::transcribe(
                &model.weights,
                &model.config,
                &model.tokenizer,
                &temp_path_for_inf,
                &language,
            )
            .map_err(|e| anyhow::anyhow!("transcribe: {e}"))?;
            Ok(result)
        })
        .await??;

        // Best-effort cleanup of temp file
        let _ = std::fs::remove_file(&temp_path);

        Ok(SttOutput {
            text,
            language_detected: None,
        })
    }
}
