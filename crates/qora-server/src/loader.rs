//! Build a Registry from ServerConfig at startup.

use std::sync::Arc;
use anyhow::Context;
use tracing::info;

use crate::config::{ModelEntry, ModelKind, ServerConfig};
use crate::engines::llm::LlmEngine;
use crate::engines::tts::QoraTtsEngine;
use crate::registry::{Registry, RegistryBuilder};

/// Load every model in the config and build the registry.
/// In `eager` mode (default), each model's weights are read from disk now.
/// In `lazy` mode, weights load on first request (not yet wired in this task).
pub fn build_registry(config: &ServerConfig) -> anyhow::Result<Registry> {
    let mut builder = Registry::builder();
    for entry in &config.models {
        builder = match entry.kind {
            ModelKind::Chat => add_chat(builder, entry)?,
            ModelKind::Tts => add_tts(builder, entry)?,
            ModelKind::Stt => {
                tracing::warn!(model_id = %entry.id, "STT engine wrapper not yet implemented; skipping");
                builder
            }
            ModelKind::Embedding => {
                tracing::warn!(model_id = %entry.id, "Embedding engine wrapper not yet implemented; skipping");
                builder
            }
            ModelKind::Rerank => {
                tracing::warn!(model_id = %entry.id, "Rerank engine wrapper not yet implemented; skipping");
                builder
            }
        };
    }

    let registry = builder.build();
    info!(model_count = registry.len(), ids = ?registry.ids(), "registry built");
    Ok(registry)
}

fn add_tts(builder: RegistryBuilder, entry: &ModelEntry) -> anyhow::Result<RegistryBuilder> {
    // The ModelEntry `path` points to the .qora-tts binary file; the model dir
    // (containing config.json, tokenizer.json, and the binary itself) is its parent.
    let model_dir = entry.path.parent()
        .map(|p| p.to_path_buf())
        .unwrap_or_else(|| entry.path.clone());

    info!(model_id = %entry.id, dir = ?model_dir, "loading TTS model...");

    let description = entry.description.clone()
        .unwrap_or_else(|| format!("TTS model: {}", entry.id));

    let engine = QoraTtsEngine::load(entry.id.clone(), model_dir, description)?;

    Ok(builder.add_tts(Arc::new(engine)))
}

fn add_chat(builder: RegistryBuilder, entry: &ModelEntry) -> anyhow::Result<RegistryBuilder> {
    let tokenizer_path = entry.tokenizer.as_ref()
        .ok_or_else(|| anyhow::anyhow!("chat model `{}` missing `tokenizer` field", entry.id))?;

    info!(model_id = %entry.id, path = ?entry.path, "loading chat model weights...");

    let weights = qor4b::save::load_model(&entry.path)
        .with_context(|| format!("loading model from {}", entry.path.display()))?;
    let weights = Arc::new(weights);

    let tokenizer = qor4b::tokenizer::QoraTokenizer::from_file(tokenizer_path)
        .map_err(|e| anyhow::anyhow!("loading tokenizer from {}: {e}", tokenizer_path.display()))?;
    let tokenizer = Arc::new(tokenizer);

    let description = entry.description.clone()
        .unwrap_or_else(|| format!("Chat model: {}", entry.id));

    let engine = Arc::new(LlmEngine::new(
        entry.id.clone(),
        weights,
        tokenizer,
        description,
    ));

    Ok(builder.add_chat(engine))
}
