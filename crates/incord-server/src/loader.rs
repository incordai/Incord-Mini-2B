//! Build a Registry from ServerConfig at startup.

use anyhow::Context;
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;
use tracing::info;

use crate::config::{ModelEntry, ModelKind, ServerConfig};
use crate::engines::llm::LlmEngine;
use crate::engines::rag::{QorEmbeddingEngine, QorRagBackend, QorRerankEngine};
use crate::engines::stt::IncordSttEngine;
use crate::engines::tts::IncordTtsEngine;
use crate::registry::{Registry, RegistryBuilder};

/// Load every model in the config and build the registry.
/// In `eager` mode (default), each model's weights are read from disk now.
/// In `lazy` mode, weights load on first request (not yet wired in this task).
///
/// # Backend sharing for RAG engines
///
/// Embedding and Rerank entries that resolve to the same model directory share
/// a single `QorRagBackend` (i.e. one copy of the loaded weights in memory).
/// We pre-scan all entries whose kind is Embedding or Rerank, deduplicate by
/// directory path, and load the backend once per unique path.
///
/// TODO(M14): If a directory contains only one of embed.qor-rag or rerank.qor-rag,
/// `QorRag::load` will fail. Ensure config entries point to a directory containing both.
pub fn build_registry(config: &ServerConfig) -> anyhow::Result<Registry> {
    // Pre-load shared qor-rag backends keyed by resolved model directory.
    // This avoids loading the same ~4 GB weights twice when both an Embedding
    // and a Rerank entry point to the same directory.
    let mut shared_rag_backends: HashMap<PathBuf, Arc<QorRagBackend>> = HashMap::new();

    for entry in &config.models {
        if matches!(entry.kind, ModelKind::Embedding | ModelKind::Rerank) {
            let dir = entry
                .path
                .parent()
                .map(|p| p.to_path_buf())
                .unwrap_or_else(|| entry.path.clone());

            if let std::collections::hash_map::Entry::Vacant(e) = shared_rag_backends.entry(dir) {
                let dir_ref = e.key();
                info!(model_id = %entry.id, dir = ?dir_ref, "loading qor-rag backend (embedding + rerank weights)...");
                let backend = QorRagBackend::load(dir_ref).map_err(|e2| {
                    anyhow::anyhow!("qor-rag backend for model `{}`: {e2}", entry.id)
                })?;
                e.insert(Arc::new(backend));
            }
        }
    }

    let mut builder = Registry::builder();
    for entry in &config.models {
        builder = match entry.kind {
            ModelKind::Chat => add_chat(builder, entry)?,
            ModelKind::Tts => add_tts(builder, entry)?,
            ModelKind::Stt => add_stt(builder, entry)?,
            ModelKind::Embedding => add_embed(builder, entry, &shared_rag_backends)?,
            ModelKind::Rerank => add_rerank(builder, entry, &shared_rag_backends)?,
        };
    }

    let registry = builder.build();
    info!(model_count = registry.len(), ids = ?registry.ids(), "registry built");
    Ok(registry)
}

fn add_tts(builder: RegistryBuilder, entry: &ModelEntry) -> anyhow::Result<RegistryBuilder> {
    // The ModelEntry `path` points to the .incord-tts binary file; the model dir
    // (containing config.json, tokenizer.json, and the binary itself) is its parent.
    let model_dir = entry
        .path
        .parent()
        .map(|p| p.to_path_buf())
        .unwrap_or_else(|| entry.path.clone());

    info!(model_id = %entry.id, dir = ?model_dir, "loading TTS model...");

    let description = entry
        .description
        .clone()
        .unwrap_or_else(|| format!("TTS model: {}", entry.id));

    let engine = IncordTtsEngine::load(entry.id.clone(), model_dir, description)?;

    Ok(builder.add_tts(Arc::new(engine)))
}

fn add_stt(builder: RegistryBuilder, entry: &ModelEntry) -> anyhow::Result<RegistryBuilder> {
    // ModelEntry `path` points to the config.json or model file inside the model dir.
    // The model dir (containing config.json, model weights, tokenizer.json) is its parent.
    let model_dir = entry
        .path
        .parent()
        .map(|p| p.to_path_buf())
        .unwrap_or_else(|| entry.path.clone());

    info!(model_id = %entry.id, dir = ?model_dir, "loading STT model...");

    let description = entry
        .description
        .clone()
        .unwrap_or_else(|| format!("STT model: {}", entry.id));

    let engine = IncordSttEngine::load(entry.id.clone(), model_dir, description)?;

    Ok(builder.add_stt(Arc::new(engine)))
}

fn add_embed(
    builder: RegistryBuilder,
    entry: &ModelEntry,
    backends: &HashMap<PathBuf, Arc<QorRagBackend>>,
) -> anyhow::Result<RegistryBuilder> {
    let dir = entry
        .path
        .parent()
        .map(|p| p.to_path_buf())
        .unwrap_or_else(|| entry.path.clone());

    let backend = backends.get(&dir).cloned().ok_or_else(|| {
        anyhow::anyhow!(
            "internal: no qor-rag backend found for dir {}",
            dir.display()
        )
    })?;

    let description = entry
        .description
        .clone()
        .unwrap_or_else(|| format!("Embedding model: {}", entry.id));

    info!(model_id = %entry.id, "registering embedding engine");

    let engine = Arc::new(QorEmbeddingEngine {
        id: entry.id.clone(),
        description,
        backend,
    });

    Ok(builder.add_embed(engine))
}

fn add_rerank(
    builder: RegistryBuilder,
    entry: &ModelEntry,
    backends: &HashMap<PathBuf, Arc<QorRagBackend>>,
) -> anyhow::Result<RegistryBuilder> {
    let dir = entry
        .path
        .parent()
        .map(|p| p.to_path_buf())
        .unwrap_or_else(|| entry.path.clone());

    let backend = backends.get(&dir).cloned().ok_or_else(|| {
        anyhow::anyhow!(
            "internal: no qor-rag backend found for dir {}",
            dir.display()
        )
    })?;

    let description = entry
        .description
        .clone()
        .unwrap_or_else(|| format!("Rerank model: {}", entry.id));

    info!(model_id = %entry.id, "registering rerank engine");

    let engine = Arc::new(QorRerankEngine {
        id: entry.id.clone(),
        description,
        backend,
    });

    Ok(builder.add_rerank(engine))
}

fn add_chat(builder: RegistryBuilder, entry: &ModelEntry) -> anyhow::Result<RegistryBuilder> {
    let tokenizer_path = entry
        .tokenizer
        .as_ref()
        .ok_or_else(|| anyhow::anyhow!("chat model `{}` missing `tokenizer` field", entry.id))?;

    info!(model_id = %entry.id, path = ?entry.path, "loading chat model weights...");

    let weights = incord::save::load_model(&entry.path)
        .with_context(|| format!("loading model from {}", entry.path.display()))?;
    let weights = Arc::new(weights);

    let tokenizer = incord::tokenizer::IncordTokenizer::from_file(tokenizer_path)
        .map_err(|e| anyhow::anyhow!("loading tokenizer from {}: {e}", tokenizer_path.display()))?;
    let tokenizer = Arc::new(tokenizer);

    let description = entry
        .description
        .clone()
        .unwrap_or_else(|| format!("Chat model: {}", entry.id));

    let engine = Arc::new(LlmEngine::new(
        entry.id.clone(),
        weights,
        tokenizer,
        description,
    ));

    Ok(builder.add_chat(engine))
}
