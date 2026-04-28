//! EmbeddingEngine + RerankEngine wrapping qor-rag's QorRag.
//!
//! Integration approach: path dependency on C:/Projects/AI/qor-main/qor-rs/crates/qor-rag.
//!
//! # API notes (from qor-rag src/lib.rs)
//!
//! - `QorRag::load(dir: impl AsRef<Path>) -> anyhow::Result<Self>`
//!   Loads embed.qor-rag + embed-tokenizer.json + rerank.qor-rag + rerank-tokenizer.json
//!   from a single directory.
//!
//! - `QorRag::embed_passage(text: &str) -> Vec<f32>`   — infallible, returns 2048-dim L2-norm vector
//! - `QorRag::embed_query(text: &str) -> Vec<f32>`     — infallible, applies Qwen3 instruct template
//! - `QorRag::rerank(query: &str, doc: &str) -> f32`   — infallible, returns P(yes|query,doc) in [0,1]
//!   NOTE: takes a SINGLE document string, not a slice. We iterate over documents ourselves.
//!
//! # Backend sharing
//!
//! Both EmbeddingEngine and RerankEngine share the same `Arc<QorRagBackend>` when registered
//! from the same model directory. See `loader.rs` for the shared-backend loading strategy.
//!
//! TODO(M14): The loader currently loads one QorRag instance per entry. When both Embedding
//! and Rerank entries point to the same directory, the backend is loaded twice. A future
//! optimisation would deduplicate by dir path in `build_registry`.

use async_trait::async_trait;
use std::sync::Arc;

use crate::engine::embed::{EmbedOutput, EmbedRequest, EmbeddingEngine};
use crate::engine::rerank::{RerankEngine, RerankOutput, RerankRequest, RerankResult};
use crate::engine::{ModelEngine, Task};

// ---------------------------------------------------------------------------
// Shared backend
// ---------------------------------------------------------------------------

/// Shared backend wrapping a `QorRag` instance that holds both the embedding
/// and reranker model weights. Wrap in `Arc` so multiple engine handles can
/// reference the same in-memory weights without duplication.
pub struct QorRagBackend {
    pub handle: Arc<qor_rag::QorRag>,
}

impl QorRagBackend {
    /// Load both the embedding and reranker models from `dir`.
    ///
    /// Expected layout:
    /// ```text
    /// dir/
    ///   embed.qor-rag          — Qwen3-Embedding-VL-2B weights
    ///   embed-tokenizer.json   — embedding tokenizer
    ///   rerank.qor-rag         — Qwen3-Reranker-VL-2B weights
    ///   rerank-tokenizer.json  — reranker tokenizer
    /// ```
    pub fn load(dir: &std::path::Path) -> anyhow::Result<Self> {
        let handle = qor_rag::QorRag::load(dir)
            .map_err(|e| anyhow::anyhow!("loading qor-rag from {}: {e}", dir.display()))?;
        Ok(Self {
            handle: Arc::new(handle),
        })
    }
}

// ---------------------------------------------------------------------------
// Embedding engine
// ---------------------------------------------------------------------------

/// Embedding engine backed by `QorRag::embed_passage`.
///
/// Uses passage-mode (no instruct template) for all inputs — this is the
/// correct semantic for indexing documents. For query embeddings the client
/// should set `task=embedding` with a different model id that wraps
/// `embed_query`, or use the `embed_query` variant below if desired.
pub struct QorEmbeddingEngine {
    pub id: String,
    pub description: String,
    /// Shared with a sibling `QorRerankEngine` when loaded from the same dir.
    pub backend: Arc<QorRagBackend>,
}

impl ModelEngine for QorEmbeddingEngine {
    fn id(&self) -> &str {
        &self.id
    }

    fn task(&self) -> Task {
        Task::Embedding
    }

    fn description(&self) -> &str {
        &self.description
    }
}

#[async_trait]
impl EmbeddingEngine for QorEmbeddingEngine {
    async fn embed(&self, req: EmbedRequest) -> anyhow::Result<EmbedOutput> {
        let backend = self.backend.clone();

        let result = tokio::task::spawn_blocking(move || -> anyhow::Result<EmbedOutput> {
            let mut embeddings: Vec<Vec<f32>> = Vec::with_capacity(req.input.len());

            // embed_passage is infallible — Vec<f32> directly, no error propagation needed.
            for text in &req.input {
                let v = backend.handle.embed_passage(text);
                embeddings.push(v);
            }

            let dim = embeddings.first().map(|v| v.len()).unwrap_or(0);
            Ok(EmbedOutput { embeddings, dim })
        })
        .await??;

        Ok(result)
    }
}

// ---------------------------------------------------------------------------
// Rerank engine
// ---------------------------------------------------------------------------

/// Rerank engine backed by `QorRag::rerank`.
///
/// `QorRag::rerank(query, doc) -> f32` scores a single (query, doc) pair.
/// We call it once per document and sort descending by score.
pub struct QorRerankEngine {
    pub id: String,
    pub description: String,
    /// Shared with a sibling `QorEmbeddingEngine` when loaded from the same dir.
    pub backend: Arc<QorRagBackend>,
}

impl ModelEngine for QorRerankEngine {
    fn id(&self) -> &str {
        &self.id
    }

    fn task(&self) -> Task {
        Task::Rerank
    }

    fn description(&self) -> &str {
        &self.description
    }
}

#[async_trait]
impl RerankEngine for QorRerankEngine {
    async fn rerank(&self, req: RerankRequest) -> anyhow::Result<RerankOutput> {
        let backend = self.backend.clone();

        let result = tokio::task::spawn_blocking(move || -> anyhow::Result<RerankOutput> {
            // QorRag::rerank(query, doc) -> f32 is a per-pair call (infallible).
            // Score each (query, doc) pair independently.
            let mut results: Vec<RerankResult> = req
                .documents
                .iter()
                .enumerate()
                .map(|(idx, doc)| {
                    let score = backend.handle.rerank(&req.query, doc);
                    RerankResult { index: idx, score }
                })
                .collect();

            // Sort descending by score (higher = more relevant).
            results.sort_by(|a, b| {
                b.score
                    .partial_cmp(&a.score)
                    .unwrap_or(std::cmp::Ordering::Equal)
            });

            if let Some(k) = req.top_k {
                results.truncate(k);
            }

            Ok(RerankOutput { results })
        })
        .await??;

        Ok(result)
    }
}
