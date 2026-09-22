//! Rerank engines.

use async_trait::async_trait;
use serde::{Deserialize, Serialize};

use crate::engine::ModelEngine;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RerankRequest {
    pub model: String,
    pub query: String,
    pub documents: Vec<String>,
    pub top_k: Option<usize>,
}

#[derive(Debug, Clone, Serialize)]
pub struct RerankOutput {
    /// Sorted descending by score. Indices refer back to the input documents.
    pub results: Vec<RerankResult>,
}

#[derive(Debug, Clone, Serialize)]
pub struct RerankResult {
    pub index: usize,
    pub score: f32,
}

#[async_trait]
pub trait RerankEngine: ModelEngine {
    async fn rerank(&self, req: RerankRequest) -> anyhow::Result<RerankOutput>;
}
