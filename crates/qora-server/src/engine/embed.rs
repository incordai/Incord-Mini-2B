//! Embedding engines.

use async_trait::async_trait;
use serde::{Deserialize, Serialize};

use crate::engine::ModelEngine;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmbedRequest {
    pub model: String,
    pub input: Vec<String>, // can be one or many
}

#[derive(Debug, Clone)]
pub struct EmbedOutput {
    pub embeddings: Vec<Vec<f32>>, // one per input
    pub dim: usize,
}

#[async_trait]
pub trait EmbeddingEngine: ModelEngine {
    async fn embed(&self, req: EmbedRequest) -> anyhow::Result<EmbedOutput>;
}
