//! Engine trait hierarchy for multi-model serving.

use serde::{Deserialize, Serialize};
use std::sync::Arc;

pub mod chat;
pub mod embed; // stub for now
pub mod rerank;
pub mod stt; // stub for now
pub mod tts; // stub for now // stub for now

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Task {
    Chat,
    Tts,
    Stt,
    Embedding,
    Rerank,
}

/// Base capabilities of any model engine.
pub trait ModelEngine: Send + Sync + 'static {
    /// Identifier matching the request's `model` field (e.g. "qora-4b").
    fn id(&self) -> &str;

    /// Task this engine performs.
    fn task(&self) -> Task;

    /// Human-readable description (shown in /v1/models).
    fn description(&self) -> &str;
}

/// Generic engine handle — used in the registry. Every concrete engine
/// implements both `ModelEngine` and one task-specific trait below.
pub type AnyEngine = Arc<dyn ModelEngine>;
