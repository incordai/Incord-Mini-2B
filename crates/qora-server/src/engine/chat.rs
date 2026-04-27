//! Chat / completion engines (LLMs).

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use tokio::sync::mpsc;

use crate::engine::ModelEngine;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatRequest {
    pub model: String,
    pub messages: Vec<ChatMessage>,
    pub max_tokens: Option<u32>,
    pub temperature: Option<f32>,
    pub top_p: Option<f32>,
    pub stream: bool,
    /// QORA-specific extensions
    pub think: Option<bool>,
    pub think_budget: Option<u32>,
    /// Optional pre-encoded vision embeddings (for chat-with-image flow).
    /// (embeds, num_vision_tokens). Set by the HTTP layer after
    /// running the vision encoder; the engine consumes them.
    #[serde(skip)]
    pub vision_embeds: Option<(Vec<f32>, usize)>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatMessage {
    pub role: String,
    pub content: String,
}

#[derive(Debug, Clone)]
pub enum ChatEvent {
    Token(u32, String),
    Finish(FinishReason),
    Error(String),
}

#[derive(Debug, Clone, Copy)]
pub enum FinishReason {
    Stop,
    Length,
    Cancelled,
    Timeout,
    Error,
}

impl FinishReason {
    pub fn as_openai(&self) -> &'static str {
        match self {
            Self::Stop | Self::Cancelled | Self::Timeout | Self::Error => "stop",
            Self::Length => "length",
        }
    }
}

#[async_trait]
pub trait ChatEngine: ModelEngine {
    /// Run a chat completion. The engine internally manages prefill, decode,
    /// cache, and sampling. Tokens stream out via sse_tx; the function returns
    /// when generation is complete (or errors).
    async fn chat(&self, req: ChatRequest, sse_tx: mpsc::Sender<ChatEvent>) -> anyhow::Result<()>;
}
