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
    /// Incord-specific extensions
    pub think: Option<bool>,
    pub think_budget: Option<u32>,
    /// If true, include the model's <think>...</think> reasoning in the
    /// streamed output. Default: false (hide thinking, show only the final answer).
    pub show_think: Option<bool>,
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
    /// A visible-output token (the assistant's actual reply).
    Token(u32, String),
    /// A reasoning/think-block token. Streamed on a separate channel so the
    /// UI can render thinking in a collapsible section instead of mixing it
    /// into the main reply, and so TTS skips it entirely.
    ThinkingToken(u32, String),
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

    /// Whether this engine supports vision input.
    fn supports_vision(&self) -> bool {
        false
    }

    /// Encode raw image bytes to vision embeddings, returning (embeds, num_tokens).
    /// Runs on spawn_blocking internally. Returns an error if the engine does not
    /// support vision.
    async fn encode_image(&self, _bytes: Vec<u8>) -> anyhow::Result<(Vec<f32>, usize)> {
        anyhow::bail!("this engine does not support vision input")
    }
}
