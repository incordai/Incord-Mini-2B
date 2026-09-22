//! The unified /v1/inference request type. Dispatches by `task` field.

use crate::engine::Task;
use crate::engine::chat::ChatMessage;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct InferenceRequest {
    pub task: Task,
    pub model: String,

    // Chat-specific
    #[serde(default)]
    pub messages: Vec<ChatMessage>,

    // Common text input (for tts/embedding)
    #[serde(default)]
    pub input: Option<serde_json::Value>, // can be string or array of strings

    // TTS-specific
    #[serde(default)]
    pub voice: Option<String>,
    #[serde(default)]
    pub format: Option<String>,
    #[serde(default)]
    pub language: Option<String>,
    /// Natural-language style/emotion instruction (TTS only).
    /// e.g. "happy", "sad", "speak slowly". Only honoured by 1.7B+ customvoice.
    #[serde(default)]
    pub instruct: Option<String>,

    // STT-specific
    #[serde(default)]
    pub audio: Option<String>, // base64-encoded
    #[serde(default)]
    pub audio_format: Option<String>,

    // Rerank-specific
    #[serde(default)]
    pub query: Option<String>,
    #[serde(default)]
    pub documents: Vec<String>,
    #[serde(default)]
    pub top_k: Option<usize>,

    // Sampling/generation
    #[serde(default)]
    pub max_tokens: Option<u32>,
    #[serde(default)]
    pub temperature: Option<f32>,
    #[serde(default)]
    pub top_p: Option<f32>,
    #[serde(default)]
    pub stream: bool,

    // QORA extensions
    #[serde(default)]
    pub think: Option<bool>,
    #[serde(default)]
    pub think_budget: Option<u32>,
    #[serde(default)]
    pub reasoning_effort: Option<crate::api::openai::request::ReasoningEffort>,
    /// If true, include the model's <think>...</think> reasoning in output.
    /// Default: false (hide thinking, return only the final answer).
    #[serde(default)]
    pub show_think: Option<bool>,

    /// If true, strip markdown formatting from the assistant's reply before
    /// returning. Honoured for non-streaming chat only.
    #[serde(default)]
    pub strip_markdown: Option<bool>,
}
