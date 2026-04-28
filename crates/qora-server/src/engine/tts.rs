//! Text-to-Speech engines.

use async_trait::async_trait;
use serde::{Deserialize, Serialize};

use crate::engine::ModelEngine;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TtsRequest {
    pub model: String,
    pub input: String,
    pub voice: Option<String>,  // built-in name OR base64 reference WAV
    pub format: Option<String>, // "wav" (default), "mp3"
    pub language: Option<String>,
}

#[derive(Debug, Clone)]
pub struct TtsOutput {
    pub audio: Vec<u8>, // encoded audio bytes
    pub format: String, // "wav", "mp3"
    pub sample_rate: u32,
}

#[async_trait]
pub trait TtsEngine: ModelEngine {
    async fn synthesize(&self, req: TtsRequest) -> anyhow::Result<TtsOutput>;
}
