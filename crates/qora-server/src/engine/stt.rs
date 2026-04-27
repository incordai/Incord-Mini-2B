//! Speech-to-Text engines.

use async_trait::async_trait;
use serde::{Deserialize, Serialize};

use crate::engine::ModelEngine;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SttRequest {
    pub model: String,
    pub audio: Vec<u8>,                   // raw audio bytes (WAV/MP3/Opus — engine decodes)
    pub format: Option<String>,           // hint, optional
    pub language: Option<String>,
}

#[derive(Debug, Clone)]
pub struct SttOutput {
    pub text: String,
    pub language_detected: Option<String>,
}

#[async_trait]
pub trait SttEngine: ModelEngine {
    async fn transcribe(&self, req: SttRequest) -> anyhow::Result<SttOutput>;
}
