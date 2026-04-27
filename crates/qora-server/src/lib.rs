//! qora-server: HTTP API + scheduler + model engine traits.
//!
//! This crate provides the multi-model API surface. Each model type
//! (LLM, TTS, STT, embedding, rerank) implements a trait below.
//! The registry holds Arc<dyn ChatEngine> etc. and dispatches
//! requests by task and model id.

pub mod engine;
pub mod engines;
