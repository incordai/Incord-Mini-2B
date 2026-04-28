//! Unified /v1/inference + OpenAI-compatible aliases.

use axum::response::IntoResponse;
use axum::{
    Json,
    extract::{Multipart, State},
};
use base64::{Engine as _, engine::general_purpose::STANDARD as B64};
use chrono::Utc;
use futures::stream::{self, StreamExt};
use std::sync::Arc;
use tokio::sync::mpsc;
use uuid::Uuid;

use crate::api::inference_request::InferenceRequest;
use crate::api::openai::request::{ChatCompletionRequest, ContentPart, MessageContent};
use crate::api::openai::response::{
    ChatCompletionChunk, ChatCompletionResponse, Choice, ChunkChoice, ChunkDelta, ResponseMessage,
    Usage,
};
use crate::engine::Task;
use crate::engine::chat::{ChatEvent, ChatMessage, ChatRequest, FinishReason};
use crate::engine::embed::EmbedRequest as EngineEmbedRequest;
use crate::engine::rerank::RerankRequest as EngineRerankRequest;
use crate::engine::stt::SttRequest as EngineSttRequest;
use crate::engine::tts::TtsRequest as EngineTtsRequest;
use crate::error::ServerError;
use crate::http::sse::into_sse_response;
use crate::registry::Registry;

#[derive(Clone)]
pub struct InferenceState {
    pub registry: Registry,
    /// Semaphore that gates vision encoding (memory-heavy, ~90 s on CPU).
    /// 1 permit = serial to avoid OOM on the EPYC.
    pub vision_semaphore: Arc<tokio::sync::Semaphore>,
    /// Semaphore for TTS synthesis.
    pub tts_semaphore: Arc<tokio::sync::Semaphore>,
    /// Semaphore for STT transcription.
    pub stt_semaphore: Arc<tokio::sync::Semaphore>,
    /// Semaphore for embedding (lighter — allow more parallel).
    pub embed_semaphore: Arc<tokio::sync::Semaphore>,
    /// Semaphore for rerank.
    pub rerank_semaphore: Arc<tokio::sync::Semaphore>,
}

impl InferenceState {
    /// Build with default-permissive semaphores for tests.
    pub fn for_test(registry: Registry) -> Self {
        Self {
            registry,
            vision_semaphore: Arc::new(tokio::sync::Semaphore::new(1)),
            tts_semaphore: Arc::new(tokio::sync::Semaphore::new(2)),
            stt_semaphore: Arc::new(tokio::sync::Semaphore::new(2)),
            embed_semaphore: Arc::new(tokio::sync::Semaphore::new(8)),
            rerank_semaphore: Arc::new(tokio::sync::Semaphore::new(4)),
        }
    }
}

/// Unified endpoint: dispatches by `task` field.
pub async fn unified_inference(
    State(state): State<Arc<InferenceState>>,
    Json(req): Json<InferenceRequest>,
) -> Result<axum::response::Response, ServerError> {
    if !state.registry.contains(&req.model) {
        return Err(ServerError::BadRequest(format!(
            "unknown model: {}",
            req.model
        )));
    }
    match req.task {
        Task::Chat => handle_chat(state, req).await,
        Task::Tts => handle_tts(state, req).await,
        Task::Stt => handle_stt(state, req).await,
        Task::Embedding => handle_embed(state, req).await,
        Task::Rerank => handle_rerank(state, req).await,
    }
}

/// OpenAI alias: /v1/chat/completions accepts ChatCompletionRequest, translates to InferenceRequest.
/// Supports image_url content parts — images are encoded via the engine's vision encoder.
pub async fn openai_chat_completions(
    State(state): State<Arc<InferenceState>>,
    Json(req): Json<ChatCompletionRequest>,
) -> Result<axum::response::Response, ServerError> {
    if !state.registry.contains(&req.model) {
        return Err(ServerError::BadRequest(format!(
            "unknown model: {}",
            req.model
        )));
    }
    let engine = state.registry.get_chat(&req.model).ok_or_else(|| {
        ServerError::BadRequest(format!("model `{}` is not a chat model", req.model))
    })?;

    // Parse messages; extract image bytes from any image_url content parts.
    let mut images: Vec<Vec<u8>> = Vec::new();
    let mut messages: Vec<ChatMessage> = Vec::new();
    for m in req.messages {
        let mut text = String::new();
        match m.content {
            MessageContent::Text(s) => text = s,
            MessageContent::Parts(parts) => {
                for p in parts {
                    match p {
                        ContentPart::Text { text: t } => {
                            text.push_str(&t);
                            text.push('\n');
                        }
                        ContentPart::ImageUrl { image_url } => {
                            let bytes = parse_image_url(&image_url.url).await?;
                            images.push(bytes);
                            // Insert vision placeholder tokens so the LLM knows where image embeds go.
                            text.push_str("<|vision_start|><|image_pad|><|vision_end|>");
                        }
                    }
                }
            }
        }
        messages.push(ChatMessage {
            role: m.role,
            content: text,
        });
    }

    // If the request contained images, encode the first one via the vision encoder.
    // (Multi-image support is future work — only the first image is used in v1.)
    let vision_embeds = if !images.is_empty() {
        if !engine.supports_vision() {
            return Err(ServerError::BadRequest(format!(
                "model `{}` does not support vision input",
                req.model
            )));
        }
        // Acquire the vision semaphore before encoding to avoid OOM.
        let _permit = state
            .vision_semaphore
            .clone()
            .acquire_owned()
            .await
            .map_err(|_| ServerError::Internal("vision semaphore closed".into()))?;
        let first = images.into_iter().next().unwrap();
        let embeds = engine
            .encode_image(first)
            .await
            .map_err(|e| ServerError::Inference(format!("vision encode: {e}")))?;
        Some(embeds)
    } else {
        None
    };

    let inference_req = InferenceRequest {
        task: Task::Chat,
        model: req.model,
        messages,
        input: None,
        voice: None,
        format: None,
        language: None,
        audio: None,
        audio_format: None,
        query: None,
        documents: vec![],
        top_k: None,
        max_tokens: req.max_tokens,
        temperature: req.temperature,
        top_p: req.top_p,
        stream: req.stream,
        think: req.think,
        think_budget: req.think_budget,
    };

    handle_chat_with_vision(state, inference_req, vision_embeds).await
}

/// Parse a data: or http(s):// image URL into raw bytes.
async fn parse_image_url(url: &str) -> Result<Vec<u8>, ServerError> {
    if let Some(rest) = url.strip_prefix("data:") {
        let (header, data) = rest
            .split_once(',')
            .ok_or_else(|| ServerError::BadRequest("invalid data url: missing comma".into()))?;
        if header.contains("base64") {
            B64.decode(data)
                .map_err(|e| ServerError::BadRequest(format!("base64 decode: {e}")))
        } else {
            Ok(data.as_bytes().to_vec())
        }
    } else if url.starts_with("http://") || url.starts_with("https://") {
        let r = reqwest::get(url)
            .await
            .map_err(|e| ServerError::BadRequest(format!("fetching image: {e}")))?;
        if !r.status().is_success() {
            return Err(ServerError::BadRequest(format!(
                "image fetch {} -> {}",
                url,
                r.status()
            )));
        }
        Ok(r.bytes()
            .await
            .map_err(|e| ServerError::BadRequest(e.to_string()))?
            .to_vec())
    } else {
        Err(ServerError::BadRequest(format!(
            "unsupported image_url scheme: {}",
            &url[..url.len().min(40)]
        )))
    }
}

// ---------------------------------------------------------------------------
// TTS handler + OpenAI /v1/audio/speech alias
// ---------------------------------------------------------------------------

async fn handle_tts(
    state: Arc<InferenceState>,
    req: InferenceRequest,
) -> Result<axum::response::Response, ServerError> {
    let _permit = state
        .tts_semaphore
        .clone()
        .acquire_owned()
        .await
        .map_err(|_| ServerError::Internal("tts semaphore closed".into()))?;

    let engine = state.registry.get_tts(&req.model).ok_or_else(|| {
        ServerError::BadRequest(format!("model `{}` is not a tts model", req.model))
    })?;

    // Extract input text from the generic `input` field.
    let input_text = match req.input {
        Some(serde_json::Value::String(s)) => s,
        Some(other) => {
            return Err(ServerError::BadRequest(format!(
                "tts input must be a string, got: {other}"
            )));
        }
        None => {
            return Err(ServerError::BadRequest(
                "tts request missing 'input' field".into(),
            ));
        }
    };

    let tts_req = EngineTtsRequest {
        model: req.model.clone(),
        input: input_text,
        voice: req.voice,
        format: req.format,
        language: req.language,
    };

    let output = engine
        .synthesize(tts_req)
        .await
        .map_err(|e| ServerError::Inference(format!("tts synthesis: {e}")))?;

    let content_type = match output.format.as_str() {
        "wav" => "audio/wav",
        "mp3" => "audio/mpeg",
        _ => "application/octet-stream",
    };

    Ok((
        [(axum::http::header::CONTENT_TYPE, content_type)],
        output.audio,
    )
        .into_response())
}

/// OpenAI-compatible request body for `/v1/audio/speech`.
#[derive(Debug, Clone, serde::Deserialize)]
pub struct OpenAiSpeechRequest {
    pub model: String,
    pub input: String,
    pub voice: Option<String>,
    /// "wav" or "mp3" (only "wav" supported in v1)
    pub response_format: Option<String>,
    pub language: Option<String>,
}

/// OpenAI-compatible alias: POST /v1/audio/speech
pub async fn openai_audio_speech(
    State(state): State<Arc<InferenceState>>,
    Json(req): Json<OpenAiSpeechRequest>,
) -> Result<axum::response::Response, ServerError> {
    let inference_req = InferenceRequest {
        task: Task::Tts,
        model: req.model,
        messages: vec![],
        input: Some(serde_json::Value::String(req.input)),
        voice: req.voice,
        format: req.response_format,
        language: req.language,
        audio: None,
        audio_format: None,
        query: None,
        documents: vec![],
        top_k: None,
        max_tokens: None,
        temperature: None,
        top_p: None,
        stream: false,
        think: None,
        think_budget: None,
    };

    if !state.registry.contains(&inference_req.model) {
        return Err(ServerError::BadRequest(format!(
            "unknown model: {}",
            inference_req.model
        )));
    }
    handle_tts(state, inference_req).await
}

// ---------------------------------------------------------------------------
// STT handler + OpenAI /v1/audio/transcriptions alias
// ---------------------------------------------------------------------------

async fn handle_stt(
    state: Arc<InferenceState>,
    req: InferenceRequest,
) -> Result<axum::response::Response, ServerError> {
    let _permit = state
        .stt_semaphore
        .clone()
        .acquire_owned()
        .await
        .map_err(|_| ServerError::Internal("stt semaphore closed".into()))?;

    let engine = state.registry.get_stt(&req.model).ok_or_else(|| {
        ServerError::BadRequest(format!("model `{}` is not an stt model", req.model))
    })?;

    // Decode base64 audio bytes
    let audio_b64 = req.audio.ok_or_else(|| {
        ServerError::BadRequest("stt request missing 'audio' field (base64-encoded WAV)".into())
    })?;
    let audio = B64
        .decode(&audio_b64)
        .map_err(|e| ServerError::BadRequest(format!("invalid base64 audio: {e}")))?;

    let stt_req = EngineSttRequest {
        model: req.model.clone(),
        audio,
        format: req.audio_format,
        language: req.language,
    };

    let output = engine
        .transcribe(stt_req)
        .await
        .map_err(|e| ServerError::Inference(format!("stt transcription: {e}")))?;

    Ok(Json(serde_json::json!({
        "text": output.text,
        "language": output.language_detected,
    }))
    .into_response())
}

/// OpenAI-compatible alias: POST /v1/audio/transcriptions (multipart upload)
pub async fn openai_audio_transcriptions(
    State(state): State<Arc<InferenceState>>,
    mut multipart: Multipart,
) -> Result<axum::response::Response, ServerError> {
    let mut model: Option<String> = None;
    let mut audio_bytes: Option<Vec<u8>> = None;
    let mut language: Option<String> = None;

    while let Some(field) = multipart
        .next_field()
        .await
        .map_err(|e| ServerError::BadRequest(format!("multipart error: {e}")))?
    {
        let name: String = field.name().unwrap_or("").to_string();
        match name.as_str() {
            "model" => {
                model = Some(
                    field
                        .text()
                        .await
                        .map_err(|e| ServerError::BadRequest(format!("model field: {e}")))?,
                );
            }
            "language" => {
                language = Some(
                    field
                        .text()
                        .await
                        .map_err(|e| ServerError::BadRequest(format!("language field: {e}")))?,
                );
            }
            "file" => {
                audio_bytes = Some(
                    field
                        .bytes()
                        .await
                        .map_err(|e| ServerError::BadRequest(format!("file field: {e}")))?
                        .to_vec(),
                );
            }
            _ => {} // ignore unknown fields per OpenAI spec
        }
    }

    let model = model.ok_or_else(|| ServerError::BadRequest("missing 'model' field".into()))?;
    let audio =
        audio_bytes.ok_or_else(|| ServerError::BadRequest("missing 'file' field".into()))?;

    if !state.registry.contains(&model) {
        return Err(ServerError::BadRequest(format!("unknown model: {}", model)));
    }

    let inference_req = InferenceRequest {
        task: Task::Stt,
        model,
        messages: vec![],
        input: None,
        voice: None,
        format: None,
        language,
        audio: Some(B64.encode(&audio)),
        audio_format: None,
        query: None,
        documents: vec![],
        top_k: None,
        max_tokens: None,
        temperature: None,
        top_p: None,
        stream: false,
        think: None,
        think_budget: None,
    };

    handle_stt(state, inference_req).await
}

// ---------------------------------------------------------------------------
// Embedding handler + OpenAI /v1/embeddings alias
// ---------------------------------------------------------------------------

async fn handle_embed(
    state: Arc<InferenceState>,
    req: InferenceRequest,
) -> Result<axum::response::Response, ServerError> {
    let _permit = state
        .embed_semaphore
        .clone()
        .acquire_owned()
        .await
        .map_err(|_| ServerError::Internal("embed semaphore closed".into()))?;

    let engine = state.registry.get_embed(&req.model).ok_or_else(|| {
        ServerError::BadRequest(format!("model `{}` is not an embedding model", req.model))
    })?;

    // `input` can be a JSON string or an array of strings (OpenAI convention).
    let input: Vec<String> = match req.input {
        Some(serde_json::Value::String(s)) => vec![s],
        Some(serde_json::Value::Array(arr)) => arr
            .into_iter()
            .map(|v| {
                v.as_str().map(|s| s.to_string()).ok_or_else(|| {
                    ServerError::BadRequest("input array must contain only strings".into())
                })
            })
            .collect::<Result<Vec<_>, _>>()?,
        Some(other) => {
            return Err(ServerError::BadRequest(format!(
                "input must be a string or array of strings, got: {other}"
            )));
        }
        None => {
            return Err(ServerError::BadRequest(
                "embedding request missing 'input' field".into(),
            ));
        }
    };

    let embed_req = EngineEmbedRequest {
        model: req.model.clone(),
        input,
    };

    let output = engine
        .embed(embed_req)
        .await
        .map_err(|e| ServerError::Inference(format!("embedding: {e}")))?;

    // OpenAI-compatible response shape.
    let data: Vec<serde_json::Value> = output
        .embeddings
        .into_iter()
        .enumerate()
        .map(|(i, vec)| {
            serde_json::json!({
                "object": "embedding",
                "embedding": vec,
                "index": i,
            })
        })
        .collect();

    Ok(axum::Json(serde_json::json!({
        "object": "list",
        "data": data,
        "model": req.model,
        "usage": {
            "prompt_tokens": 0,
            "total_tokens": 0,
        }
    }))
    .into_response())
}

/// OpenAI-compatible request body for `POST /v1/embeddings`.
#[derive(Debug, Clone, serde::Deserialize)]
pub struct OpenAiEmbedRequest {
    pub model: String,
    /// String or array of strings to embed.
    pub input: serde_json::Value,
    /// "float" (default) or "base64" — only "float" supported in v1.
    #[serde(default)]
    pub encoding_format: Option<String>,
    /// Ignored — model defines its own dimensionality (2048 for Qwen3-Embedding-2B).
    #[serde(default)]
    pub dimensions: Option<usize>,
    /// Ignored — for billing pass-through only.
    #[serde(default)]
    pub user: Option<String>,
}

/// OpenAI alias: POST /v1/embeddings
pub async fn openai_embeddings(
    State(state): State<Arc<InferenceState>>,
    Json(req): Json<OpenAiEmbedRequest>,
) -> Result<axum::response::Response, ServerError> {
    if !state.registry.contains(&req.model) {
        return Err(ServerError::BadRequest(format!(
            "unknown model: {}",
            req.model
        )));
    }
    let inference_req = InferenceRequest {
        task: Task::Embedding,
        model: req.model,
        messages: vec![],
        input: Some(req.input),
        voice: None,
        format: None,
        language: None,
        audio: None,
        audio_format: None,
        query: None,
        documents: vec![],
        top_k: None,
        max_tokens: None,
        temperature: None,
        top_p: None,
        stream: false,
        think: None,
        think_budget: None,
    };
    handle_embed(state, inference_req).await
}

// ---------------------------------------------------------------------------
// Rerank handler + custom /v1/rerank endpoint
// ---------------------------------------------------------------------------

async fn handle_rerank(
    state: Arc<InferenceState>,
    req: InferenceRequest,
) -> Result<axum::response::Response, ServerError> {
    let _permit = state
        .rerank_semaphore
        .clone()
        .acquire_owned()
        .await
        .map_err(|_| ServerError::Internal("rerank semaphore closed".into()))?;

    let engine = state.registry.get_rerank(&req.model).ok_or_else(|| {
        ServerError::BadRequest(format!("model `{}` is not a rerank model", req.model))
    })?;

    let query = req
        .query
        .ok_or_else(|| ServerError::BadRequest("rerank request missing 'query' field".into()))?;

    if req.documents.is_empty() {
        return Err(ServerError::BadRequest(
            "rerank request must have a non-empty 'documents' array".into(),
        ));
    }

    let rerank_req = EngineRerankRequest {
        model: req.model.clone(),
        query,
        documents: req.documents,
        top_k: req.top_k,
    };

    let output = engine
        .rerank(rerank_req)
        .await
        .map_err(|e| ServerError::Inference(format!("rerank: {e}")))?;

    Ok(axum::Json(serde_json::json!({
        "results": output.results,
        "model": req.model,
    }))
    .into_response())
}

/// Custom request body for `POST /v1/rerank`.
#[derive(Debug, Clone, serde::Deserialize)]
pub struct CustomRerankRequest {
    pub model: String,
    pub query: String,
    pub documents: Vec<String>,
    #[serde(default)]
    pub top_k: Option<usize>,
}

/// Custom endpoint: POST /v1/rerank
pub async fn custom_rerank(
    State(state): State<Arc<InferenceState>>,
    Json(req): Json<CustomRerankRequest>,
) -> Result<axum::response::Response, ServerError> {
    if !state.registry.contains(&req.model) {
        return Err(ServerError::BadRequest(format!(
            "unknown model: {}",
            req.model
        )));
    }
    let inference_req = InferenceRequest {
        task: Task::Rerank,
        model: req.model,
        messages: vec![],
        input: None,
        voice: None,
        format: None,
        language: None,
        audio: None,
        audio_format: None,
        query: Some(req.query),
        documents: req.documents,
        top_k: req.top_k,
        max_tokens: None,
        temperature: None,
        top_p: None,
        stream: false,
        think: None,
        think_budget: None,
    };
    handle_rerank(state, inference_req).await
}

// ---------------------------------------------------------------------------
// Chat handler
// ---------------------------------------------------------------------------

/// Unified-endpoint entry point for chat (no vision; image_url via /v1/chat/completions only).
async fn handle_chat(
    state: Arc<InferenceState>,
    req: InferenceRequest,
) -> Result<axum::response::Response, ServerError> {
    handle_chat_with_vision(state, req, None).await
}

/// Core chat dispatch that accepts pre-encoded vision embeddings (may be None).
async fn handle_chat_with_vision(
    state: Arc<InferenceState>,
    req: InferenceRequest,
    vision_embeds: Option<(Vec<f32>, usize)>,
) -> Result<axum::response::Response, ServerError> {
    let engine = state.registry.get_chat(&req.model).ok_or_else(|| {
        ServerError::BadRequest(format!("model `{}` is not a chat model", req.model))
    })?;

    let stream = req.stream;
    let model_name = req.model.clone();
    let max_tokens = req.max_tokens.unwrap_or(1024);

    if req.messages.is_empty() {
        return Err(ServerError::BadRequest(
            "messages must not be empty for chat".into(),
        ));
    }

    let chat_req = ChatRequest {
        model: req.model,
        messages: req.messages,
        max_tokens: req.max_tokens,
        temperature: req.temperature,
        top_p: req.top_p,
        stream: req.stream,
        think: req.think,
        think_budget: req.think_budget,
        vision_embeds,
    };

    let (tx, rx) = mpsc::channel::<ChatEvent>(64);
    let engine_clone = engine.clone();
    tokio::spawn(async move {
        let _ = engine_clone.chat(chat_req, tx).await;
    });

    if stream {
        Ok(stream_response(model_name, max_tokens, rx).into_response())
    } else {
        Ok(non_stream_response(model_name, max_tokens, rx)
            .await?
            .into_response())
    }
}

async fn non_stream_response(
    model: String,
    _max_tokens: u32,
    mut rx: mpsc::Receiver<ChatEvent>,
) -> Result<Json<ChatCompletionResponse>, ServerError> {
    let started = Utc::now().timestamp();
    let mut content = String::new();
    let mut output_tokens: u32 = 0;
    let mut finish: FinishReason = FinishReason::Length;

    while let Some(ev) = rx.recv().await {
        match ev {
            ChatEvent::Token(_, text) => {
                content.push_str(&text);
                output_tokens += 1;
            }
            ChatEvent::Finish(r) => {
                finish = r;
                break;
            }
            ChatEvent::Error(e) => return Err(ServerError::Inference(e)),
        }
    }

    Ok(Json(ChatCompletionResponse {
        id: format!("chatcmpl-{}", Uuid::new_v4()),
        object: "chat.completion",
        created: started as u64,
        model,
        choices: vec![Choice {
            index: 0,
            message: ResponseMessage {
                role: "assistant".into(),
                content,
            },
            finish_reason: Some(finish.as_openai().into()),
        }],
        usage: Usage {
            prompt_tokens: 0, // engine doesn't currently report this; v1 leaves as 0
            completion_tokens: output_tokens,
            total_tokens: output_tokens,
        },
    }))
}

fn stream_response(
    model: String,
    _max_tokens: u32,
    rx: mpsc::Receiver<ChatEvent>,
) -> impl IntoResponse {
    let id = format!("chatcmpl-{}", Uuid::new_v4());
    let created = Utc::now().timestamp() as u64;

    let stream = stream::unfold(
        (rx, id.clone(), model, created),
        |(mut rx, id, model, created)| async move {
            match rx.recv().await {
                Some(ChatEvent::Token(_, text)) => {
                    let chunk = ChatCompletionChunk {
                        id: id.clone(),
                        object: "chat.completion.chunk",
                        created,
                        model: model.clone(),
                        choices: vec![ChunkChoice {
                            index: 0,
                            delta: ChunkDelta {
                                role: None,
                                content: Some(text),
                            },
                            finish_reason: None,
                        }],
                    };
                    let ev = axum::response::sse::Event::default()
                        .data(serde_json::to_string(&chunk).unwrap());
                    Some((ev, (rx, id, model, created)))
                }
                Some(ChatEvent::Finish(r)) => {
                    let chunk = ChatCompletionChunk {
                        id: id.clone(),
                        object: "chat.completion.chunk",
                        created,
                        model: model.clone(),
                        choices: vec![ChunkChoice {
                            index: 0,
                            delta: ChunkDelta::default(),
                            finish_reason: Some(r.as_openai().into()),
                        }],
                    };
                    let ev = axum::response::sse::Event::default()
                        .data(serde_json::to_string(&chunk).unwrap());
                    Some((ev, (rx, id, model, created)))
                }
                Some(ChatEvent::Error(e)) => {
                    let ev = axum::response::sse::Event::default().event("error").data(e);
                    Some((ev, (rx, id, model, created)))
                }
                None => None,
            }
        },
    );

    let stream = stream.chain(stream::once(async {
        axum::response::sse::Event::default().data("[DONE]")
    }));

    into_sse_response(stream)
}
