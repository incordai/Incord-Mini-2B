//! Unified /v1/inference + OpenAI-compatible aliases.

use axum::response::IntoResponse;
use axum::{
    Extension, Json,
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
use crate::auth::middleware::{AuthIdentity, ClientInfo};
use crate::engine::Task;
use crate::engine::chat::{ChatEvent, ChatMessage, ChatRequest, FinishReason};
use crate::engine::embed::EmbedRequest as EngineEmbedRequest;
use crate::engine::rerank::RerankRequest as EngineRerankRequest;
use crate::engine::stt::SttRequest as EngineSttRequest;
use crate::engine::tts::TtsRequest as EngineTtsRequest;
use crate::error::ServerError;
use crate::history::{retrieve, store};
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
    /// Shared redb instance for both auth keys AND chat history.
    /// (Separate logical tables in the same DB file.)
    pub history_db: Arc<redb::Database>,
    /// History config: embedding model + top_k for context injection.
    pub history_config: crate::config::HistoryConfig,
}

impl InferenceState {
    /// Build with default-permissive semaphores for tests.
    /// Caller must supply a `history_db` (use `crate::db::open` with a tempdir).
    pub fn for_test(registry: Registry, history_db: Arc<redb::Database>) -> Self {
        Self {
            registry,
            vision_semaphore: Arc::new(tokio::sync::Semaphore::new(1)),
            tts_semaphore: Arc::new(tokio::sync::Semaphore::new(2)),
            stt_semaphore: Arc::new(tokio::sync::Semaphore::new(2)),
            embed_semaphore: Arc::new(tokio::sync::Semaphore::new(8)),
            rerank_semaphore: Arc::new(tokio::sync::Semaphore::new(4)),
            history_db,
            history_config: crate::config::HistoryConfig::default(),
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
/// Also supports conversation_id + use_history for RAG-style context injection from chat history.
pub async fn openai_chat_completions(
    State(state): State<Arc<InferenceState>>,
    Extension(identity): Extension<AuthIdentity>,
    Extension(client): Extension<ClientInfo>,
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

    // -----------------------------------------------------------------------
    // History injection: if use_history + conversation_id, embed the query,
    // find top-K relevant prior messages, and prepend them as system context.
    // -----------------------------------------------------------------------
    let conversation_id = req.conversation_id.clone();
    // Derive a stable user_id from whichever identity path was used.
    let user_id = match &identity {
        AuthIdentity::ApiKey { key } => key.user_id.clone(),
        AuthIdentity::User { user } => user.id.clone(),
        AuthIdentity::Guest { guest_id, .. } => guest_id.clone(),
    };

    // History injection — all early-exit conditions checked up front so nesting stays flat.
    'history: {
        let Some(conv_id) = conversation_id.as_ref() else { break 'history };
        if !req.use_history { break 'history }
        let Some(embed_model_id) = state.history_config.embed_model.as_ref() else { break 'history };
        let Some(embed_engine) = state.registry.get_embed(embed_model_id) else { break 'history };

        let query_text = messages
            .iter()
            .rev()
            .find(|m| m.role == "user")
            .map(|m| m.content.clone())
            .unwrap_or_default();
        if query_text.is_empty() { break 'history }

        let embed_req = EngineEmbedRequest {
            model: embed_model_id.clone(),
            input: vec![query_text],
        };
        let _permit = state
            .embed_semaphore
            .clone()
            .acquire_owned()
            .await
            .map_err(|_| ServerError::Internal("embed semaphore closed".into()))?;
        let Ok(out) = embed_engine.embed(embed_req).await else { break 'history };
        let Some(query_emb) = out.embeddings.first() else { break 'history };

        let relevant = retrieve::find_relevant_history(
            &state.history_db,
            conv_id,
            &user_id,
            query_emb,
            state.history_config.top_k,
        )
        .unwrap_or_default();

        if !relevant.is_empty() {
            let mut history_text = String::from("Relevant prior conversation:\n");
            let mut total_chars = 0usize;
            for m in &relevant {
                let line = format!("- [{}] {}\n", m.role, m.content);
                if total_chars + line.len() > state.history_config.max_context_chars {
                    break;
                }
                total_chars += line.len();
                history_text.push_str(&line);
            }
            let mut new_msgs = vec![ChatMessage {
                role: "system".to_string(),
                content: history_text,
            }];
            new_msgs.extend(messages);
            messages = new_msgs;
        }
    }

    // Inject geo context as a system message so the model knows where the user is.
    // Prepended before any history-injected context; only when at least one field is present.
    if let Some(geo_ctx) = build_geo_context(&client) {
        messages.insert(0, ChatMessage {
            role: "system".to_string(),
            content: geo_ctx,
        });
    }

    // Snapshot of user-role messages for history persistence.
    let user_messages_snapshot: Vec<ChatMessage> = messages
        .iter()
        .filter(|m| m.role == "user")
        .cloned()
        .collect();

    let model_name = req.model.clone();
    let stream = req.stream;

    let inference_req = InferenceRequest {
        task: Task::Chat,
        model: req.model.clone(),
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
        stream,
        think: req.think,
        think_budget: req.think_budget,
        show_think: req.show_think,
    };

    // When a conversation_id is set and non-stream: run chat inline so we can
    // capture and persist both user messages and the assistant response.
    // For stream + conversation_id: persist user messages immediately; assistant
    // persistence is deferred (TODO M17.x — wrap stream to accumulate on [DONE]).
    if let Some(conv_id) = conversation_id {
        // Persist user messages (best-effort, no embedding computed in M17 v1).
        for um in &user_messages_snapshot {
            let _ = store::append_message(
                &state.history_db,
                &conv_id,
                &user_id,
                &um.role,
                &um.content,
                None,
            );
        }

        if !stream {
            // Run the chat engine inline so we can capture the assistant response.
            let chat_engine = state.registry.get_chat(&model_name).ok_or_else(|| {
                ServerError::BadRequest(format!("model `{}` is not a chat model", model_name))
            })?;
            let max_tokens = inference_req.max_tokens.unwrap_or(1024);
            let chat_req = ChatRequest {
                model: inference_req.model.clone(),
                messages: inference_req.messages.clone(),
                max_tokens: inference_req.max_tokens,
                temperature: inference_req.temperature,
                top_p: inference_req.top_p,
                stream: false,
                think: inference_req.think,
                think_budget: inference_req.think_budget,
                show_think: inference_req.show_think,
                vision_embeds,
            };
            let (tx, rx) = mpsc::channel::<ChatEvent>(64);
            let engine_clone = chat_engine.clone();
            tokio::spawn(async move {
                let _ = engine_clone.chat(chat_req, tx).await;
            });

            let json_resp = non_stream_response(model_name.clone(), max_tokens, rx).await?;
            let assistant_content = json_resp
                .0
                .choices
                .first()
                .map(|c| c.message.content.clone())
                .unwrap_or_default();

            // Persist assistant response (best-effort).
            let _ = store::append_message(
                &state.history_db,
                &conv_id,
                &user_id,
                "assistant",
                &assistant_content,
                None,
            );

            // Persist guest messages for admin visibility (non-stream path).
            if let AuthIdentity::Guest { guest_id, .. } = &identity {
                let user_content = user_messages_snapshot
                    .first()
                    .map(|m| m.content.as_str())
                    .unwrap_or("");
                let _ = crate::history::guest::append(
                    &state.history_db,
                    guest_id,
                    "user",
                    user_content,
                    &client,
                );
                let _ = crate::history::guest::append(
                    &state.history_db,
                    guest_id,
                    "assistant",
                    &assistant_content,
                    &client,
                );
            }

            return Ok(json_resp.into_response());
        }
        // Stream with conversation_id: user messages already persisted above.
        // TODO M17.x: wrap stream to accumulate tokens and persist assistant on [DONE].
    } else if !stream {
        // No conversation_id — guest messages still need to be persisted (non-stream only).
        // For stream without conversation_id: TODO persist on [DONE].
        if let AuthIdentity::Guest { guest_id, .. } = &identity {
            let chat_engine = state.registry.get_chat(&model_name).ok_or_else(|| {
                ServerError::BadRequest(format!("model `{}` is not a chat model", model_name))
            })?;
            let max_tokens = inference_req.max_tokens.unwrap_or(1024);
            let chat_req = ChatRequest {
                model: inference_req.model.clone(),
                messages: inference_req.messages.clone(),
                max_tokens: inference_req.max_tokens,
                temperature: inference_req.temperature,
                top_p: inference_req.top_p,
                stream: false,
                think: inference_req.think,
                think_budget: inference_req.think_budget,
                show_think: inference_req.show_think,
                vision_embeds,
            };
            let (tx, rx) = mpsc::channel::<ChatEvent>(64);
            let engine_clone = chat_engine.clone();
            tokio::spawn(async move {
                let _ = engine_clone.chat(chat_req, tx).await;
            });

            let json_resp = non_stream_response(model_name.clone(), max_tokens, rx).await?;
            let assistant_content = json_resp
                .0
                .choices
                .first()
                .map(|c| c.message.content.clone())
                .unwrap_or_default();

            let user_content = user_messages_snapshot
                .first()
                .map(|m| m.content.as_str())
                .unwrap_or("");
            let _ = crate::history::guest::append(
                &state.history_db,
                guest_id,
                "user",
                user_content,
                &client,
            );
            let _ = crate::history::guest::append(
                &state.history_db,
                guest_id,
                "assistant",
                &assistant_content,
                &client,
            );
            // TODO M21.x: persist streamed assistant response for guests

            return Ok(json_resp.into_response());
        }
    }

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
        show_think: None,
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
        show_think: None,
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
        show_think: None,
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
        show_think: None,
    };
    handle_rerank(state, inference_req).await
}

// ---------------------------------------------------------------------------
// Geo context helpers
// ---------------------------------------------------------------------------

/// Build a system-message string that tells the AI where the user is located.
/// Returns `None` when no geo fields are present (avoids injecting an empty prefix).
fn build_geo_context(client: &ClientInfo) -> Option<String> {
    let parts: Vec<String> = [
        client.city.as_deref(),
        client.region.as_deref(),
        client.country_name.as_deref().or(client.country.as_deref()),
    ]
    .into_iter()
    .flatten()
    .map(String::from)
    .collect();

    if parts.is_empty() && client.timezone.is_none() {
        return None;
    }

    let mut s = String::from("[Context for assistant]\n");
    if !parts.is_empty() {
        s.push_str(&format!("User location: {}\n", parts.join(", ")));
    }
    if let Some(tz) = &client.timezone {
        s.push_str(&format!("User timezone: {tz}\n"));
    }
    Some(s)
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
        show_think: req.show_think,
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
