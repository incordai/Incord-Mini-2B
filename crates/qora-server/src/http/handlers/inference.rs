//! Unified /v1/inference + OpenAI-compatible aliases.

use std::sync::Arc;
use axum::{extract::State, Json};
use axum::response::IntoResponse;
use chrono::Utc;
use futures::stream::{self, StreamExt};
use tokio::sync::mpsc;
use uuid::Uuid;

use crate::api::inference_request::InferenceRequest;
use crate::api::openai::request::{ChatCompletionRequest, MessageContent, ContentPart};
use crate::api::openai::response::{
    ChatCompletionChunk, ChatCompletionResponse, Choice, ChunkChoice, ChunkDelta,
    ResponseMessage, Usage,
};
use crate::engine::chat::{ChatEvent, ChatMessage, ChatRequest, FinishReason};
use crate::engine::tts::TtsRequest as EngineTtsRequest;
use crate::engine::Task;
use crate::error::ServerError;
use crate::http::sse::into_sse_response;
use crate::registry::Registry;

#[derive(Clone)]
pub struct InferenceState {
    pub registry: Registry,
}

/// Unified endpoint: dispatches by `task` field.
pub async fn unified_inference(
    State(state): State<Arc<InferenceState>>,
    Json(req): Json<InferenceRequest>,
) -> Result<axum::response::Response, ServerError> {
    if !state.registry.contains(&req.model) {
        return Err(ServerError::BadRequest(format!("unknown model: {}", req.model)));
    }
    match req.task {
        Task::Chat => handle_chat(state, req).await,
        Task::Tts => handle_tts(state, req).await,
        Task::Stt => Err(ServerError::Internal("stt engines not yet wired (M12)".into())),
        Task::Embedding => Err(ServerError::Internal("embedding engines not yet wired (M13)".into())),
        Task::Rerank => Err(ServerError::Internal("rerank engines not yet wired (M13)".into())),
    }
}

/// OpenAI alias: /v1/chat/completions accepts ChatCompletionRequest, translates to InferenceRequest.
pub async fn openai_chat_completions(
    State(state): State<Arc<InferenceState>>,
    Json(req): Json<ChatCompletionRequest>,
) -> Result<axum::response::Response, ServerError> {
    let messages: Vec<ChatMessage> = req.messages.into_iter().map(|m| ChatMessage {
        role: m.role,
        content: match m.content {
            MessageContent::Text(s) => s,
            MessageContent::Parts(parts) => {
                let mut s = String::new();
                for p in parts {
                    if let ContentPart::Text { text } = p {
                        s.push_str(&text);
                        s.push('\n');
                    }
                }
                s
            }
        },
    }).collect();

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

    if !state.registry.contains(&inference_req.model) {
        return Err(ServerError::BadRequest(format!("unknown model: {}", inference_req.model)));
    }
    handle_chat(state, inference_req).await
}

// ---------------------------------------------------------------------------
// TTS handler + OpenAI /v1/audio/speech alias
// ---------------------------------------------------------------------------

async fn handle_tts(
    state: Arc<InferenceState>,
    req: InferenceRequest,
) -> Result<axum::response::Response, ServerError> {
    let engine = state.registry.get_tts(&req.model)
        .ok_or_else(|| ServerError::BadRequest(format!("model `{}` is not a tts model", req.model)))?;

    // Extract input text from the generic `input` field.
    let input_text = match req.input {
        Some(serde_json::Value::String(s)) => s,
        Some(other) => return Err(ServerError::BadRequest(
            format!("tts input must be a string, got: {other}")
        )),
        None => return Err(ServerError::BadRequest("tts request missing 'input' field".into())),
    };

    let tts_req = EngineTtsRequest {
        model: req.model.clone(),
        input: input_text,
        voice: req.voice,
        format: req.format,
        language: req.language,
    };

    let output = engine.synthesize(tts_req).await
        .map_err(|e| ServerError::Inference(format!("tts synthesis: {e}")))?;

    let content_type = match output.format.as_str() {
        "wav"  => "audio/wav",
        "mp3"  => "audio/mpeg",
        _      => "application/octet-stream",
    };

    Ok((
        [(axum::http::header::CONTENT_TYPE, content_type)],
        output.audio,
    ).into_response())
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
        return Err(ServerError::BadRequest(
            format!("unknown model: {}", inference_req.model)
        ));
    }
    handle_tts(state, inference_req).await
}

// ---------------------------------------------------------------------------
// Chat handler
// ---------------------------------------------------------------------------

async fn handle_chat(
    state: Arc<InferenceState>,
    req: InferenceRequest,
) -> Result<axum::response::Response, ServerError> {
    let engine = state.registry.get_chat(&req.model)
        .ok_or_else(|| ServerError::BadRequest(format!("model `{}` is not a chat model", req.model)))?;

    let stream = req.stream;
    let model_name = req.model.clone();
    let max_tokens = req.max_tokens.unwrap_or(1024);

    let messages = req.messages.clone();
    if messages.is_empty() {
        return Err(ServerError::BadRequest("messages must not be empty for chat".into()));
    }

    let chat_req = ChatRequest {
        model: req.model,
        messages,
        max_tokens: req.max_tokens,
        temperature: req.temperature,
        top_p: req.top_p,
        stream: req.stream,
        think: req.think,
        think_budget: req.think_budget,
        vision_embeds: None,
    };

    let (tx, rx) = mpsc::channel::<ChatEvent>(64);
    let engine_clone = engine.clone();
    tokio::spawn(async move {
        let _ = engine_clone.chat(chat_req, tx).await;
    });

    if stream {
        Ok(stream_response(model_name, max_tokens, rx).into_response())
    } else {
        Ok(non_stream_response(model_name, max_tokens, rx).await?.into_response())
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
            ChatEvent::Token(_, text) => { content.push_str(&text); output_tokens += 1; }
            ChatEvent::Finish(r) => { finish = r; break; }
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
            message: ResponseMessage { role: "assistant".into(), content },
            finish_reason: Some(finish.as_openai().into()),
        }],
        usage: Usage {
            prompt_tokens: 0,  // engine doesn't currently report this; v1 leaves as 0
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
                            delta: ChunkDelta { role: None, content: Some(text) },
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
