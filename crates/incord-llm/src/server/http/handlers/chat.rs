use axum::response::IntoResponse;
use axum::{Extension, Json, extract::State};
use chrono::Utc;
use futures::stream::{self, StreamExt};
use std::sync::Arc;
use tokio::sync::mpsc;
use uuid::Uuid;

use crate::gemv::HybridCache;
use crate::gemv::ModelWeights;
use crate::server::auth::keys::ApiKey;
use crate::server::error::{ServerError, ServerResult};
use crate::server::http::sse::into_sse_response;
use crate::server::openai::request::{ChatCompletionRequest, ContentPart, Message, MessageContent};
use crate::server::openai::response::{
    ChatCompletionChunk, ChatCompletionResponse, Choice, ChunkChoice, ChunkDelta, ResponseMessage,
    Usage,
};
use crate::server::scheduler::{
    Event as SchedEvent, FinishReason,
    queue::{PendingRequest, Queue},
};
use crate::server::session::core::Session;
use crate::server::session::sampler::Sampler;
use crate::server::usage::tracker::{UsageRecord, UsageTracker};
use crate::tokenizer::IncordTokenizer;

#[derive(Clone)]
pub struct ChatState {
    pub queue: Arc<Queue>,
    pub weights: Arc<ModelWeights>,
    pub tokenizer: Arc<IncordTokenizer>,
    pub usage: UsageTracker,
    /// Semaphore limiting concurrent vision encodes (protects peak memory).
    pub vision_semaphore: Arc<tokio::sync::Semaphore>,
}

pub async fn chat_completions(
    State(state): State<Arc<ChatState>>,
    Extension(key): Extension<ApiKey>,
    Json(req): Json<ChatCompletionRequest>,
) -> Result<axum::response::Response, ServerError> {
    let valid_model = matches!(
        req.model.as_str(),
        "incord" | "incord-mini-2b" | "incord-2b"
    );
    if !valid_model {
        return Err(ServerError::BadRequest(format!(
            "unknown model: {}",
            req.model
        )));
    }

    // Build the user message string and collect any inline/remote image bytes.
    // OpenAI sends a sequence of system/user/assistant messages;
    // Incord's tokenizer.format_chat takes a single user_message and builds the full prompt
    // (with system prompt + chat template) internally. For v1, concatenate all messages
    // into a single user message; system content becomes a prefix.
    let (user_message, images) = render_user_message_with_images(&req.messages).await?;
    let think = req.think.unwrap_or(true);
    let max_tokens = req.max_tokens.unwrap_or(1024) as usize;

    let prompt_tokens = state
        .tokenizer
        .format_chat(&user_message, think, max_tokens);

    // Build session
    let session = Session {
        id: Uuid::new_v4(),
        key_id: key.id.clone(),
        user_id: key.user_id.clone(),
        model_name: req.model.clone(),
        cache: HybridCache::new(&state.weights.config),
        prompt_tokens: prompt_tokens.clone(),
        generated_tokens: Vec::new(),
        max_new_tokens: max_tokens,
        eos_token_id: state.weights.config.eos_token_id,
        sampler: Sampler::new(
            req.temperature.unwrap_or(0.7),
            req.top_p.unwrap_or(0.95),
            20,
            req.presence_penalty.unwrap_or(1.5),
            req.frequency_penalty.unwrap_or(1.0),
        ),
        started_at: Utc::now().timestamp(),
    };

    // If there are images, encode the first one (v1: single-image per request).
    let vision_embeds: Option<(Vec<f32>, usize)> = if !images.is_empty() {
        let img_bytes = images.into_iter().next().unwrap();
        let permit = state
            .vision_semaphore
            .clone()
            .acquire_owned()
            .await
            .map_err(|_| ServerError::Internal("vision semaphore closed".into()))?;
        let weights_clone = state.weights.clone();
        let result: Result<_, ServerError> = tokio::task::spawn_blocking(move || {
            let _permit = permit; // hold permit for duration of encode
            let vision = weights_clone
                .vision
                .as_ref()
                .ok_or_else(|| ServerError::Internal("model has no vision weights".into()))?;
            let img = image::load_from_memory(&img_bytes)
                .map_err(|e| ServerError::BadRequest(format!("decode image: {e}")))?;
            let rgb = img.to_rgb8();
            let h = rgb.height() as usize;
            let w = rgb.width() as usize;
            let (pixels, ph, pw) = crate::vision::pixels_from_rgb_bytes(rgb.as_raw(), w, h);
            let (embeds, num_tokens) = vision.encode_image(&pixels, ph, pw);
            Ok::<_, ServerError>((embeds, num_tokens))
        })
        .await
        .map_err(|e| ServerError::Internal(format!("vision task panicked: {e}")))?;
        Some(result?)
    } else {
        None
    };

    let (tx, rx) = mpsc::channel::<SchedEvent>(64);
    let pending = PendingRequest {
        session,
        sse_tx: tx,
        vision_embeds,
    };
    crate::server::metrics::REQUESTS_TOTAL
        .with_label_values(&["chat_completions", "received"])
        .inc();

    state
        .queue
        .push(pending)
        .map_err(|_| ServerError::QueueFull {
            retry_after_secs: 5,
        })?;

    if req.stream {
        Ok(
            stream_response(state, req.model.clone(), prompt_tokens.len(), rx)
                .await
                .into_response(),
        )
    } else {
        Ok(
            non_stream_response(state, req.model.clone(), key, prompt_tokens.len(), rx)
                .await?
                .into_response(),
        )
    }
}

/// Build the user message string and collect raw image bytes from any ImageUrl content parts.
/// Returns (combined_text, image_bytes_vec).
async fn render_user_message_with_images(
    messages: &[Message],
) -> ServerResult<(String, Vec<Vec<u8>>)> {
    use crate::server::openai::vision::{VisionContent, parse_image_url};

    if messages.is_empty() {
        return Err(ServerError::BadRequest(
            "messages array must not be empty".into(),
        ));
    }

    let mut combined = String::new();
    let mut images: Vec<Vec<u8>> = Vec::new();

    for m in messages {
        let mut text = String::new();
        match &m.content {
            MessageContent::Text(t) => text.push_str(t),
            MessageContent::Parts(parts) => {
                for p in parts {
                    match p {
                        ContentPart::Text { text: t } => {
                            text.push_str(t);
                            text.push('\n');
                        }
                        ContentPart::ImageUrl { image_url } => {
                            let parsed = parse_image_url(&image_url.url)
                                .map_err(|e| ServerError::BadRequest(e.to_string()))?;
                            let bytes = match parsed {
                                VisionContent::Inline { bytes, .. } => bytes,
                                VisionContent::Remote { url } => {
                                    let r = reqwest::get(&url).await.map_err(|e| {
                                        ServerError::BadRequest(format!("fetching image: {e}"))
                                    })?;
                                    if !r.status().is_success() {
                                        return Err(ServerError::BadRequest(format!(
                                            "image fetch {} -> {}",
                                            url,
                                            r.status()
                                        )));
                                    }
                                    r.bytes()
                                        .await
                                        .map_err(|e| ServerError::BadRequest(e.to_string()))?
                                        .to_vec()
                                }
                            };
                            // Insert the vision placeholder token sequence into the text.
                            text.push_str("<|vision_start|><|image_pad|><|vision_end|>");
                            images.push(bytes);
                        }
                    }
                }
            }
        }
        // Prefix non-user messages with role marker so the model sees context.
        if m.role == "user" {
            combined.push_str(&text);
        } else {
            combined.push_str(&format!("[{}] {}\n", m.role, text));
        }
    }

    Ok((combined, images))
}

async fn non_stream_response(
    state: Arc<ChatState>,
    model: String,
    key: ApiKey,
    prompt_tokens: usize,
    mut rx: mpsc::Receiver<SchedEvent>,
) -> Result<Json<ChatCompletionResponse>, ServerError> {
    let started = Utc::now().timestamp();
    let mut content = String::new();
    let mut output_tokens = 0u32;
    let mut finish: FinishReason = FinishReason::Length;

    while let Some(ev) = rx.recv().await {
        match ev {
            SchedEvent::Token(_, text) => {
                content.push_str(&text);
                output_tokens += 1;
            }
            SchedEvent::Finish(r) => {
                finish = r;
                break;
            }
            SchedEvent::Error(e) => {
                return Err(ServerError::Inference(e));
            }
        }
    }

    let resp = ChatCompletionResponse {
        id: format!("chatcmpl-{}", Uuid::new_v4()),
        object: "chat.completion",
        created: started as u64,
        model: model.clone(),
        choices: vec![Choice {
            index: 0,
            message: ResponseMessage {
                role: "assistant".into(),
                content,
            },
            finish_reason: Some(finish.as_openai().into()),
        }],
        usage: Usage {
            prompt_tokens: prompt_tokens as u32,
            completion_tokens: output_tokens,
            total_tokens: prompt_tokens as u32 + output_tokens,
        },
    };

    let elapsed_secs = (Utc::now().timestamp() - started).max(0) as f64;
    let duration_ms = (elapsed_secs * 1000.0) as u32;
    let _ = state.usage.record(UsageRecord {
        request_id: resp.id.clone(),
        user_id: key.user_id,
        key_id: key.id,
        model,
        input_tokens: prompt_tokens as u32,
        output_tokens,
        started_at: started,
        duration_ms,
        status: "success".into(),
    });

    crate::server::metrics::TOKENS_EMITTED.inc_by(output_tokens as f64);
    crate::server::metrics::REQUEST_DURATION
        .with_label_values(&["chat_completions"])
        .observe(elapsed_secs);

    Ok(Json(resp))
}

async fn stream_response(
    _state: Arc<ChatState>,
    model: String,
    _prompt_tokens: usize,
    rx: mpsc::Receiver<SchedEvent>,
) -> impl IntoResponse {
    let id = format!("chatcmpl-{}", Uuid::new_v4());
    let created = Utc::now().timestamp() as u64;
    let started = Utc::now().timestamp();

    let stream = stream::unfold(
        (rx, id.clone(), model.clone(), created, started),
        |(mut rx, id, model, created, started)| async move {
            match rx.recv().await {
                Some(SchedEvent::Token(_, text)) => {
                    crate::server::metrics::TOKENS_EMITTED.inc();
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
                    Some((ev, (rx, id, model, created, started)))
                }
                Some(SchedEvent::Finish(r)) => {
                    let elapsed = (Utc::now().timestamp() - started).max(0) as f64;
                    crate::server::metrics::REQUEST_DURATION
                        .with_label_values(&["chat_completions"])
                        .observe(elapsed);
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
                    Some((ev, (rx, id, model, created, started)))
                }
                Some(SchedEvent::Error(e)) => {
                    let ev = axum::response::sse::Event::default().event("error").data(e);
                    Some((ev, (rx, id, model, created, started)))
                }
                None => None,
            }
        },
    );

    // Append the OpenAI [DONE] marker after the channel closes
    let stream = stream.chain(stream::once(async {
        axum::response::sse::Event::default().data("[DONE]")
    }));

    into_sse_response(stream)
}
