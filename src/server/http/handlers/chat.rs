use std::sync::Arc;
use axum::{extract::State, Extension, Json};
use axum::response::IntoResponse;
use chrono::Utc;
use futures::stream::{self, StreamExt};
use tokio::sync::mpsc;
use uuid::Uuid;

use crate::gemv::HybridCache;
use crate::server::auth::keys::ApiKey;
use crate::server::error::{ServerError, ServerResult};
use crate::server::http::sse::into_sse_response;
use crate::server::openai::request::{ChatCompletionRequest, MessageContent, ContentPart, Message};
use crate::server::openai::response::{
    ChatCompletionResponse, Choice, ResponseMessage, Usage,
    ChatCompletionChunk, ChunkChoice, ChunkDelta,
};
use crate::server::scheduler::{queue::{PendingRequest, Queue}, Event as SchedEvent, FinishReason};
use crate::server::session::sampler::Sampler;
use crate::server::session::session::Session;
use crate::server::usage::tracker::{UsageRecord, UsageTracker};
use crate::tokenizer::QoraTokenizer;
use crate::gemv::ModelWeights;

#[derive(Clone)]
pub struct ChatState {
    pub queue: Arc<Queue>,
    pub weights: Arc<ModelWeights>,
    pub tokenizer: Arc<QoraTokenizer>,
    pub usage: UsageTracker,
}

pub async fn chat_completions(
    State(state): State<Arc<ChatState>>,
    Extension(key): Extension<ApiKey>,
    Json(req): Json<ChatCompletionRequest>,
) -> Result<axum::response::Response, ServerError> {
    if req.model != "qora-4b" {
        return Err(ServerError::BadRequest(format!("unknown model: {}", req.model)));
    }

    // Build the user message string. OpenAI sends a sequence of system/user/assistant messages;
    // QORA's tokenizer.format_chat takes a single user_message and builds the full prompt
    // (with system prompt + chat template) internally. For v1, concatenate all messages
    // into a single user message; system content becomes a prefix.
    let user_message = render_user_message(&req.messages)?;
    let think = req.think.unwrap_or(true);
    let max_tokens = req.max_tokens.unwrap_or(1024) as usize;

    let prompt_tokens = state.tokenizer.format_chat(&user_message, think, max_tokens);

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

    let (tx, rx) = mpsc::channel::<SchedEvent>(64);
    let pending = PendingRequest { session, sse_tx: tx };
    state.queue.push(pending).map_err(|_| ServerError::QueueFull { retry_after_secs: 5 })?;

    if req.stream {
        Ok(stream_response(state, req.model.clone(), prompt_tokens.len(), rx).await.into_response())
    } else {
        Ok(non_stream_response(state, req.model.clone(), key, prompt_tokens.len(), rx).await?.into_response())
    }
}

fn render_user_message(messages: &[Message]) -> ServerResult<String> {
    if messages.is_empty() {
        return Err(ServerError::BadRequest("messages array must not be empty".into()));
    }
    let mut combined = String::new();
    for m in messages {
        let text = match &m.content {
            MessageContent::Text(t) => t.clone(),
            MessageContent::Parts(parts) => {
                let mut s = String::new();
                for p in parts {
                    if let ContentPart::Text { text } = p {
                        s.push_str(text);
                        s.push('\n');
                    }
                    // ImageUrl handled in M6 vision integration
                }
                s
            }
        };
        // Prefix non-user messages with role marker so the model sees context.
        if m.role == "user" {
            combined.push_str(&text);
        } else {
            combined.push_str(&format!("[{}] {}\n", m.role, text));
        }
    }
    Ok(combined)
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
            SchedEvent::Finish(r) => { finish = r; break; }
            SchedEvent::Error(e) => { return Err(ServerError::Inference(e)); }
        }
    }

    let resp = ChatCompletionResponse {
        id: format!("chatcmpl-{}", Uuid::new_v4()),
        object: "chat.completion",
        created: started as u64,
        model: model.clone(),
        choices: vec![Choice {
            index: 0,
            message: ResponseMessage { role: "assistant".into(), content },
            finish_reason: Some(finish.as_openai().into()),
        }],
        usage: Usage {
            prompt_tokens: prompt_tokens as u32,
            completion_tokens: output_tokens,
            total_tokens: prompt_tokens as u32 + output_tokens,
        },
    };

    let duration_ms = ((Utc::now().timestamp() - started).max(0)) as u32 * 1000;
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

    let stream = stream::unfold(
        (rx, id.clone(), model.clone(), created),
        |(mut rx, id, model, created)| async move {
            match rx.recv().await {
                Some(SchedEvent::Token(_, text)) => {
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
                Some(SchedEvent::Finish(r)) => {
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
                Some(SchedEvent::Error(e)) => {
                    let ev = axum::response::sse::Event::default().event("error").data(e);
                    Some((ev, (rx, id, model, created)))
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
