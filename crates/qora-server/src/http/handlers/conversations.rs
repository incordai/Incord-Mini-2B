//! Chat history HTTP endpoints.

use std::sync::Arc;

use axum::{
    Extension, Json,
    extract::{Path, State},
    response::IntoResponse,
};
use serde::Serialize;
use serde_json::json;

use crate::auth::middleware::AuthIdentity;
use crate::error::ServerError;
use crate::history::{retrieve, store};
use crate::http::handlers::inference::InferenceState;

/// Derive a stable user_id string from any AuthIdentity.
fn user_id_from_identity(identity: &AuthIdentity) -> String {
    match identity {
        AuthIdentity::ApiKey { key } => key.user_id.clone(),
        AuthIdentity::User { user } => user.id.clone(),
        AuthIdentity::Guest { guest_id, .. } => guest_id.clone(),
    }
}

// ---------------------------------------------------------------------------
// POST /v1/conversations — create a new conversation
// ---------------------------------------------------------------------------

#[derive(serde::Deserialize)]
pub struct CreateConvRequest {
    pub title: Option<String>,
}

#[derive(Serialize)]
pub struct CreateConvResponse {
    pub id: String,
    pub created_at: i64,
}

pub async fn create_conversation_handler(
    State(state): State<Arc<InferenceState>>,
    Extension(identity): Extension<AuthIdentity>,
    Json(req): Json<CreateConvRequest>,
) -> Result<Json<CreateConvResponse>, ServerError> {
    let user_id = user_id_from_identity(&identity);
    let conv = store::create_conversation(&state.history_db, &user_id, req.title)
        .map_err(|e| ServerError::Db(e.to_string()))?;
    Ok(Json(CreateConvResponse {
        id: conv.id,
        created_at: conv.created_at,
    }))
}

// ---------------------------------------------------------------------------
// GET /v1/conversations — list all conversations for the authenticated user
// ---------------------------------------------------------------------------

pub async fn list_conversations_handler(
    State(state): State<Arc<InferenceState>>,
    Extension(identity): Extension<AuthIdentity>,
) -> Result<Json<serde_json::Value>, ServerError> {
    let user_id = user_id_from_identity(&identity);
    let convs = retrieve::list_user_conversations(&state.history_db, &user_id)
        .map_err(|e| ServerError::Db(e.to_string()))?;
    Ok(Json(json!({
        "object": "list",
        "data": convs,
    })))
}

// ---------------------------------------------------------------------------
// GET /v1/conversations/{id} — get conversation + messages
// ---------------------------------------------------------------------------

pub async fn get_conversation_handler(
    State(state): State<Arc<InferenceState>>,
    Extension(identity): Extension<AuthIdentity>,
    Path(id): Path<String>,
) -> Result<Json<serde_json::Value>, ServerError> {
    let user_id = user_id_from_identity(&identity);
    let conv = retrieve::get_conversation(&state.history_db, &id, &user_id)
        .map_err(|e| ServerError::Db(e.to_string()))?
        .ok_or_else(|| ServerError::BadRequest("conversation not found".into()))?;

    let msgs = retrieve::get_messages(&state.history_db, &id, &user_id)
        .map_err(|e| ServerError::Db(e.to_string()))?;

    Ok(Json(json!({
        "id": conv.id,
        "title": conv.title,
        "created_at": conv.created_at,
        "updated_at": conv.updated_at,
        "messages": msgs.iter().map(|m| json!({
            "idx": m.idx,
            "role": m.role,
            "content": m.content,
            "timestamp": m.timestamp,
        })).collect::<Vec<_>>(),
    })))
}

// ---------------------------------------------------------------------------
// GET /v1/conversations/{id}/export — full export including embeddings
// ---------------------------------------------------------------------------

pub async fn export_conversation_handler(
    State(state): State<Arc<InferenceState>>,
    Extension(identity): Extension<AuthIdentity>,
    Path(id): Path<String>,
) -> Result<axum::response::Response, ServerError> {
    let user_id = user_id_from_identity(&identity);
    let conv = retrieve::get_conversation(&state.history_db, &id, &user_id)
        .map_err(|e| ServerError::Db(e.to_string()))?
        .ok_or_else(|| ServerError::BadRequest("conversation not found".into()))?;

    let msgs = retrieve::get_messages(&state.history_db, &id, &user_id)
        .map_err(|e| ServerError::Db(e.to_string()))?;

    // Export full data INCLUDING embeddings
    let body = json!({
        "id": conv.id,
        "user_id": conv.user_id,
        "title": conv.title,
        "created_at": conv.created_at,
        "updated_at": conv.updated_at,
        "message_count": conv.message_count,
        "messages": msgs,
    });

    let body_bytes = serde_json::to_vec_pretty(&body)
        .map_err(|e| ServerError::Internal(e.to_string()))?;

    let filename = format!("qora-chat-{}.json", id);
    Ok((
        [
            (
                axum::http::header::CONTENT_TYPE,
                "application/json".to_string(),
            ),
            (
                axum::http::header::CONTENT_DISPOSITION,
                format!("attachment; filename=\"{}\"", filename),
            ),
        ],
        body_bytes,
    )
        .into_response())
}

// ---------------------------------------------------------------------------
// DELETE /v1/conversations/{id} — delete a conversation
// ---------------------------------------------------------------------------

pub async fn delete_conversation_handler(
    State(state): State<Arc<InferenceState>>,
    Extension(identity): Extension<AuthIdentity>,
    Path(id): Path<String>,
) -> Result<Json<serde_json::Value>, ServerError> {
    let user_id = user_id_from_identity(&identity);
    // Pre-check: verify existence + ownership (returns None for both not-found and wrong owner).
    let conv = retrieve::get_conversation(&state.history_db, &id, &user_id)
        .map_err(|e| ServerError::Db(e.to_string()))?;
    if conv.is_none() {
        return Err(ServerError::BadRequest("conversation not found".into()));
    }

    // Now delete — ownership is already verified above, so bail! won't fire.
    let existed = store::delete_conversation(&state.history_db, &id, &user_id)
        .map_err(|e| ServerError::Db(e.to_string()))?;
    if !existed {
        return Err(ServerError::BadRequest("conversation not found".into()));
    }
    Ok(Json(json!({"id": id, "deleted": true})))
}
