use axum::{Json, extract::State};
use serde::Serialize;
use std::sync::Arc;

use crate::http::handlers::inference::InferenceState;

#[derive(Serialize)]
pub struct ModelsListResponse {
    object: &'static str,
    data: Vec<ModelInfo>,
}

#[derive(Serialize)]
pub struct ModelInfo {
    id: String,
    object: &'static str,
    created: u64,
    owned_by: String,
    description: String,
    task: String,
}

pub async fn list_models(State(state): State<Arc<InferenceState>>) -> Json<ModelsListResponse> {
    let mut data: Vec<ModelInfo> = state
        .registry
        .all()
        .into_iter()
        .map(|engine| ModelInfo {
            id: engine.id().to_string(),
            object: "model",
            created: 1_700_000_000,
            owned_by: "incord".to_string(),
            description: engine.description().to_string(),
            task: format!("{:?}", engine.task()).to_lowercase(),
        })
        .collect();

    // Synthesise the auto-router as a virtual chat model, pinned to the top
    // so the chat UI can default-select it. Only exposed when there's at
    // least one real chat model behind it.
    let has_chat = data.iter().any(|m| m.task == "chat");
    if has_chat {
        data.insert(
            0,
            ModelInfo {
                id: crate::router::AUTO_MODEL_ID.to_string(),
                object: "model",
                created: 1_700_000_000,
                owned_by: "incord".to_string(),
                description:
                    "Auto — small model for chitchat, big model for reasoning."
                        .to_string(),
                task: "chat".to_string(),
            },
        );
    }

    Json(ModelsListResponse {
        object: "list",
        data,
    })
}
