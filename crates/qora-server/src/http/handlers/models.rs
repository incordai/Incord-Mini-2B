use std::sync::Arc;
use axum::{extract::State, Json};
use serde::Serialize;

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
    let data = state.registry.all().into_iter().map(|engine| ModelInfo {
        id: engine.id().to_string(),
        object: "model",
        created: 1_700_000_000,
        owned_by: "qora".to_string(),
        description: engine.description().to_string(),
        task: format!("{:?}", engine.task()).to_lowercase(),
    }).collect();
    Json(ModelsListResponse {
        object: "list",
        data,
    })
}
