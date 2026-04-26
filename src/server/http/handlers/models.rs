use axum::Json;
use crate::server::openai::response::{ModelInfo, ModelsListResponse};

pub async fn list_models() -> Json<ModelsListResponse> {
    Json(ModelsListResponse {
        object: "list",
        data: vec![ModelInfo {
            id: "qora-4b".into(),
            object: "model",
            created: 1_700_000_000,
            owned_by: "qora".into(),
        }],
    })
}
