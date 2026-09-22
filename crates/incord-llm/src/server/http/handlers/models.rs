use crate::server::openai::response::{ModelInfo, ModelsListResponse};
use axum::Json;

pub async fn list_models() -> Json<ModelsListResponse> {
    Json(ModelsListResponse {
        object: "list",
        data: vec![
            ModelInfo {
                id: "incord-mini-2b".into(),
                object: "model",
                created: 1_700_000_000,
                owned_by: "incord".into(),
            },
            ModelInfo {
                id: "incord".into(),
                object: "model",
                created: 1_700_000_000,
                owned_by: "incord".into(),
            },
            ModelInfo {
                id: "incord-mini-2b".into(),
                object: "model",
                created: 1_700_000_000,
                owned_by: "incord".into(),
            },
        ],
    })
}
