//! Admin handlers stub — auth/key management wired in M14.
//! Placeholder so the module tree compiles now.

use axum::Json;
use serde_json::json;

pub async fn not_implemented() -> Json<serde_json::Value> {
    Json(json!({"error": "admin endpoints wired in M14"}))
}
