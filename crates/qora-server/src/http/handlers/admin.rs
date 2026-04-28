//! Admin HTTP handlers — protected by admin_guard (separate admin_token).
//! All endpoints are under /v1/admin/*.
//!
//! SECURITY NOTE: All functions here have elevated privileges. Never call
//! `get_conversation_admin`, `list_all_conversations_admin`, or `get_messages_admin`
//! from user-facing handlers — they bypass user_id privacy checks by design.

use std::collections::HashMap;
use std::sync::Arc;

use axum::{
    Json,
    extract::{Path, Query, State},
    response::IntoResponse,
};
use chrono::Utc;
use serde::{Deserialize, Serialize};
use serde_json::json;

use crate::auth::keys::{create_key, list_all, revoke_key};
use crate::auth::middleware::AuthState;
use crate::error::ServerError;
use crate::history::retrieve as hist_retrieve;
use crate::http::handlers::inference::InferenceState;
use crate::usage::tracker::UsageRecord;

// ─────────────────────────────────────────────────────────────────────────────
// A1. GET /v1/admin/keys
// ─────────────────────────────────────────────────────────────────────────────

/// Public shape for a key list item — NEVER includes hash fields.
#[derive(Serialize)]
pub struct KeyListItem {
    pub id: String,
    pub user_id: String,
    pub name: String,
    pub created_at: i64,
    pub last_used_at: Option<i64>,
    pub is_active: bool,
    pub rpm_limit: u32,
    pub tpm_limit: u32,
}

pub async fn list_keys_handler(
    State(state): State<Arc<AuthState>>,
) -> Result<Json<serde_json::Value>, ServerError> {
    let keys = list_all(&state.db).map_err(|e| ServerError::Db(e.to_string()))?;
    let data: Vec<KeyListItem> = keys
        .into_iter()
        .map(|k| KeyListItem {
            id: k.id,
            user_id: k.user_id,
            name: k.name,
            created_at: k.created_at,
            last_used_at: k.last_used_at,
            is_active: k.is_active,
            rpm_limit: k.rpm_limit,
            tpm_limit: k.tpm_limit,
        })
        .collect();
    Ok(Json(json!({ "object": "list", "data": data })))
}

// ─────────────────────────────────────────────────────────────────────────────
// A1 (existing) — POST /v1/admin/keys + DELETE /v1/admin/keys/{id}
// ─────────────────────────────────────────────────────────────────────────────

#[derive(Deserialize)]
pub struct CreateKeyRequest {
    pub user_id: String,
    pub name: String,
    pub rpm_limit: u32,
    pub tpm_limit: u32,
}

#[derive(Serialize)]
pub struct CreateKeyResponse {
    pub id: String,
    pub raw_key: String,
    pub user_id: String,
    pub created_at: i64,
}

pub async fn create_key_handler(
    State(state): State<Arc<AuthState>>,
    Json(req): Json<CreateKeyRequest>,
) -> Result<Json<CreateKeyResponse>, ServerError> {
    let nk = create_key(
        &state.db,
        &state.server_secret,
        &req.user_id,
        &req.name,
        req.rpm_limit,
        req.tpm_limit,
    )
    .map_err(|e| ServerError::Db(e.to_string()))?;
    Ok(Json(CreateKeyResponse {
        id: nk.key.id,
        raw_key: nk.raw_key,
        user_id: nk.key.user_id,
        created_at: nk.key.created_at,
    }))
}

pub async fn revoke_key_handler(
    State(state): State<Arc<AuthState>>,
    Path(id): Path<String>,
) -> Result<Json<serde_json::Value>, ServerError> {
    let ok = revoke_key(&state.db, &id).map_err(|e| ServerError::Db(e.to_string()))?;
    if !ok {
        return Err(ServerError::BadRequest(format!("no key with id {id}")));
    }
    Ok(Json(json!({"id": id, "revoked": true})))
}

// ─────────────────────────────────────────────────────────────────────────────
// A2. GET /v1/admin/users
// ─────────────────────────────────────────────────────────────────────────────

#[derive(Serialize)]
pub struct UserSummary {
    pub user_id: String,
    pub key_count: u32,
    pub active_key_count: u32,
    pub first_key_created_at: i64,
    pub last_key_used_at: Option<i64>,
    pub total_requests: u32,
    pub total_tokens: u64,
    pub conversation_count: u32,
}

/// Combined state for admin handlers that need both auth (DB) and inference (registry/history).
#[derive(Clone)]
pub struct AdminFullState {
    pub auth: Arc<AuthState>,
    pub inference: Arc<InferenceState>,
}

pub async fn list_users_handler(
    State(state): State<Arc<AdminFullState>>,
) -> Result<Json<serde_json::Value>, ServerError> {
    // TODO: for >10k users, consider a pre-aggregated user stats table; current O(n) is fine for
    // thousands of users but will become slow at very large scale.

    let keys = list_all(&state.auth.db).map_err(|e| ServerError::Db(e.to_string()))?;

    // Group by user_id
    let mut by_user: HashMap<String, Vec<_>> = HashMap::new();
    for k in keys {
        by_user.entry(k.user_id.clone()).or_default().push(k);
    }

    // Load usage records (all, no filter — aggregate per user)
    let tracker = crate::usage::tracker::UsageTracker::new(state.inference.history_db.clone());
    let all_usage = tracker
        .list_usage(Some(0), Some(10000), None, None, None)
        .map_err(|e| ServerError::Db(e.to_string()))?;

    let mut usage_by_user: HashMap<String, (u32, u64)> = HashMap::new(); // (requests, tokens)
    for rec in all_usage {
        let entry = usage_by_user.entry(rec.user_id).or_insert((0, 0));
        entry.0 += 1;
        entry.1 += (rec.input_tokens + rec.output_tokens) as u64;
    }

    let mut summaries: Vec<UserSummary> = by_user
        .into_iter()
        .map(|(user_id, user_keys)| {
            let key_count = user_keys.len() as u32;
            let active_key_count = user_keys.iter().filter(|k| k.is_active).count() as u32;
            let first_key_created_at = user_keys.iter().map(|k| k.created_at).min().unwrap_or(0);
            let last_key_used_at = user_keys
                .iter()
                .filter_map(|k| k.last_used_at)
                .max();

            let conv_count = hist_retrieve::count_conversations_for_user(
                &state.inference.history_db,
                &user_id,
            )
            .unwrap_or(0);

            let (total_requests, total_tokens) =
                usage_by_user.get(&user_id).copied().unwrap_or((0, 0));

            UserSummary {
                user_id,
                key_count,
                active_key_count,
                first_key_created_at,
                last_key_used_at,
                total_requests,
                total_tokens,
                conversation_count: conv_count,
            }
        })
        .collect();

    summaries.sort_by(|a, b| b.first_key_created_at.cmp(&a.first_key_created_at));

    Ok(Json(json!({ "object": "list", "data": summaries })))
}

// ─────────────────────────────────────────────────────────────────────────────
// A3. GET /v1/admin/usage
// ─────────────────────────────────────────────────────────────────────────────

#[derive(Deserialize)]
pub struct UsageQueryParams {
    pub since: Option<i64>,
    pub limit: Option<usize>,
    pub user_id: Option<String>,
    pub model: Option<String>,
    pub status: Option<String>,
}

pub async fn list_usage_handler(
    State(state): State<Arc<AdminFullState>>,
    Query(params): Query<UsageQueryParams>,
) -> Result<Json<serde_json::Value>, ServerError> {
    let tracker = crate::usage::tracker::UsageTracker::new(state.inference.history_db.clone());
    let records = tracker
        .list_usage(
            params.since,
            params.limit,
            params.user_id.as_deref(),
            params.model.as_deref(),
            params.status.as_deref(),
        )
        .map_err(|e| ServerError::Db(e.to_string()))?;

    let count = records.len();
    Ok(Json(json!({
        "object": "list",
        "data": records,
        "count": count,
    })))
}

// ─────────────────────────────────────────────────────────────────────────────
// A4. GET /v1/admin/usage/export.csv
// ─────────────────────────────────────────────────────────────────────────────

pub async fn export_usage_csv_handler(
    State(state): State<Arc<AdminFullState>>,
    Query(params): Query<UsageQueryParams>,
) -> Result<axum::response::Response, ServerError> {
    let tracker = crate::usage::tracker::UsageTracker::new(state.inference.history_db.clone());
    let records: Vec<UsageRecord> = tracker
        .list_usage(
            params.since,
            params.limit,
            params.user_id.as_deref(),
            params.model.as_deref(),
            params.status.as_deref(),
        )
        .map_err(|e| ServerError::Db(e.to_string()))?;

    let mut csv = String::from(
        "request_id,user_id,key_id,model,input_tokens,output_tokens,started_at,duration_ms,status\n",
    );
    for r in &records {
        csv.push_str(&format!(
            "{},{},{},{},{},{},{},{},{}\n",
            escape_csv(&r.request_id),
            escape_csv(&r.user_id),
            escape_csv(&r.key_id),
            escape_csv(&r.model),
            r.input_tokens,
            r.output_tokens,
            r.started_at,
            r.duration_ms,
            escape_csv(&r.status),
        ));
    }

    let ts = Utc::now().timestamp();
    let filename = format!("qora-usage-{ts}.csv");

    Ok((
        [
            (
                axum::http::header::CONTENT_TYPE,
                "text/csv; charset=utf-8".to_string(),
            ),
            (
                axum::http::header::CONTENT_DISPOSITION,
                format!("attachment; filename=\"{filename}\""),
            ),
        ],
        csv,
    )
        .into_response())
}

fn escape_csv(s: &str) -> String {
    if s.contains(',') || s.contains('"') || s.contains('\n') {
        format!("\"{}\"", s.replace('"', "\"\""))
    } else {
        s.to_string()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// A5. GET /v1/admin/stats
// ─────────────────────────────────────────────────────────────────────────────

pub async fn stats_handler(
    State(state): State<Arc<AdminFullState>>,
) -> Result<Json<serde_json::Value>, ServerError> {
    let keys = list_all(&state.auth.db).map_err(|e| ServerError::Db(e.to_string()))?;
    let users: std::collections::HashSet<&str> = keys.iter().map(|k| k.user_id.as_str()).collect();
    let user_count = users.len();
    let active_keys = keys.iter().filter(|k| k.is_active).count();

    let tracker = crate::usage::tracker::UsageTracker::new(state.inference.history_db.clone());

    let now = Utc::now().timestamp();
    let today_start = now - (now % 86400); // floor to today UTC
    let week_start = now - 7 * 86400;

    let today_usage = tracker
        .list_usage(Some(today_start), Some(10000), None, None, None)
        .unwrap_or_default();
    let week_usage = tracker
        .list_usage(Some(week_start), Some(10000), None, None, None)
        .unwrap_or_default();

    let requests_today = today_usage.len() as u32;
    let requests_week = week_usage.len() as u32;
    let tokens_today: u64 = today_usage
        .iter()
        .map(|r| (r.input_tokens + r.output_tokens) as u64)
        .sum();
    let tokens_week: u64 = week_usage
        .iter()
        .map(|r| (r.input_tokens + r.output_tokens) as u64)
        .sum();

    // Count all conversations (admin scan, no user filter)
    let all_convs = hist_retrieve::list_all_conversations_admin(
        &state.inference.history_db,
        None,
        usize::MAX,
    )
    .unwrap_or_default();
    let conversation_count = all_convs.len();

    // Models from registry
    let models: Vec<serde_json::Value> = state
        .inference
        .registry
        .all()
        .iter()
        .map(|e| {
            json!({
                "id": e.id(),
                "kind": format!("{:?}", e.task()).to_lowercase(),
                "description": e.description(),
                "task": format!("{:?}", e.task()).to_lowercase(),
            })
        })
        .collect();

    Ok(Json(json!({
        "totals": {
            "users": user_count,
            "active_keys": active_keys,
            "requests_today": requests_today,
            "requests_week": requests_week,
            "tokens_today": tokens_today,
            "tokens_week": tokens_week,
            "conversations": conversation_count,
        },
        "models": models,
        // active_slots and queue_depth: not tracked as atomic gauges in current metrics.rs
        // (only Prometheus counters/histograms). Stubbed at 0 for v1; add AtomicUsize
        // gauges to metrics.rs in a future milestone if needed.
        "active_slots": 0,
        "queue_depth": 0,
    })))
}

// ─────────────────────────────────────────────────────────────────────────────
// A6. GET /v1/admin/conversations — list all conversations (admin override)
// ─────────────────────────────────────────────────────────────────────────────

#[derive(Deserialize)]
pub struct AdminConvQueryParams {
    pub user_id: Option<String>,
    pub limit: Option<usize>,
}

pub async fn list_all_conversations_handler(
    State(state): State<Arc<AdminFullState>>,
    Query(params): Query<AdminConvQueryParams>,
) -> Result<Json<serde_json::Value>, ServerError> {
    let limit = params.limit.unwrap_or(200).min(1000);
    let convs = hist_retrieve::list_all_conversations_admin(
        &state.inference.history_db,
        params.user_id.as_deref(),
        limit,
    )
    .map_err(|e| ServerError::Db(e.to_string()))?;

    let data: Vec<serde_json::Value> = convs
        .iter()
        .map(|c| {
            json!({
                "id": c.id,
                "user_id": c.user_id,
                "title": c.title,
                "message_count": c.message_count,
                "created_at": c.created_at,
                "updated_at": c.updated_at,
            })
        })
        .collect();

    Ok(Json(json!({ "object": "list", "data": data })))
}

// ─────────────────────────────────────────────────────────────────────────────
// A7. GET /v1/admin/conversations/{id} — full conversation detail (admin override)
// ─────────────────────────────────────────────────────────────────────────────

pub async fn get_conversation_admin_handler(
    State(state): State<Arc<AdminFullState>>,
    Path(id): Path<String>,
) -> Result<Json<serde_json::Value>, ServerError> {
    // ADMIN OVERRIDE: bypasses user_id check — restricted to admin_guard
    let conv = hist_retrieve::get_conversation_admin(&state.inference.history_db, &id)
        .map_err(|e| ServerError::Db(e.to_string()))?
        .ok_or_else(|| ServerError::BadRequest("conversation not found".into()))?;

    let msgs = hist_retrieve::get_messages_admin(&state.inference.history_db, &id)
        .map_err(|e| ServerError::Db(e.to_string()))?;

    Ok(Json(json!({
        "id": conv.id,
        "user_id": conv.user_id,
        "title": conv.title,
        "created_at": conv.created_at,
        "updated_at": conv.updated_at,
        "message_count": conv.message_count,
        "messages": msgs.iter().map(|m| json!({
            "idx": m.idx,
            "role": m.role,
            "content": m.content,
            "timestamp": m.timestamp,
        })).collect::<Vec<_>>(),
    })))
}
