//! Thin HTTP client for the internal brain RAG service.
//!
//! Brain is a separate process under `C:/Projects/Chat/brain/` that owns
//! its own embedding model, knowledge graph, and ingestion daemons. The
//! chat server reaches it over HTTP — no direct linking to brain's crates.
//!
//! End-users interact with brain only through the chat completions API
//! (`use_brain: true`); brain's port is not exposed publicly. See the
//! project memory `project_brain_via_chat_server.md` for the rationale.

use serde::{Deserialize, Serialize};
use std::time::Duration;

/// Task instruction sent on every `BrainClient::context` call. Qwen3-Embedding
/// is instruct-tuned: the brain's `/v1/context` handler prepends this as
/// `Instruct: {task}\nQuery:{user_text}` before tokenisation, which shapes
/// the output vector toward the retrieval task we're performing.
///
/// This is purely a *retrieval-shaping* instruction — it does NOT tell the
/// embedder to drop or ignore any part of the user's query. Numbers, dates,
/// tickers, and named entities the user mentions are all signal worth
/// matching on; the brain corpus contains nodes for "BTC at $80k", "Fed
/// meeting Nov 5", "MSFT earnings beat", etc. and the embedder should use
/// every token the user wrote.
pub const QORA_BRAIN_INSTRUCT: &str =
    "Given a user query about markets, news, prices, current events, dates, \
     economic releases, or general knowledge, retrieve the most relevant \
     passages from a heterogeneous knowledge graph that mixes market-data \
     snapshots, news articles, and macro/on-chain events. Match on the full \
     query — topic, named entities, tickers, dates, and numeric values are \
     all valid signals. Prefer items that are recent, on-topic, and from \
     authoritative sources.";

#[derive(Debug, Clone)]
pub struct BrainClient {
    base_url: String,
    http: reqwest::Client,
    pub default_top_k: usize,
}

impl BrainClient {
    pub fn new(
        base_url: impl Into<String>,
        default_top_k: usize,
        timeout_secs: u64,
    ) -> anyhow::Result<Self> {
        let http = reqwest::Client::builder()
            .timeout(Duration::from_secs(timeout_secs))
            .build()?;
        Ok(Self {
            base_url: base_url.into(),
            http,
            default_top_k,
        })
    }

    /// POST `/v1/context` and return the parsed response.
    /// Rerank is intentionally off — it's slow on CPU (~80 s for top-30) and
    /// the chat LLM will reason across all top_k nodes regardless of order.
    /// The `instruct` field is always populated with [`QORA_BRAIN_INSTRUCT`]
    /// so retrieval is keyed on topic, not on numeric claims in the query.
    pub async fn context(&self, query: &str, top_k: usize) -> anyhow::Result<ContextResponse> {
        let req = ContextRequestBody {
            query: query.to_string(),
            top_k,
            rerank: false,
            filters: serde_json::json!({}),
            instruct: Some(QORA_BRAIN_INSTRUCT.to_string()),
        };
        let url = format!("{}/v1/context", self.base_url);
        let started = std::time::Instant::now();
        tracing::info!(url = %url, query = %query, top_k, "BrainClient::context HTTP POST start");
        let resp = self.http.post(&url).json(&req).send().await?;
        let status = resp.status();
        let http_ms = started.elapsed().as_millis();
        tracing::info!(http_ms, status = %status, "BrainClient::context HTTP response received");
        if !status.is_success() {
            let body = resp.text().await.unwrap_or_default();
            tracing::warn!(status = %status, body_preview = %body.chars().take(200).collect::<String>(), "BrainClient::context non-success status");
            anyhow::bail!("brain /v1/context returned {status}: {body}");
        }
        let parsed: ContextResponse = resp.json().await?;
        tracing::info!(
            total_ms = started.elapsed().as_millis(),
            n_nodes = parsed.nodes.len(),
            "BrainClient::context parsed response"
        );
        Ok(parsed)
    }
}

#[derive(Serialize)]
struct ContextRequestBody {
    query: String,
    top_k: usize,
    rerank: bool,
    filters: serde_json::Value,
    /// Task instruction for the Qwen3-Embedding model on the brain side.
    /// `None` lets brain fall back to its default; we always send our own.
    #[serde(skip_serializing_if = "Option::is_none")]
    instruct: Option<String>,
}

#[derive(Deserialize, Serialize, Debug, Clone)]
pub struct ContextResponse {
    pub query: String,
    pub nodes: Vec<NodeView>,
    pub confidence: ConfidenceSignal,
}

#[derive(Deserialize, Serialize, Debug, Clone)]
pub struct NodeView {
    pub id: String,
    pub content: String,
    pub source: String,
    pub timestamp: String,
    pub embed_score: f32,
    pub rerank_score: Option<f32>,
}

#[derive(Deserialize, Serialize, Debug, Clone)]
pub struct ConfidenceSignal {
    pub sufficient: bool,
    pub reason: String,
}
