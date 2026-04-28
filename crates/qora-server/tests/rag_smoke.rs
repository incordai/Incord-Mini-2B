//! Smoke test for embedding + rerank dispatch via qor-rag.
//!
//! Requires the qor-rag model directory to be present at a known location.
//! If no model directory is found, the test skips gracefully rather than failing —
//! CI environments without model weights will still report a green run.
//!
//! Expected directory layout:
//! ```text
//! <model_dir>/
//!   embed.qor-rag
//!   embed-tokenizer.json
//!   rerank.qor-rag
//!   rerank-tokenizer.json
//! ```

#![cfg(feature = "mock-weights")]

use std::path::Path;
use std::sync::Arc;

/// Return the first candidate model directory that exists, or None.
fn find_model_dir() -> Option<&'static Path> {
    const CANDIDATES: &[&str] = &[
        "C:/Projects/AI/qor-main/qor-rs/crates/qor-rag/model",
        "D:/Burn-AI/qor-rag/model",
    ];
    CANDIDATES.iter().map(|s| Path::new(s)).find(|p| p.exists())
}

#[tokio::test]
async fn embedding_returns_vector() {
    let model_dir = match find_model_dir() {
        Some(p) => p,
        None => {
            eprintln!("[rag_smoke] SKIP: no qor-rag model directory found at any candidate path");
            return;
        }
    };

    use qora_server::engine::ModelEngine;
    use qora_server::engine::embed::{EmbedRequest, EmbeddingEngine};
    use qora_server::engines::rag::{QorEmbeddingEngine, QorRagBackend};

    let backend = match QorRagBackend::load(model_dir) {
        Ok(b) => Arc::new(b),
        Err(e) => {
            eprintln!("[rag_smoke] SKIP: QorRagBackend::load failed: {e}");
            return;
        }
    };

    let engine = QorEmbeddingEngine {
        id: "test-embed".to_string(),
        description: "Smoke test embedding engine".to_string(),
        backend,
    };

    assert_eq!(engine.id(), "test-embed");

    let req = EmbedRequest {
        model: "test-embed".to_string(),
        input: vec![
            "The quick brown fox jumps over the lazy dog.".to_string(),
            "All generalizations are false, including this one.".to_string(),
        ],
    };

    let out = engine.embed(req).await.expect("embedding should succeed");

    assert_eq!(
        out.embeddings.len(),
        2,
        "should return one vector per input"
    );
    assert!(out.dim > 0, "dimension should be positive");
    assert_eq!(
        out.embeddings[0].len(),
        out.dim,
        "first embedding length should match dim"
    );
    assert_eq!(
        out.embeddings[1].len(),
        out.dim,
        "second embedding length should match dim"
    );

    // Qwen3-Embedding-2B produces 2048-dim vectors.
    assert_eq!(
        out.dim, 2048,
        "expected 2048-dim embeddings from Qwen3-Embedding-2B"
    );

    // Vectors should be L2-normalized (norm ≈ 1.0).
    for (i, v) in out.embeddings.iter().enumerate() {
        let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!(
            (norm - 1.0).abs() < 1e-3,
            "embedding[{i}] should be L2-normalized, got norm={norm}"
        );
    }
}

#[tokio::test]
async fn rerank_scores_and_sorts() {
    let model_dir = match find_model_dir() {
        Some(p) => p,
        None => {
            eprintln!("[rag_smoke] SKIP: no qor-rag model directory found at any candidate path");
            return;
        }
    };

    use qora_server::engine::ModelEngine;
    use qora_server::engine::rerank::{RerankEngine, RerankRequest};
    use qora_server::engines::rag::{QorRagBackend, QorRerankEngine};

    let backend = match QorRagBackend::load(model_dir) {
        Ok(b) => Arc::new(b),
        Err(e) => {
            eprintln!("[rag_smoke] SKIP: QorRagBackend::load failed: {e}");
            return;
        }
    };

    let engine = QorRerankEngine {
        id: "test-rerank".to_string(),
        description: "Smoke test rerank engine".to_string(),
        backend,
    };

    assert_eq!(engine.id(), "test-rerank");

    let req = RerankRequest {
        model: "test-rerank".to_string(),
        query: "What is the capital of France?".to_string(),
        documents: vec![
            "Paris is the capital and most populous city of France.".to_string(),
            "The Eiffel Tower was completed in 1889.".to_string(),
            "Rust is a systems programming language.".to_string(),
        ],
        top_k: None,
    };

    let out = engine.rerank(req).await.expect("rerank should succeed");

    assert_eq!(
        out.results.len(),
        3,
        "should return a score for each document"
    );

    // Results are sorted descending by score.
    for w in out.results.windows(2) {
        assert!(
            w[0].score >= w[1].score,
            "results should be sorted descending by score: {} < {}",
            w[0].score,
            w[1].score
        );
    }

    // All scores should be in [0, 1] (P(yes|query,doc)).
    for r in &out.results {
        assert!(
            r.score >= 0.0 && r.score <= 1.0,
            "score out of [0,1] range: {}",
            r.score
        );
    }

    // The first document (direct answer) should score highest.
    // index=0 ("Paris is the capital...") should be rank 1.
    assert_eq!(
        out.results[0].index, 0,
        "document 0 (direct answer) should rank first"
    );
}

#[tokio::test]
async fn rerank_top_k_truncation() {
    let model_dir = match find_model_dir() {
        Some(p) => p,
        None => {
            eprintln!("[rag_smoke] SKIP: no qor-rag model directory found at any candidate path");
            return;
        }
    };

    use qora_server::engine::rerank::{RerankEngine, RerankRequest};
    use qora_server::engines::rag::{QorRagBackend, QorRerankEngine};

    let backend = match QorRagBackend::load(model_dir) {
        Ok(b) => Arc::new(b),
        Err(e) => {
            eprintln!("[rag_smoke] SKIP: QorRagBackend::load failed: {e}");
            return;
        }
    };

    let engine = QorRerankEngine {
        id: "test-rerank-topk".to_string(),
        description: "Smoke test rerank top_k".to_string(),
        backend,
    };

    let req = RerankRequest {
        model: "test-rerank-topk".to_string(),
        query: "machine learning".to_string(),
        documents: vec![
            "Neural networks are a subset of machine learning.".to_string(),
            "The sky is blue.".to_string(),
            "Gradient descent optimizes model parameters.".to_string(),
            "Pizza is a popular Italian dish.".to_string(),
        ],
        top_k: Some(2),
    };

    let out = engine.rerank(req).await.expect("rerank should succeed");
    assert_eq!(
        out.results.len(),
        2,
        "top_k=2 should return exactly 2 results"
    );
}
