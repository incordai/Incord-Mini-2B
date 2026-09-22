//! Smoke test verifying LlmEngine implements ChatEngine over mock weights.

use incord::gemv::mock::tiny_weights;
use incord::tokenizer::IncordTokenizer;
use incord_server::engine::chat::{ChatEngine, ChatMessage, ChatRequest};
use incord_server::engines::llm::LlmEngine;
use std::sync::Arc;
use tokio::sync::mpsc;

/// Build a minimal tokenizer from the mock model file path used by other test suites.
fn test_tokenizer() -> Arc<IncordTokenizer> {
    let candidates = [
        concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../incord-llm/model/tokenizer.json"
        ),
        concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../incord-llm/tests/fixtures/tokenizer.json"
        ),
        concat!(env!("CARGO_MANIFEST_DIR"), "/../../model/tokenizer.json"),
    ];
    for path in candidates {
        let p = std::path::Path::new(path);
        if p.exists() {
            if let Ok(t) = IncordTokenizer::from_file(p) {
                return Arc::new(t);
            }
        }
    }
    panic!("no tokenizer.json found");
}

#[tokio::test]
async fn llm_engine_implements_chat_engine() {
    let weights = Arc::new(tiny_weights());
    let tokenizer = test_tokenizer();
    let engine = LlmEngine::new("incord-mini-2b-mock", weights, tokenizer, "Mock LLM for tests");

    use incord_server::engine::ModelEngine;
    assert_eq!(engine.id(), "incord-mini-2b-mock");
    assert_eq!(engine.task(), incord_server::engine::Task::Chat);

    let (tx, mut rx) = mpsc::channel(64);
    let req = ChatRequest {
        model: "incord-mini-2b-mock".to_string(),
        messages: vec![ChatMessage {
            role: "user".to_string(),
            content: "hi".to_string(),
        }],
        max_tokens: Some(2),
        temperature: Some(0.0),
        top_p: Some(0.95),
        stream: false,
        think: Some(false),
        think_budget: None,
        vision_embeds: None,
    };

    // Run engine.chat — should produce at least one token and a Finish event
    let engine_arc = std::sync::Arc::new(engine);
    let engine_clone = engine_arc.clone();
    let handle = tokio::spawn(async move { engine_clone.chat(req, tx).await });

    let mut tokens = 0;
    let mut finished = false;
    while let Some(ev) = rx.recv().await {
        match ev {
            incord_server::engine::chat::ChatEvent::Token(_, _) => tokens += 1,
            incord_server::engine::chat::ChatEvent::Finish(_) => {
                finished = true;
                break;
            }
            incord_server::engine::chat::ChatEvent::Error(e) => panic!("unexpected error: {e}"),
        }
    }
    handle.await.unwrap().unwrap();

    assert!(tokens > 0, "no tokens emitted");
    assert!(finished);
}
