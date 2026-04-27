//! Smoke test verifying LlmEngine implements ChatEngine over mock weights.

use std::sync::Arc;
use qor4b::gemv::mock::tiny_weights;
use qor4b::tokenizer::QoraTokenizer;
use qora_server::engine::chat::{ChatEngine, ChatMessage, ChatRequest};
use qora_server::engines::llm::LlmEngine;
use tokio::sync::mpsc;

/// Build a minimal tokenizer from the mock model file path used by other test suites.
fn test_tokenizer() -> Arc<QoraTokenizer> {
    let candidates = [
        concat!(env!("CARGO_MANIFEST_DIR"), "/../qora-llm/model/tokenizer.json"),
        concat!(env!("CARGO_MANIFEST_DIR"), "/../qora-llm/tests/fixtures/tokenizer.json"),
        concat!(env!("CARGO_MANIFEST_DIR"), "/../../model/tokenizer.json"),
    ];
    for path in candidates {
        let p = std::path::Path::new(path);
        if p.exists() {
            if let Ok(t) = QoraTokenizer::from_file(p) {
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
    let engine = LlmEngine::new("qora-4b-mock", weights, tokenizer, "Mock LLM for tests");

    use qora_server::engine::ModelEngine;
    assert_eq!(engine.id(), "qora-4b-mock");
    assert_eq!(engine.task(), qora_server::engine::Task::Chat);

    let (tx, mut rx) = mpsc::channel(64);
    let req = ChatRequest {
        model: "qora-4b-mock".to_string(),
        messages: vec![ChatMessage { role: "user".to_string(), content: "hi".to_string() }],
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
    let handle = tokio::spawn(async move {
        engine_clone.chat(req, tx).await
    });

    let mut tokens = 0;
    let mut finished = false;
    while let Some(ev) = rx.recv().await {
        match ev {
            qora_server::engine::chat::ChatEvent::Token(_, _) => tokens += 1,
            qora_server::engine::chat::ChatEvent::Finish(_) => { finished = true; break; }
            qora_server::engine::chat::ChatEvent::Error(e) => panic!("unexpected error: {e}"),
        }
    }
    handle.await.unwrap().unwrap();

    assert!(tokens > 0, "no tokens emitted");
    assert!(finished);
}
