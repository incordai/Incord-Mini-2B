#!/usr/bin/env bash
# QORA unified /v1/inference endpoint — one endpoint for all task types.
# Set QORA_KEY and QORA_URL. Uncomment the block for the task you want to test.

QORA_KEY=${QORA_KEY:?set QORA_KEY env var}
QORA_URL=${QORA_URL:-https://api.example.com}

# --- Chat ---
curl -sS "$QORA_URL/v1/inference" \
    -H "Authorization: Bearer $QORA_KEY" \
    -H "Content-Type: application/json" \
    -d '{
        "task": "chat",
        "model": "qora-llm",
        "messages": [{"role": "user", "content": "Explain entropy in one sentence."}],
        "max_tokens": 100
    }'

# --- TTS ---
# curl -sS "$QORA_URL/v1/inference" \
#     -H "Authorization: Bearer $QORA_KEY" \
#     -H "Content-Type: application/json" \
#     -d '{
#         "task": "tts",
#         "model": "qora-tts",
#         "input": "Hello from the unified endpoint.",
#         "voice": "en-female-1",
#         "format": "wav"
#     }' --output output.wav

# --- Embedding ---
# curl -sS "$QORA_URL/v1/inference" \
#     -H "Authorization: Bearer $QORA_KEY" \
#     -H "Content-Type: application/json" \
#     -d '{
#         "task": "embedding",
#         "model": "qora-embed",
#         "input": "Sentence to embed."
#     }'

# --- Rerank ---
# curl -sS "$QORA_URL/v1/inference" \
#     -H "Authorization: Bearer $QORA_KEY" \
#     -H "Content-Type: application/json" \
#     -d '{
#         "task": "rerank",
#         "model": "qora-rerank",
#         "query": "capital of France",
#         "documents": ["Paris is the capital of France.", "Berlin is the capital of Germany."]
#     }'
