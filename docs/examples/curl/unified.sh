#!/usr/bin/env bash
# Incord unified /v1/inference endpoint — one endpoint for all task types.
# Set INCORD_KEY and INCORD_URL. Uncomment the block for the task you want to test.

INCORD_KEY=${INCORD_KEY:?set INCORD_KEY env var}
INCORD_URL=${INCORD_URL:-https://api.example.com}

# --- Chat ---
curl -sS "$INCORD_URL/v1/inference" \
    -H "Authorization: Bearer $INCORD_KEY" \
    -H "Content-Type: application/json" \
    -d '{
        "task": "chat",
        "model": "incord-llm",
        "messages": [{"role": "user", "content": "Explain entropy in one sentence."}],
        "max_tokens": 100
    }'

# --- TTS ---
# curl -sS "$INCORD_URL/v1/inference" \
#     -H "Authorization: Bearer $INCORD_KEY" \
#     -H "Content-Type: application/json" \
#     -d '{
#         "task": "tts",
#         "model": "incord-tts",
#         "input": "Hello from the unified endpoint.",
#         "voice": "en-female-1",
#         "format": "wav"
#     }' --output output.wav

# --- Embedding ---
# curl -sS "$INCORD_URL/v1/inference" \
#     -H "Authorization: Bearer $INCORD_KEY" \
#     -H "Content-Type: application/json" \
#     -d '{
#         "task": "embedding",
#         "model": "incord-embed",
#         "input": "Sentence to embed."
#     }'

# --- Rerank ---
# curl -sS "$INCORD_URL/v1/inference" \
#     -H "Authorization: Bearer $INCORD_KEY" \
#     -H "Content-Type: application/json" \
#     -d '{
#         "task": "rerank",
#         "model": "incord-rerank",
#         "query": "capital of France",
#         "documents": ["Paris is the capital of France.", "Berlin is the capital of Germany."]
#     }'
