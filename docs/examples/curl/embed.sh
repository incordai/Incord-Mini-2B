#!/usr/bin/env bash
# QORA embeddings via /v1/embeddings. Set QORA_KEY and QORA_URL.

QORA_KEY=${QORA_KEY:?set QORA_KEY env var}
QORA_URL=${QORA_URL:-https://api.example.com}

curl -sS "$QORA_URL/v1/embeddings" \
    -H "Authorization: Bearer $QORA_KEY" \
    -H "Content-Type: application/json" \
    -d '{
        "model": "qora-embed",
        "input": ["Hello world", "QORA embedding example"]
    }'
