#!/usr/bin/env bash
# Incord embeddings via /v1/embeddings. Set INCORD_KEY and INCORD_URL.

INCORD_KEY=${INCORD_KEY:?set INCORD_KEY env var}
INCORD_URL=${INCORD_URL:-https://api.example.com}

curl -sS "$INCORD_URL/v1/embeddings" \
    -H "Authorization: Bearer $INCORD_KEY" \
    -H "Content-Type: application/json" \
    -d '{
        "model": "incord-embed",
        "input": ["Hello world", "Incord embedding example"]
    }'
