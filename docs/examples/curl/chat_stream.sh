#!/usr/bin/env bash
# Streaming chat completion via curl. Outputs raw SSE — pipe through grep/jq for parsing.

INCORD_KEY=${INCORD_KEY:?set INCORD_KEY env var}
INCORD_URL=${INCORD_URL:-https://api.example.com}

curl -sN "$INCORD_URL/v1/chat/completions" \
    -H "Authorization: Bearer $INCORD_KEY" \
    -H "Content-Type: application/json" \
    -d '{
        "model": "incord-mini-2b",
        "messages": [{"role": "user", "content": "Hello!"}],
        "max_tokens": 200,
        "stream": true
    }'
