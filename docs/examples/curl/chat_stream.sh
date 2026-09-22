#!/usr/bin/env bash
# Streaming chat completion via curl. Outputs raw SSE — pipe through grep/jq for parsing.

QORA_KEY=${QORA_KEY:?set QORA_KEY env var}
QORA_URL=${QORA_URL:-https://api.example.com}

curl -sN "$QORA_URL/v1/chat/completions" \
    -H "Authorization: Bearer $QORA_KEY" \
    -H "Content-Type: application/json" \
    -d '{
        "model": "incord-mini-2b",
        "messages": [{"role": "user", "content": "Hello!"}],
        "max_tokens": 200,
        "stream": true
    }'
