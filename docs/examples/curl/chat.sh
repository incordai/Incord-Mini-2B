#!/usr/bin/env bash
# QORA chat completion via curl. Set QORA_KEY and QORA_URL.

QORA_KEY=${QORA_KEY:?set QORA_KEY env var}
QORA_URL=${QORA_URL:-https://api.example.com}

curl -sS "$QORA_URL/v1/chat/completions" \
    -H "Authorization: Bearer $QORA_KEY" \
    -H "Content-Type: application/json" \
    -d '{
        "model": "qora-4b",
        "messages": [{"role": "user", "content": "Hello!"}],
        "max_tokens": 200
    }'
