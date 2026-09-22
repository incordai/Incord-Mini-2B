#!/usr/bin/env bash
# Incord chat completion via curl. Set INCORD_KEY and INCORD_URL.

INCORD_KEY=${INCORD_KEY:?set INCORD_KEY env var}
INCORD_URL=${INCORD_URL:-https://api.example.com}

curl -sS "$INCORD_URL/v1/chat/completions" \
    -H "Authorization: Bearer $INCORD_KEY" \
    -H "Content-Type: application/json" \
    -d '{
        "model": "incord-mini-2b",
        "messages": [{"role": "user", "content": "Hello!"}],
        "max_tokens": 200
    }'
