#!/usr/bin/env bash
# QORA reranking via /v1/rerank. Set QORA_KEY and QORA_URL.

QORA_KEY=${QORA_KEY:?set QORA_KEY env var}
QORA_URL=${QORA_URL:-https://api.example.com}

curl -sS "$QORA_URL/v1/rerank" \
    -H "Authorization: Bearer $QORA_KEY" \
    -H "Content-Type: application/json" \
    -d '{
        "model": "qora-rerank",
        "query": "What is the capital of France?",
        "documents": [
            "Paris is the capital and most populous city of France.",
            "Berlin is the capital of Germany.",
            "Lyon is a major city in south-eastern France."
        ],
        "top_k": 2
    }'
