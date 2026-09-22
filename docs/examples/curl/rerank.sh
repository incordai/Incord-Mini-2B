#!/usr/bin/env bash
# Incord reranking via /v1/rerank. Set INCORD_KEY and INCORD_URL.

INCORD_KEY=${INCORD_KEY:?set INCORD_KEY env var}
INCORD_URL=${INCORD_URL:-https://api.example.com}

curl -sS "$INCORD_URL/v1/rerank" \
    -H "Authorization: Bearer $INCORD_KEY" \
    -H "Content-Type: application/json" \
    -d '{
        "model": "incord-rerank",
        "query": "What is the capital of France?",
        "documents": [
            "Paris is the capital and most populous city of France.",
            "Berlin is the capital of Germany.",
            "Lyon is a major city in south-eastern France."
        ],
        "top_k": 2
    }'
