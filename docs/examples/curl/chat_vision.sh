#!/usr/bin/env bash
# Chat completion with image input.

INCORD_KEY=${INCORD_KEY:?set INCORD_KEY env var}
INCORD_URL=${INCORD_URL:-https://api.example.com}
IMG_PATH=${1:?usage: chat_vision.sh path/to/image.png}

B64=$(base64 -w0 "$IMG_PATH")
DATA_URL="data:image/png;base64,$B64"

curl -sS "$INCORD_URL/v1/chat/completions" \
    -H "Authorization: Bearer $INCORD_KEY" \
    -H "Content-Type: application/json" \
    -d "{
        \"model\": \"incord-mini-2b\",
        \"messages\": [{
            \"role\": \"user\",
            \"content\": [
                {\"type\": \"text\", \"text\": \"What's in this image?\"},
                {\"type\": \"image_url\", \"image_url\": {\"url\": \"$DATA_URL\"}}
            ]
        }],
        \"max_tokens\": 200
    }"
