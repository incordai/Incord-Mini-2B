#!/usr/bin/env bash
# QORA TTS via /v1/audio/speech. Set QORA_KEY and QORA_URL.
# The server returns raw WAV bytes; this saves them to output.wav.

QORA_KEY=${QORA_KEY:?set QORA_KEY env var}
QORA_URL=${QORA_URL:-https://api.example.com}

curl -sS "$QORA_URL/v1/audio/speech" \
    -H "Authorization: Bearer $QORA_KEY" \
    -H "Content-Type: application/json" \
    -d '{
        "model": "qora-tts",
        "input": "Hello, this is QORA text-to-speech.",
        "voice": "en-female-1",
        "response_format": "wav"
    }' \
    --output output.wav

echo "Audio saved to output.wav"
