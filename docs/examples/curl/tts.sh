#!/usr/bin/env bash
# Incord TTS via /v1/audio/speech. Set INCORD_KEY and INCORD_URL.
# The server returns raw WAV bytes; this saves them to output.wav.

INCORD_KEY=${INCORD_KEY:?set INCORD_KEY env var}
INCORD_URL=${INCORD_URL:-https://api.example.com}

curl -sS "$INCORD_URL/v1/audio/speech" \
    -H "Authorization: Bearer $INCORD_KEY" \
    -H "Content-Type: application/json" \
    -d '{
        "model": "incord-tts",
        "input": "Hello, this is Incord text-to-speech.",
        "voice": "en-female-1",
        "response_format": "wav"
    }' \
    --output output.wav

echo "Audio saved to output.wav"
