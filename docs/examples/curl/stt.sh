#!/usr/bin/env bash
# Incord STT via /v1/audio/transcriptions (multipart). Set INCORD_KEY and INCORD_URL.
# Pass the path to a WAV file as the first argument:  ./stt.sh recording.wav

INCORD_KEY=${INCORD_KEY:?set INCORD_KEY env var}
INCORD_URL=${INCORD_URL:-https://api.example.com}
AUDIO_FILE=${1:?usage: stt.sh <audio.wav>}

curl -sS "$INCORD_URL/v1/audio/transcriptions" \
    -H "Authorization: Bearer $INCORD_KEY" \
    -F "model=incord-stt" \
    -F "file=@$AUDIO_FILE" \
    -F "language=en"
