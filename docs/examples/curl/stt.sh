#!/usr/bin/env bash
# QORA STT via /v1/audio/transcriptions (multipart). Set QORA_KEY and QORA_URL.
# Pass the path to a WAV file as the first argument:  ./stt.sh recording.wav

QORA_KEY=${QORA_KEY:?set QORA_KEY env var}
QORA_URL=${QORA_URL:-https://api.example.com}
AUDIO_FILE=${1:?usage: stt.sh <audio.wav>}

curl -sS "$QORA_URL/v1/audio/transcriptions" \
    -H "Authorization: Bearer $QORA_KEY" \
    -F "model=qora-stt" \
    -F "file=@$AUDIO_FILE" \
    -F "language=en"
