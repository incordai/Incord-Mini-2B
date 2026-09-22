"""Incord STT via the openai Python SDK (/v1/audio/transcriptions).

Install: pip install openai
Run:     INCORD_KEY=incord_xxx INCORD_URL=https://api.example.com python stt.py recording.wav
"""
import os
import sys

from openai import OpenAI

client = OpenAI(
    api_key=os.environ["INCORD_KEY"],
    base_url=os.environ.get("INCORD_URL", "https://api.example.com") + "/v1",
)

audio_file = sys.argv[1] if len(sys.argv) > 1 else "recording.wav"

with open(audio_file, "rb") as f:
    transcript = client.audio.transcriptions.create(
        model="incord-stt",
        file=f,
        language="en",
    )

print(transcript.text)
