"""QORA STT via the openai Python SDK (/v1/audio/transcriptions).

Install: pip install openai
Run:     QORA_KEY=qora_xxx QORA_URL=https://api.example.com python stt.py recording.wav
"""
import os
import sys

from openai import OpenAI

client = OpenAI(
    api_key=os.environ["QORA_KEY"],
    base_url=os.environ.get("QORA_URL", "https://api.example.com") + "/v1",
)

audio_file = sys.argv[1] if len(sys.argv) > 1 else "recording.wav"

with open(audio_file, "rb") as f:
    transcript = client.audio.transcriptions.create(
        model="qora-stt",
        file=f,
        language="en",
    )

print(transcript.text)
