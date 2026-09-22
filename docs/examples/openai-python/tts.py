"""Incord TTS via the openai Python SDK (/v1/audio/speech).

Install: pip install openai
Run:     INCORD_KEY=incord_xxx INCORD_URL=https://api.example.com python tts.py

Output is saved to output.wav.
"""
import os
import pathlib

from openai import OpenAI

client = OpenAI(
    api_key=os.environ["INCORD_KEY"],
    base_url=os.environ.get("INCORD_URL", "https://api.example.com") + "/v1",
)

response = client.audio.speech.create(
    model="incord-tts",
    voice="en-female-1",
    input="Hello, this is Incord text-to-speech.",
    response_format="wav",
)

output_path = pathlib.Path("output.wav")
response.stream_to_file(output_path)
print(f"Audio saved to {output_path}")
