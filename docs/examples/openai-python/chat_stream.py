"""Streaming chat completion via the openai Python SDK."""
import os

from openai import OpenAI

client = OpenAI(
    api_key=os.environ["INCORD_KEY"],
    base_url=os.environ.get("INCORD_URL", "https://api.example.com") + "/v1",
)

stream = client.chat.completions.create(
    model="incord-mini-2b",
    messages=[{"role": "user", "content": "Hello!"}],
    max_tokens=200,
    stream=True,
)
for chunk in stream:
    delta = chunk.choices[0].delta.content or ""
    print(delta, end="", flush=True)
print()
