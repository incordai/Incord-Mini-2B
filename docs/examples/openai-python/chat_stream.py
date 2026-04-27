"""Streaming chat completion via the openai Python SDK."""
import os

from openai import OpenAI

client = OpenAI(
    api_key=os.environ["QORA_KEY"],
    base_url=os.environ.get("QORA_URL", "https://api.example.com") + "/v1",
)

stream = client.chat.completions.create(
    model="qora-4b",
    messages=[{"role": "user", "content": "Hello!"}],
    max_tokens=200,
    stream=True,
)
for chunk in stream:
    delta = chunk.choices[0].delta.content or ""
    print(delta, end="", flush=True)
print()
