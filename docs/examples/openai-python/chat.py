"""Incord chat completion via the openai Python SDK.

Install: pip install openai
Run:     INCORD_KEY=incord_xxx INCORD_URL=https://api.example.com python chat.py
"""
import os

from openai import OpenAI

client = OpenAI(
    api_key=os.environ["INCORD_KEY"],
    base_url=os.environ.get("INCORD_URL", "https://api.example.com") + "/v1",
)

resp = client.chat.completions.create(
    model="incord-mini-2b",
    messages=[{"role": "user", "content": "Hello!"}],
    max_tokens=200,
)
print(resp.choices[0].message.content)
