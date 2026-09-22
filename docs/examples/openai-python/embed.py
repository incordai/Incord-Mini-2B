"""Incord embeddings via the openai Python SDK (/v1/embeddings).

Install: pip install openai
Run:     INCORD_KEY=incord_xxx INCORD_URL=https://api.example.com python embed.py
"""
import os

from openai import OpenAI

client = OpenAI(
    api_key=os.environ["INCORD_KEY"],
    base_url=os.environ.get("INCORD_URL", "https://api.example.com") + "/v1",
)

response = client.embeddings.create(
    model="incord-embed",
    input=["Hello world", "Incord embedding example"],
)

for item in response.data:
    print(f"index={item.index}  dims={len(item.embedding)}  first5={item.embedding[:5]}")
