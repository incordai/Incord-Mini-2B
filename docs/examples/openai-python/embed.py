"""QORA embeddings via the openai Python SDK (/v1/embeddings).

Install: pip install openai
Run:     QORA_KEY=qora_xxx QORA_URL=https://api.example.com python embed.py
"""
import os

from openai import OpenAI

client = OpenAI(
    api_key=os.environ["QORA_KEY"],
    base_url=os.environ.get("QORA_URL", "https://api.example.com") + "/v1",
)

response = client.embeddings.create(
    model="qora-embed",
    input=["Hello world", "QORA embedding example"],
)

for item in response.data:
    print(f"index={item.index}  dims={len(item.embedding)}  first5={item.embedding[:5]}")
