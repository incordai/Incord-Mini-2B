"""Chat completion with image input via the openai Python SDK."""
import base64
import os
import sys

from openai import OpenAI

client = OpenAI(
    api_key=os.environ["INCORD_KEY"],
    base_url=os.environ.get("INCORD_URL", "https://api.example.com") + "/v1",
)

img_path = sys.argv[1] if len(sys.argv) > 1 else "tests/fixtures/tiny.png"
with open(img_path, "rb") as f:
    b64 = base64.b64encode(f.read()).decode()

resp = client.chat.completions.create(
    model="incord-mini-2b",
    messages=[{
        "role": "user",
        "content": [
            {"type": "text", "text": "What's in this image?"},
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
        ],
    }],
    max_tokens=200,
)
print(resp.choices[0].message.content)
