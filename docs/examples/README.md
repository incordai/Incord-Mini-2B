# QORA API Examples

Drop-in examples for using the QORA API.

## Setup

Set environment variables:

```bash
export QORA_KEY=qora_yourkey
export QORA_URL=https://api.example.com
```

## Examples

- **curl/** — shell script examples
  - `chat.sh` — non-streaming completion
  - `chat_stream.sh` — streaming completion (raw SSE)
  - `chat_vision.sh` — image input
- **openai-python/** — Python with the official `openai` SDK
  - `chat.py`, `chat_stream.py`, `chat_vision.py`
- **openai-node/** — Node with the official `openai` SDK
  - `chat.mjs`, `chat_stream.mjs`

All examples target `qora-4b`. The QORA API is wire-compatible with OpenAI's `/v1/chat/completions`, so the standard SDKs work unchanged — just point them at your QORA URL.
