# Incord API Examples

Drop-in examples for using the Incord API.

## Setup

Set environment variables:

```bash
export INCORD_KEY=incord_yourkey
export INCORD_URL=https://api.example.com
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

All examples target `incord-mini-2b` (also aliased as `incord`). The Incord API is wire-compatible with OpenAI's `/v1/chat/completions`, so standard SDKs work unchanged — just point them at your Incord URL.
