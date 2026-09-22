# QORA API Reference

QORA exposes an OpenAI-compatible HTTP API. If your code already targets `https://api.openai.com/v1`, change the base URL to your QORA endpoint and the API key — most clients work unchanged.

## Authentication

All `/v1/*` endpoints (except `/health` and `/metrics`) require a Bearer token:

```
Authorization: Bearer qora_<32-byte hex>
```

Keys are issued by an admin. Contact your operator if you don't have one.

## Endpoints

### `GET /health`

Liveness probe. Returns 200 with `{"status":"ok"}`. No auth.

### `GET /v1/models`

List available models.

```json
{
  "object": "list",
  "data": [
    {
      "id": "incord-mini-2b",
      "object": "model",
      "created": 1700000000,
      "owned_by": "incord"
    },
    {
      "id": "incord",
      "object": "model",
      "created": 1700000000,
      "owned_by": "incord"
    }
  ]
}
```

---

### `POST /v1/inference`

Unified inference endpoint. Dispatches to the correct engine based on the `task` field. All other endpoints are OpenAI-compatible aliases that map to this one internally.

#### Request

```json
{
  "task": "chat",
  "model": "qora-llm",
  "messages": [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Hello!"}
  ],
  "max_tokens": 200,
  "temperature": 0.7,
  "stream": false
}
```

The `task` field selects the engine:

| `task` value | Engine used | Required fields |
|---|---|---|
| `chat` | LLM chat engine | `messages` |
| `tts` | TTS engine | `input` (string) |
| `stt` | STT engine | `audio` (base64 WAV bytes) |
| `embedding` | Embedding engine | `input` (string or array of strings) |
| `rerank` | Rerank engine | `query`, `documents` |

All other task-specific fields follow the same schema as the dedicated endpoints below.

#### curl example

```bash
curl -sS "$QORA_URL/v1/inference" \
    -H "Authorization: Bearer $QORA_KEY" \
    -H "Content-Type: application/json" \
    -d '{
        "task": "chat",
        "model": "qora-llm",
        "messages": [{"role": "user", "content": "Hello!"}],
        "max_tokens": 100
    }'
```

---

### `POST /v1/chat/completions`

The main LLM inference endpoint. OpenAI-compatible.

#### Request

```json
{
  "model": "qora-llm",
  "messages": [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Hello!"}
  ],
  "max_tokens": 200,
  "temperature": 0.7,
  "top_p": 0.95,
  "stream": false
}
```

Vision (text + image):

```json
{
  "model": "qora-llm",
  "messages": [{
    "role": "user",
    "content": [
      {"type": "text", "text": "What's in this image?"},
      {"type": "image_url", "image_url": {"url": "data:image/png;base64,..."}}
    ]
  }],
  "max_tokens": 200
}
```

The image_url field accepts:
- `data:` URLs with base64-encoded bytes (recommended)
- `http://` or `https://` URLs (server fetches; ensure the URL is reachable from the server)

#### Response (non-streaming)

```json
{
  "id": "chatcmpl-...",
  "object": "chat.completion",
  "created": 1700000000,
  "model": "qora-llm",
  "choices": [{
    "index": 0,
    "message": {"role": "assistant", "content": "Hello! How can I help?"},
    "finish_reason": "stop"
  }],
  "usage": {
    "prompt_tokens": 12,
    "completion_tokens": 8,
    "total_tokens": 20
  }
}
```

#### Response (streaming, `stream: true`)

Server-Sent Events. Each event is a JSON `chat.completion.chunk`:

```
data: {"id":"chatcmpl-...","object":"chat.completion.chunk","created":1700000000,"model":"qora-llm","choices":[{"index":0,"delta":{"content":"Hello"},"finish_reason":null}]}

data: {"id":"chatcmpl-...","object":"chat.completion.chunk","created":1700000000,"model":"qora-llm","choices":[{"index":0,"delta":{"content":"!"},"finish_reason":null}]}

data: {"id":"chatcmpl-...","object":"chat.completion.chunk","created":1700000000,"model":"qora-llm","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}

data: [DONE]
```

#### Supported parameters

| Parameter | Type | Default | Notes |
|-----------|------|---------|-------|
| `model` | string | required | Must match a registered `chat` model id |
| `messages` | array | required | OpenAI message format |
| `stream` | bool | `false` | Stream tokens via SSE |
| `max_tokens` | int | 1024 | Max completion tokens |
| `temperature` | float | 0.7 | 0.0 = greedy |
| `top_p` | float | 0.95 | Nucleus sampling |
| `presence_penalty` | float | 1.5 | Reduce repetition |
| `frequency_penalty` | float | 1.0 | Reduce repetition |
| `n` | int | 1 | Always 1 in v1 |
| `stop` | string or array | none | Stop sequences |

QORA-specific extensions (additive to OpenAI schema; ignored by standard clients):

| Parameter | Type | Default | Notes |
|-----------|------|---------|-------|
| `reasoning_effort` | string | none | Reasoning effort level: `"low"` (512 tokens), `"medium"` (2048 tokens), `"high"` (8192 tokens). Standard OpenAI parameter for reasoning models. |
| `think` | bool | `true` | Enable extended chain-of-thought |
| `think_budget` | int | 1024 | Max thinking tokens (overrides `reasoning_effort` preset if specified) |
| `show_think` | bool | `false` | Include thinking tokens in output |

#### curl example

```bash
curl -sS "$QORA_URL/v1/chat/completions" \
    -H "Authorization: Bearer $QORA_KEY" \
    -H "Content-Type: application/json" \
    -d '{
        "model": "incord-mini-2b",
        "messages": [{"role": "user", "content": "Hello!"}],
        "max_tokens": 200
    }'
```

---

### `POST /v1/audio/speech`

Text-to-speech synthesis. OpenAI-compatible.

#### Request

```json
{
  "model": "qora-tts",
  "input": "Hello, this is QORA text-to-speech.",
  "voice": "en-female-1",
  "response_format": "wav"
}
```

| Parameter | Type | Default | Notes |
|-----------|------|---------|-------|
| `model` | string | required | Must match a registered `tts` model id |
| `input` | string | required | Text to synthesize |
| `voice` | string | engine default | Voice identifier (engine-specific) |
| `response_format` | string | `"wav"` | `"wav"` or `"mp3"` (v1 supports `"wav"`) |
| `language` | string | auto-detect | BCP 47 language tag |

#### Response

Raw audio bytes with `Content-Type: audio/wav` (or `audio/mpeg` for mp3).

#### curl example

```bash
curl -sS "$QORA_URL/v1/audio/speech" \
    -H "Authorization: Bearer $QORA_KEY" \
    -H "Content-Type: application/json" \
    -d '{
        "model": "qora-tts",
        "input": "Hello from QORA.",
        "voice": "en-female-1",
        "response_format": "wav"
    }' --output output.wav
```

---

### `POST /v1/audio/transcriptions`

Speech-to-text transcription. OpenAI-compatible multipart upload.

#### Request

`multipart/form-data` with fields:

| Field | Type | Notes |
|-------|------|-------|
| `model` | string | Must match a registered `stt` model id |
| `file` | binary | Audio file bytes (WAV recommended) |
| `language` | string | Optional BCP 47 language tag (e.g. `"en"`) |

#### Response

```json
{
  "text": "Hello, this is QORA speech-to-text.",
  "language": "en"
}
```

#### curl example

```bash
curl -sS "$QORA_URL/v1/audio/transcriptions" \
    -H "Authorization: Bearer $QORA_KEY" \
    -F "model=qora-stt" \
    -F "file=@recording.wav" \
    -F "language=en"
```

---

### `POST /v1/embeddings`

Sentence / document embedding. OpenAI-compatible.

#### Request

```json
{
  "model": "qora-embed",
  "input": "Text to embed"
}
```

`input` can be a single string or an array of strings.

| Parameter | Type | Default | Notes |
|-----------|------|---------|-------|
| `model` | string | required | Must match a registered `embedding` model id |
| `input` | string or array | required | Text(s) to embed |
| `encoding_format` | string | `"float"` | Only `"float"` supported in v1 |

#### Response

```json
{
  "object": "list",
  "data": [
    {
      "object": "embedding",
      "embedding": [0.012, -0.034, ...],
      "index": 0
    }
  ],
  "model": "qora-embed",
  "usage": {"prompt_tokens": 0, "total_tokens": 0}
}
```

Embedding dimensionality is 2048 for the default multilingual embedding model.

#### curl example

```bash
curl -sS "$QORA_URL/v1/embeddings" \
    -H "Authorization: Bearer $QORA_KEY" \
    -H "Content-Type: application/json" \
    -d '{
        "model": "qora-embed",
        "input": ["Hello world", "QORA embedding example"]
    }'
```

---

### `POST /v1/rerank`

Cross-encoder reranking. QORA custom endpoint (not in OpenAI spec).

#### Request

```json
{
  "model": "qora-rerank",
  "query": "What is the capital of France?",
  "documents": [
    "Paris is the capital and most populous city of France.",
    "Berlin is the capital of Germany.",
    "Lyon is a major city in south-eastern France."
  ],
  "top_k": 2
}
```

| Parameter | Type | Default | Notes |
|-----------|------|---------|-------|
| `model` | string | required | Must match a registered `rerank` model id |
| `query` | string | required | The query to rank documents against |
| `documents` | array | required | Non-empty array of strings to rerank |
| `top_k` | int | all | Return only the top-k results |

#### Response

```json
{
  "results": [
    {"index": 0, "score": 0.97, "document": "Paris is the capital and most populous city of France."},
    {"index": 2, "score": 0.43, "document": "Lyon is a major city in south-eastern France."}
  ],
  "model": "qora-rerank"
}
```

Results are sorted by score descending.

#### curl example

```bash
curl -sS "$QORA_URL/v1/rerank" \
    -H "Authorization: Bearer $QORA_KEY" \
    -H "Content-Type: application/json" \
    -d '{
        "model": "qora-rerank",
        "query": "capital of France",
        "documents": [
            "Paris is the capital of France.",
            "Berlin is the capital of Germany."
        ],
        "top_k": 1
    }'
```

---

## Errors

All errors return JSON:

```json
{
  "error": {
    "type": "rate_limit_error",
    "message": "rate limit exceeded",
    "request_id": "..."
  }
}
```

| Status | Type | Meaning |
|--------|------|---------|
| 400 | `invalid_request_error` | Malformed request, unknown model, bad image URL |
| 401 | `authentication_error` | Missing or invalid API key |
| 408 | `timeout_error` | Request timed out |
| 429 | `rate_limit_error` | Per-key RPM/TPM exceeded; check `Retry-After` header |
| 500 | `inference_error` / `internal_error` | Server-side failure |
| 503 | `server_error` | Queue full; check `Retry-After` header |

The `request_id` field is also returned in the `x-request-id` response header — include it when reporting issues.

## Rate limits

Each API key has per-minute limits:
- `rpm` — requests per minute
- `tpm` — tokens per minute (input + output)

Defaults are configurable per key by the operator. Hitting either returns 429 with `Retry-After: <seconds>`.

## Drop-in OpenAI compatibility

QORA implements the same wire format as OpenAI's v1 API for chat completions, audio speech, audio transcriptions, and embeddings. The `openai` Python and Node SDKs work unchanged — just point them at your QORA URL.

See examples under `docs/examples/`.
