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
      "id": "qora-4b",
      "object": "model",
      "created": 1700000000,
      "owned_by": "qora"
    }
  ]
}
```

### `POST /v1/chat/completions`

The main inference endpoint. OpenAI-compatible.

#### Request

```json
{
  "model": "qora-4b",
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
  "model": "qora-4b",
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
  "model": "qora-4b",
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
data: {"id":"chatcmpl-...","object":"chat.completion.chunk","created":1700000000,"model":"qora-4b","choices":[{"index":0,"delta":{"content":"Hello"},"finish_reason":null}]}

data: {"id":"chatcmpl-...","object":"chat.completion.chunk","created":1700000000,"model":"qora-4b","choices":[{"index":0,"delta":{"content":"!"},"finish_reason":null}]}

data: {"id":"chatcmpl-...","object":"chat.completion.chunk","created":1700000000,"model":"qora-4b","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}

data: [DONE]
```

#### Supported parameters

| Parameter | Type | Default | Notes |
|-----------|------|---------|-------|
| `model` | string | required | Must be `"qora-4b"` |
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
| `think` | bool | `true` | Enable extended chain-of-thought |
| `think_budget` | int | 1024 | Max thinking tokens |
| `show_think` | bool | `false` | Include thinking tokens in output |

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

QORA implements the same wire format as OpenAI's `/v1/chat/completions`. The `openai` Python and Node SDKs work unchanged — just point them at your QORA URL.

See examples under `docs/examples/`.
