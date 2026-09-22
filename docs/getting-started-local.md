# Getting Started — Local End-to-End Test

This guide walks through running the full stack locally: incord-server backend + admin UI + chat UI, with API key creation and a test chat.

## 1. Build and start the backend

In one terminal:

```bash
cd D:\Burn-AI\Incord-Mini-2B
cargo build --release --workspace --bin incord-server
```

Set required environment variables:

```powershell
$env:INCORD_SERVER_SECRET = "<GENERATE_RANDOM_SECRET_HEX>"
$env:INCORD_ADMIN_TOKEN   = "dev-admin-token"
```

Edit (or create) a local config `incord-server.toml`:

```toml
[server]
bind = "127.0.0.1:8080"

[scheduler]
slot_count = 8
max_queue = 100
max_context = 8192
request_timeout_secs = 90

[auth]
admin_token_env = "INCORD_ADMIN_TOKEN"
db_path = "./local-auth.redb"

[limits.default]
rpm = 60
tpm = 60000

[observability]
prometheus_bind = "127.0.0.1:9090"
log_level = "info"
log_format = "json"

[[models]]
id = "incord-mini-2b"
kind = "chat"
path = "D:/Burn-AI/Incord-Mini-2B/model/model.incord"
tokenizer = "D:/Burn-AI/Incord-Mini-2B/model/tokenizer.json"
description = "Local LLM for testing"
```

(Adjust paths to wherever your `model.incord` and `tokenizer.json` actually live.)

Start the server:

```powershell
.\target\release\incord-server.exe incord-server.toml
```

You should see logs about loading the model and binding to 127.0.0.1:8080.

Verify with another terminal:

```bash
curl http://localhost:8080/health
# {"status":"ok"}

curl http://localhost:8080/v1/models
# {"object":"list","data":[{"id":"incord-mini-2b",...}]}
```

## 2. Start the admin UI

In a second terminal:

```bash
cd C:\Projects\Chat\fruntend\admin
npm install   # first time only
npm run dev
```

Open http://localhost:5173. You'll see the login screen.

- Backend URL: `http://localhost:8080`
- Admin token: `dev-admin-token` (matches what you set in `INCORD_ADMIN_TOKEN`)

Click LOGIN. The dashboard loads.

## 3. Create a user API key

In the admin UI:

1. Go to the **Keys** tab.
2. Click **Create Key**.
3. Fill in: user_id = `test-user`, name = `chat-test`, rpm_limit = 60, tpm_limit = 60000.
4. Submit.

A modal appears with the raw key (looks like `incord_<64 hex chars>`). **Copy it now** — it's never shown again. We'll need it in the next step.

## 4. Start the chat UI

In a third terminal:

```bash
cd C:\Projects\Chat\fruntend\chat
npm install   # first time only
npm run dev
```

Open http://localhost:5174.

In the chat UI:

1. Open Settings → Preferences (or the dev panel).
2. Set **Backend URL** to `http://localhost:8080` (if it isn't already the default).
3. Set **API Key** to the `incord_…` key you copied from the admin UI.
4. Save.

## 5. Send a test message

Type a message in the chat input and send. You should see the LLM respond.

If it errors:
- 401: API key is wrong. Re-paste from admin.
- Connection refused: backend isn't running, or the URL is wrong.
- Unknown model: the chat UI is sending a model id that isn't in the registry. Check the model dropdown — set it to `incord-mini-2b`.

## 6. Test other endpoints (optional)

From a terminal, with the user key from step 3:

```bash
KEY=incord_<your-key>

# Embedding
curl -X POST http://localhost:8080/v1/embeddings \
  -H "Authorization: Bearer $KEY" \
  -H "Content-Type: application/json" \
  -d '{"model":"incord-embed","input":["hello world"]}'

# (Add models to your config first if you want to test TTS / STT / Rerank)
```

## Troubleshooting

- **CORS errors in browser console**: backend isn't sending CORS headers. Check that `tower-http` cors layer is enabled (it is by default in M19+).
- **Model fails to load at startup**: check the path in `[[models]]` is correct and the file exists. Check log output for the actual error.
- **Out of memory**: reduce `slot_count` in `[scheduler]` or omit larger models from the config.
