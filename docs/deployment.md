# Incord Server Deployment Guide

This guide covers deploying the Incord multi-model API server on a single Linux box.

## Hardware Requirements

Minimum (light traffic, ~10 concurrent users, all engines loaded):
- CPU with AVX-512 support (Intel 11th gen+, AMD Zen 4+)
- 16 GB RAM (LLM ~4 GB + TTS ~1 GB + STT ~500 MB + RAG ~500 MB + OS overhead)
- 20 GB disk

Recommended (production, ~150 concurrent users):
- AMD EPYC 9354 or similar (32C/64T, 12-channel DDR5, ~400 GB/s memory bandwidth)
- 256 GB DDR5
- 2× 4 TB NVMe SSD (RAID 1 for the auth.redb file)
- 1 Gbit/s network minimum (2 Gbit/s recommended for streaming many concurrent SSE connections)

RAM sizing for all five engines loaded simultaneously:

| Engine | Model size (approx) |
|--------|-------------------|
| LLM (chat + vision) | ~4 GB (4B Q4) |
| TTS | ~1 GB |
| STT (Whisper-small) | ~500 MB |
| Embedding | ~400 MB |
| Reranker | ~400 MB |
| **Total** | **~6–7 GB weights** + OS + buffers ≈ **12–15 GB minimum** |

If you load a larger LLM variant (9B or 27B), add proportionally more RAM.

## Required Files

Per-engine model files (all paths configurable in `incord-server.toml`):

- `incord-server` binary (built from this repo via `cargo build --release --bin incord-server`)
- **LLM engine**: `model.incord` (~3.5 GB), `tokenizer.json`
- **TTS engine**: `model.incord-tts`, `tokenizer.json`
- **STT engine**: `model.incord-stt-small`
- **Embedding engine**: `embed.bin`
- **Rerank engine**: `rerank.bin`
- `incord-server.toml` — server config
- `incord.env` — environment variables (secrets)

Engines you do not need can simply be omitted from the `[[models]]` list in the config — they will not be loaded.

## Install

From a fresh checkout on the target machine:

```bash
sudo ./scripts/install.sh
```

This:
- Creates the `incord` system user
- Installs `incord-server` to `/usr/local/bin/`
- Writes default config to `/etc/incord/incord-server.toml`
- Creates env-file template at `/etc/incord/incord.env`
- Installs systemd unit
- Reloads systemd

## Configuration

Edit `/etc/incord/incord-server.toml`. The key sections are:

### `[server]`

- **`bind`**: `"0.0.0.0:443"` for TLS, or `"127.0.0.1:8080"` if behind a reverse proxy
- **`tls_cert` / `tls_key`**: paths to fullchain and privkey PEM files (omit both for plain HTTP)

### `[scheduler]`

- **`slot_count`**: number of concurrent LLM decoders. Start with 64 for the EPYC reference; tune based on memory bandwidth and per-user latency target.
- **`max_queue`**: how many requests to queue before returning 503. Default 1000.

### `[auth]`

- **`db_path`**: path to redb file (must be writable by incord user)

### `[limits.default]`

- **`rpm` / `tpm`**: default per-key rate limits

### `[[models]]` — registering engines

Each `[[models]]` entry loads one model at startup and registers it in the runtime registry. The `id` is what clients send in the `model` field of their requests.

```toml
[[models]]
id = "incord-llm"
kind = "chat"           # one of: chat | tts | stt | embedding | rerank
path = "/var/lib/incord/llm/model.incord"
tokenizer = "/var/lib/incord/llm/tokenizer.json"
description = "Incord hybrid LLM (text + vision)"
```

Required fields per `kind`:

| `kind` | Required fields | Optional fields |
|--------|----------------|-----------------|
| `chat` | `path`, `tokenizer` | `description` |
| `tts` | `path`, `tokenizer` | `description` |
| `stt` | `path` | `description` |
| `embedding` | `path` | `description` |
| `rerank` | `path` | `description` |

You can register multiple models of the same kind under different `id` values (e.g., a 4B and a 9B LLM at the same time). See `config/incord-server.toml.example` for a full example.

### `[loading]`

- **`mode`**: `"eager"` (default) loads all models at startup. `"lazy"` loads on first request. Use `eager` for production to surface load errors before traffic arrives.

### `/etc/incord/incord.env`

- **`INCORD_SERVER_SECRET`**: 32-byte random hex. Generate with:
  ```bash
  INCORD_SERVER_SECRET=$(head -c 32 /dev/urandom | xxd -p -c 64)
  ```
  This is the HMAC key for API key indexing. **Rotating this invalidates all existing API keys** — back up before changing.

- **`INCORD_ADMIN_TOKEN`**: bearer token for `/v1/admin/*` endpoints. Generate similarly. Keep secret.

## TLS Certificates

For Let's Encrypt:

```bash
sudo apt install certbot
sudo certbot certonly --standalone -d api.your-domain.com
# Cert is in /etc/letsencrypt/live/api.your-domain.com/
```

Update `incord-server.toml`:

```toml
[server]
tls_cert = "/etc/letsencrypt/live/api.your-domain.com/fullchain.pem"
tls_key  = "/etc/letsencrypt/live/api.your-domain.com/privkey.pem"
```

Note: certbot needs read access to the cert directory. Either:
- Run incord-server as root (not recommended)
- Add the incord user to a group with read access (preferred)
- Use a reverse proxy (nginx/Caddy) terminating TLS

## Backup Procedure

The redb auth/usage database lives at `auth.db_path` (default `/var/lib/incord/auth.redb`).

Daily backup via cron:

```bash
# /etc/cron.daily/incord-backup
#!/bin/bash
DATE=$(date +%Y-%m-%d)
cp /var/lib/incord/auth.redb /var/lib/incord/backups/auth-$DATE.redb
find /var/lib/incord/backups -name "auth-*.redb" -mtime +30 -delete
```

(redb supports concurrent reads so live backup is safe.)

To restore:

```bash
systemctl stop incord-server
cp /var/lib/incord/backups/auth-2026-04-25.redb /var/lib/incord/auth.redb
chown incord:incord /var/lib/incord/auth.redb
systemctl start incord-server
```

## Operations

Start: `systemctl start incord-server`
Stop: `systemctl stop incord-server` (graceful — finishes in-flight requests)
Restart: `systemctl restart incord-server`
Logs: `journalctl -u incord-server -f`
Health: `curl https://api.your-domain.com/health`
Models: `curl https://api.your-domain.com/v1/models` (lists all registered engines + IDs)
Metrics: `curl http://localhost:9090/metrics` (or wherever Prometheus is configured)

### Inference endpoints

| Endpoint | Method | Task |
|----------|--------|------|
| `/v1/inference` | POST | Unified — dispatches by `task` field |
| `/v1/chat/completions` | POST | LLM chat (OpenAI-compat) |
| `/v1/audio/speech` | POST | TTS synthesis (OpenAI-compat) |
| `/v1/audio/transcriptions` | POST | STT transcription (OpenAI-compat, multipart) |
| `/v1/embeddings` | POST | Sentence embedding (OpenAI-compat) |
| `/v1/rerank` | POST | Cross-encoder reranking |

## Admin operations

Create an API key:

```bash
curl -X POST https://api.your-domain.com/v1/admin/keys \
  -H "Authorization: Bearer $INCORD_ADMIN_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"user_id":"customer-1","name":"production","rpm_limit":600,"tpm_limit":600000}'
```

The response includes `raw_key` — show this to the customer ONCE. It's never recoverable.

Revoke a key:

```bash
curl -X DELETE https://api.your-domain.com/v1/admin/keys/<key-id> \
  -H "Authorization: Bearer $INCORD_ADMIN_TOKEN"
```

## Admin Token Rotation

To rotate `INCORD_ADMIN_TOKEN`:
1. Edit `/etc/incord/incord.env` with the new value.
2. `systemctl restart incord-server`.
3. Update any deployment automation that uses the old token.

To rotate `INCORD_SERVER_SECRET` (rare; invalidates all API keys):
1. Schedule downtime — all customers must be informed.
2. Edit `/etc/incord/incord.env`.
3. `systemctl restart incord-server`.
4. Re-issue API keys to all customers.

## Monitoring

Recommended Prometheus scrape:

```yaml
# /etc/prometheus/prometheus.yml
scrape_configs:
  - job_name: 'incord'
    scrape_interval: 15s
    static_configs:
      - targets: ['localhost:9090']
```

Key metrics to alert on:
- `incord_active_slots / config.slot_count > 0.9` for 5 min → over capacity, consider another node
- `incord_queue_depth > 0` sustained → backlog growing
- Increase in `incord_errors_total{error_type="inference_error"}` → investigate logs
- `rate(incord_request_duration_seconds_bucket{le="30"}[5m]) / rate(incord_request_duration_seconds_count[5m]) < 0.95` → p95 above 30s

## Capacity Tuning

### LLM (chat) slot pool

The LLM scheduler uses a fixed slot pool. The right `slot_count` balances:
- **Higher** → more concurrent users, lower per-user throughput
- **Lower** → fewer concurrent users, higher per-user throughput

For the EPYC 9354 reference target (~400 GB/s memory bandwidth, 3.5 GB Q4 weights), each batched forward pass takes ~9 ms minimum (3.5 / 400 × 1000). Slot counts:

- 64 slots → ~110 batched fwd passes/sec → ~1700 tok/s aggregate, ~25 tok/s per user
- 128 slots → similar aggregate throughput, ~13 tok/s per user
- 32 slots → ~110 fwd/sec, ~50 tok/s per user, but only 32 concurrent

Tune via `scheduler.slot_count` in `incord-server.toml`.

### Per-engine concurrency semaphores

Each non-LLM engine is gated by a Tokio semaphore that caps parallelism to protect memory and CPU:

| Engine | Default permits | Source constant |
|--------|----------------|-----------------|
| Vision encoder | 1 | `crates/incord-server/src/http/handlers/inference.rs` |
| TTS | 2 | same file |
| STT | 2 | same file |
| Embedding | 8 | same file |
| Rerank | 4 | same file |

These defaults are set in `InferenceState::for_test` and the production `main.rs` wiring. Adjust the permit counts in `crates/incord-server/src/bin/incord_server.rs` to match your hardware profile (more RAM / more cores → raise the cap).

## Troubleshooting

- **Server fails to start with `tokenizer.json not found`**: check the `path`/`tokenizer` fields for the affected `[[models]]` entry.
- **`INCORD_SERVER_SECRET env var must be set`**: check `/etc/incord/incord.env`.
- **Unknown model error**: verify the `id` in the request matches a `[[models]]` entry in the config.
- **All requests return 401 immediately**: verify the `Authorization: Bearer` header has a valid `incord_…` key. Use `curl -v` to inspect.
- **Requests hang forever**: check `journalctl -u incord-server` for batcher panic; restart service. Most likely a model file corruption or incompatible config version.
- **Memory growth over time**: redb compaction may be needed. Stop service, run `redb compact /var/lib/incord/auth.redb` (if redb-cli is installed), restart.
- **TTS/STT requests queue up**: the engine semaphore permits may be too low for your traffic. Increase the permit count in `incord_server.rs` and rebuild.
