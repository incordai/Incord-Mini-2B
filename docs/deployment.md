# QORA Server Deployment Guide

This guide covers deploying the QORA API server on a single Linux box.

## Hardware Requirements

Minimum (light traffic, ~10 concurrent users):
- CPU with AVX-512 support (Intel 11th gen+, AMD Zen 4+)
- 8 GB RAM
- 8 GB disk

Recommended (production, ~150 concurrent users):
- AMD EPYC 9354 or similar (32C/64T, 12-channel DDR5, ~400 GB/s memory bandwidth)
- 256 GB DDR5
- 2× 4 TB NVMe SSD (RAID 1 for the auth.redb file)
- 1 Gbit/s network minimum (2 Gbit/s recommended for streaming many concurrent SSE connections)

## Required Files

- `qora-server` binary (built from this repo via `cargo build --release --bin qora-server`)
- `model.qor4b` — quantized model weights (~3.5 GB)
- `tokenizer.json` — HuggingFace tokenizer file
- `qora-server.toml` — server config
- `qora.env` — environment variables (secrets)

## Install

From a fresh checkout on the target machine:

```bash
sudo ./scripts/install.sh
```

This:
- Creates the `qora` system user
- Installs `qora-server` to `/usr/local/bin/`
- Writes default config to `/etc/qora/qora-server.toml`
- Creates env-file template at `/etc/qora/qora.env`
- Installs systemd unit
- Reloads systemd

## Configuration

Edit `/etc/qora/qora-server.toml`:

- **`server.bind`**: `"0.0.0.0:443"` for TLS, or `"127.0.0.1:8080"` if behind a reverse proxy
- **`server.tls_cert` / `tls_key`**: paths to fullchain and privkey PEM files (omit both for plain HTTP)
- **`model.path`**, **`model.tokenizer`**: paths to model + tokenizer
- **`scheduler.slot_count`**: number of concurrent decoders. Start with 64 for the EPYC reference; tune based on memory bandwidth and per-user latency target.
- **`scheduler.max_queue`**: how many requests to queue before returning 503. Default 1000.
- **`auth.db_path`**: path to redb file (must be writable by qora user)
- **`limits.default.rpm` / `tpm`**: default per-key rate limits

Edit `/etc/qora/qora.env`:

- **`QORA_SERVER_SECRET`**: 32-byte random hex. Generate with:
  ```bash
  QORA_SERVER_SECRET=$(head -c 32 /dev/urandom | xxd -p -c 64)
  ```
  This is the HMAC key for API key indexing. **Rotating this invalidates all existing API keys** — back up before changing.

- **`QORA_ADMIN_TOKEN`**: bearer token for `/v1/admin/*` endpoints. Generate similarly. Keep secret.

## TLS Certificates

For Let's Encrypt:

```bash
sudo apt install certbot
sudo certbot certonly --standalone -d api.your-domain.com
# Cert is in /etc/letsencrypt/live/api.your-domain.com/
```

Update `qora-server.toml`:

```toml
[server]
tls_cert = "/etc/letsencrypt/live/api.your-domain.com/fullchain.pem"
tls_key  = "/etc/letsencrypt/live/api.your-domain.com/privkey.pem"
```

Note: certbot needs read access to the cert directory. Either:
- Run qora-server as root (not recommended)
- Add the qora user to a group with read access (preferred)
- Use a reverse proxy (nginx/Caddy) terminating TLS

## Backup Procedure

The redb auth/usage database lives at `auth.db_path` (default `/var/lib/qora/auth.redb`).

Daily backup via cron:

```bash
# /etc/cron.daily/qora-backup
#!/bin/bash
DATE=$(date +%Y-%m-%d)
cp /var/lib/qora/auth.redb /var/lib/qora/backups/auth-$DATE.redb
find /var/lib/qora/backups -name "auth-*.redb" -mtime +30 -delete
```

(redb supports concurrent reads so live backup is safe.)

To restore:

```bash
systemctl stop qora-server
cp /var/lib/qora/backups/auth-2026-04-25.redb /var/lib/qora/auth.redb
chown qora:qora /var/lib/qora/auth.redb
systemctl start qora-server
```

## Operations

Start: `systemctl start qora-server`
Stop: `systemctl stop qora-server` (graceful — finishes in-flight requests)
Restart: `systemctl restart qora-server`
Logs: `journalctl -u qora-server -f`
Health: `curl https://api.your-domain.com/health`
Metrics: `curl http://localhost:9090/metrics` (or wherever Prometheus is configured)

## Admin operations

Create an API key:

```bash
curl -X POST https://api.your-domain.com/v1/admin/keys \
  -H "Authorization: Bearer $QORA_ADMIN_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"user_id":"customer-1","name":"production","rpm_limit":600,"tpm_limit":600000}'
```

The response includes `raw_key` — show this to the customer ONCE. It's never recoverable.

Revoke a key:

```bash
curl -X DELETE https://api.your-domain.com/v1/admin/keys/<key-id> \
  -H "Authorization: Bearer $QORA_ADMIN_TOKEN"
```

## Admin Token Rotation

To rotate `QORA_ADMIN_TOKEN`:
1. Edit `/etc/qora/qora.env` with the new value.
2. `systemctl restart qora-server`.
3. Update any deployment automation that uses the old token.

To rotate `QORA_SERVER_SECRET` (rare; invalidates all API keys):
1. Schedule downtime — all customers must be informed.
2. Edit `/etc/qora/qora.env`.
3. `systemctl restart qora-server`.
4. Re-issue API keys to all customers.

## Monitoring

Recommended Prometheus scrape:

```yaml
# /etc/prometheus/prometheus.yml
scrape_configs:
  - job_name: 'qora'
    scrape_interval: 15s
    static_configs:
      - targets: ['localhost:9090']
```

Key metrics to alert on:
- `qora_active_slots / config.slot_count > 0.9` for 5 min → over capacity, consider another node
- `qora_queue_depth > 0` sustained → backlog growing
- Increase in `qora_errors_total{error_type="inference_error"}` → investigate logs
- `rate(qora_request_duration_seconds_bucket{le="30"}[5m]) / rate(qora_request_duration_seconds_count[5m]) < 0.95` → p95 above 30s

## Capacity Tuning

The Batcher uses a fixed slot pool. The right `slot_count` balances:
- **Higher** → more concurrent users, lower per-user throughput
- **Lower** → fewer concurrent users, higher per-user throughput

For the EPYC 9354 reference target (~400 GB/s memory bandwidth, 3.5 GB Q4 weights), each batched forward pass takes ~9 ms minimum (3.5 / 400 × 1000). Slot counts:

- 64 slots → ~110 batched fwd passes/sec → ~1700 tok/s aggregate, ~25 tok/s per user
- 128 slots → similar aggregate throughput, ~13 tok/s per user
- 32 slots → ~110 fwd/sec, ~50 tok/s per user, but only 32 concurrent

Tune via `scheduler.slot_count` in `qora-server.toml`.

## Troubleshooting

- **Server fails to start with `tokenizer.json not found`**: edit `model.tokenizer` path in toml.
- **`QORA_SERVER_SECRET env var must be set`**: check `/etc/qora/qora.env`.
- **All requests return 401 immediately**: verify the `Authorization: Bearer` header has a valid `qora_…` key. Use `curl -v` to inspect.
- **Requests hang forever**: check `journalctl -u qora-server` for batcher panic; restart service. Most likely a model file corruption or incompatible config version.
- **Memory growth over time**: redb compaction may be needed. Stop service, run `redb compact /var/lib/qora/auth.redb` (if redb-cli is installed), restart.
