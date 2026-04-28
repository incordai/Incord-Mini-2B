# Production Load Test Runbook (EPYC 9354)

This runbook describes the load test you should run on the production server **after** initial deploy and **before** opening the gates to paying customers. It does not run in CI — it requires the real model files and the EPYC server.

## Goal

Empirically verify per-server capacity matches the design targets across all five engines:

**LLM (chat)**
- **150 concurrent**: p95 < 30 s for a 500-token completion
- **100 concurrent**: p95 < 20 s for a 500-token completion
- 24 h soak at 100 concurrent: RSS drift < 5%, no panics, no slot leaks

**TTS**
- **8 concurrent**: p95 < 5 s for a 100-word synthesis

**STT**
- **8 concurrent**: p95 < 3 s for a 30-second audio clip

**Embedding**
- **32 concurrent**: p95 < 500 ms per batch of 8 sentences

**Rerank**
- **16 concurrent**: p95 < 1 s for ranking 20 documents

If the LLM 150-concurrent target does not meet SLO, do not launch — profile and tune first.

## Prerequisites

On the EPYC server:

1. Service installed via `scripts/install.sh` and started: `systemctl status qora-server`
2. All model files in place per the `[[models]]` config entries:
   - `model.qor4b` + `tokenizer.json` (LLM)
   - `model.qora-tts` + `tokenizer.json` (TTS)
   - `model.qora-stt-small` (STT)
   - `embed.bin` (Embedding)
   - `rerank.bin` (Rerank)
3. `QORA_SERVER_SECRET` and `QORA_ADMIN_TOKEN` set in `/etc/qora/qora.env`
4. `/health` returns 200 from another machine
5. `target/release/load_test` binary built (or copied from another build host)
6. `GET /v1/models` lists all five engine IDs

## Step 1 — Create a load-test API key

```bash
ADMIN_TOKEN=$(grep QORA_ADMIN_TOKEN /etc/qora/qora.env | cut -d= -f2)

curl -sS https://api.your-domain.com/v1/admin/keys \
    -H "Authorization: Bearer $ADMIN_TOKEN" \
    -H "Content-Type: application/json" \
    -d '{"user_id":"loadtest","name":"loadtest","rpm_limit":100000,"tpm_limit":100000000}' \
    > /tmp/loadtest-key.json

KEY=$(jq -r .raw_key /tmp/loadtest-key.json)
echo "$KEY" > /tmp/loadtest.key  # save for later
```

The high `rpm_limit` and `tpm_limit` ensure the load test isn't itself rate-limited.

## Step 2 — Ramp tests: LLM (chat)

Run from a separate machine on the same network (or localhost if testing from the EPYC box itself):

```bash
URL=https://api.your-domain.com
KEY=$(cat /tmp/loadtest.key)

for c in 10 50 100 150 200 300; do
    echo "=== concurrency $c ==="
    ./target/release/load_test \
        --url "$URL" \
        --key "$KEY" \
        --endpoint /v1/chat/completions \
        --model qora-llm \
        --concurrency $c \
        --duration 300 \
        --max-tokens 500 \
        --prompt "Explain entropy in 200 words." \
        | tee "/tmp/loadtest_llm_c${c}.log"
done
```

Each band runs 5 minutes. Total wall-clock: ~30 minutes.

## Step 3 — Compare LLM results against the SLO table

```bash
for c in 10 50 100 150 200 300; do
    p95=$(grep 'p95' /tmp/loadtest_llm_c${c}.log | awk '{print $NF}')
    tput=$(grep 'throughput' /tmp/loadtest_llm_c${c}.log | awk '{print $2}')
    echo "concurrency=$c p95_ms=$p95 throughput=$tput req/sec"
done
```

Target table:

| Concurrency | Target p95 | Target throughput |
|---|---|---|
| 10 | < 5 s | > 0.5 req/s |
| 50 | < 10 s | > 2 req/s |
| 100 | < 20 s | > 4 req/s |
| 150 | < 30 s | > 5 req/s |
| 200 | < 60 s | > 5 req/s (saturated) |
| 300 | best-effort | per-user latency degrades |

If 150 concurrent fails p95 < 30 s, do NOT launch. Profile (via `perf record` or `cargo flamegraph`) and identify the bottleneck before retrying.

## Step 4 — Ramp tests: TTS, STT, Embedding, Rerank

Run each engine at its expected concurrency ceiling. Adjust `--concurrency` based on your expected traffic mix.

```bash
URL=https://api.your-domain.com
KEY=$(cat /tmp/loadtest.key)

# TTS — synthesize ~100 words at up to 8 concurrent requests
./target/release/load_test \
    --url "$URL" \
    --key "$KEY" \
    --endpoint /v1/audio/speech \
    --model qora-tts \
    --concurrency 8 \
    --duration 180 \
    --tts-input "The quick brown fox jumps over the lazy dog. QORA delivers fast, natural speech synthesis." \
    | tee /tmp/loadtest_tts.log

# STT — transcribe a 30-second clip at up to 8 concurrent requests
./target/release/load_test \
    --url "$URL" \
    --key "$KEY" \
    --endpoint /v1/audio/transcriptions \
    --model qora-stt \
    --concurrency 8 \
    --duration 180 \
    --audio-file /tmp/loadtest_30s.wav \
    | tee /tmp/loadtest_stt.log

# Embedding — batch of 8 sentences at up to 32 concurrent
./target/release/load_test \
    --url "$URL" \
    --key "$KEY" \
    --endpoint /v1/embeddings \
    --model qora-embed \
    --concurrency 32 \
    --duration 180 \
    --embed-batch 8 \
    | tee /tmp/loadtest_embed.log

# Rerank — 20 documents per request at up to 16 concurrent
./target/release/load_test \
    --url "$URL" \
    --key "$KEY" \
    --endpoint /v1/rerank \
    --model qora-rerank \
    --concurrency 16 \
    --duration 180 \
    --rerank-docs 20 \
    | tee /tmp/loadtest_rerank.log
```

Expected SLOs:

| Engine | Concurrency | Target p95 |
|--------|------------|-----------|
| TTS | 8 | < 5 s |
| STT | 8 | < 3 s |
| Embedding | 32 | < 500 ms |
| Rerank | 16 | < 1 s |

## Step 5 — 24-hour soak test (LLM)

Once the ramp tests pass, run the soak script:

```bash
URL=https://api.your-domain.com KEY=$(cat /tmp/loadtest.key) \
    CONCURRENCY=100 \
    ./scripts/soak_24h.sh
```

This runs for 24 hours. Monitor `/var/log/qora/soak_*.log` and `*.mem`. The script will print PASS/FAIL based on RSS drift.

While the soak runs, periodically:
- Check `journalctl -u qora-server --since "1 hour ago" | grep -i panic` — should be empty
- Check Prometheus `qora_active_slots` — should oscillate around the working set, not climb monotonically
- Check `qora_errors_total` — should stay near zero (or grow only with deliberate 429s if you're saturating limits)

## Step 6 — Capture baseline numbers

After the ramp + soak pass, record the actual numbers in `docs/load-test-results/<date>/results.md`:

```markdown
# Load Test — <DATE>

Hardware: AMD EPYC 9354 32C/64T, 256 GB DDR5, 2× 4 TB NVMe SSD
QORA-server: <git SHA>
slot_count: <value from config>
Models loaded: qora-llm, qora-tts, qora-stt, qora-embed, qora-rerank

## LLM ramp results

| Concurrency | requests | errors | throughput | p50 | p95 | p99 |
|---|---|---|---|---|---|---|
| 10  | ... | ... | ... | ... | ... | ... |
| 50  | ... | ... | ... | ... | ... | ... |
| 100 | ... | ... | ... | ... | ... | ... |
| 150 | ... | ... | ... | ... | ... | ... |
| 200 | ... | ... | ... | ... | ... | ... |
| 300 | ... | ... | ... | ... | ... | ... |

## Non-LLM engine results

| Engine | Concurrency | requests | p95 | result |
|--------|------------|---------|-----|--------|
| TTS    | 8  | ... | ... | PASS/FAIL |
| STT    | 8  | ... | ... | PASS/FAIL |
| Embed  | 32 | ... | ... | PASS/FAIL |
| Rerank | 16 | ... | ... | PASS/FAIL |

## 24h soak (LLM, concurrency=100)

- requests served: ...
- errors: ...
- RSS first: ... KB
- RSS last:  ... KB
- drift: ...%
- result: PASS/FAIL
```

These numbers become the baseline. Future hardware changes or scheduler tunings get compared against this baseline before launch.

## Step 7 — Cleanup

After tests:

```bash
# Revoke the load-test key
KEY_ID=$(jq -r .id /tmp/loadtest-key.json)
ADMIN_TOKEN=$(grep QORA_ADMIN_TOKEN /etc/qora/qora.env | cut -d= -f2)
curl -X DELETE "https://api.your-domain.com/v1/admin/keys/$KEY_ID" \
    -H "Authorization: Bearer $ADMIN_TOKEN"

# Remove key file
rm /tmp/loadtest.key /tmp/loadtest-key.json
```

## Notes

- The load test sets `temperature: 0.0` so LLM completions are deterministic (greedy). Keep this — non-deterministic generations make latency comparisons noisier.
- `--max-tokens 500` sized to roughly match the median customer request. Adjust if your traffic pattern differs.
- The numbers above are targets, not guarantees. Real-world performance depends on prompt distribution, context lengths, concurrent vision encodes (which serialize on the vision_semaphore), and the number of non-LLM requests happening concurrently.
- Non-LLM engines (TTS, STT, Embed, Rerank) each have their own concurrency semaphore. If you see queuing at the SLO limits, raise the semaphore permit counts in `crates/qora-server/src/bin/qora_server.rs` and rebuild.
- If you scale beyond 150 LLM-concurrent on a single node, you've outgrown one box — add a second EPYC behind a load balancer rather than chasing single-node optimizations past this point.
