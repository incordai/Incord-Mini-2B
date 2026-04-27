# Production Load Test Runbook (EPYC 9354)

This runbook describes the load test you should run on the production server **after** initial deploy and **before** opening the gates to paying customers. It does not run in CI — it requires the real model file and the EPYC server.

## Goal

Empirically verify per-server capacity matches the design targets:

- **150 concurrent**: p95 < 30 s for a 500-token completion
- **100 concurrent**: p95 < 20 s for a 500-token completion
- 24 h soak at 100 concurrent: RSS drift < 5%, no panics, no slot leaks

If 150 concurrent does not meet the SLO, the corrective action is profiling + scheduler tuning before launch (do not launch with under-capacity).

## Prerequisites

On the EPYC server:

1. Service installed via `scripts/install.sh` and started: `systemctl status qora-server`
2. `model.qor4b` and `tokenizer.json` in place
3. `QORA_SERVER_SECRET` and `QORA_ADMIN_TOKEN` set in `/etc/qora/qora.env`
4. `/health` returns 200 from another machine
5. `target/release/load_test` binary built (or copied from another build host)

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

## Step 2 — Ramp tests at increasing concurrency

Run from a separate machine on the same network (or localhost if testing from the EPYC box itself):

```bash
URL=https://api.your-domain.com
KEY=$(cat /tmp/loadtest.key)

for c in 10 50 100 150 200 300; do
    echo "=== concurrency $c ==="
    ./target/release/load_test \
        --url "$URL" \
        --key "$KEY" \
        --concurrency $c \
        --duration 300 \
        --max-tokens 500 \
        --prompt "Explain entropy in 200 words." \
        | tee "/tmp/loadtest_c${c}.log"
done
```

Each band runs 5 minutes. Total wall-clock: ~30 minutes.

## Step 3 — Compare against the SLO table

Extract p95 from each log:

```bash
for c in 10 50 100 150 200 300; do
    p95=$(grep 'p95' /tmp/loadtest_c${c}.log | awk '{print $NF}')
    tput=$(grep 'throughput' /tmp/loadtest_c${c}.log | awk '{print $2}')
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

## Step 4 — 24-hour soak test

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

## Step 5 — Capture baseline numbers

After the ramp + soak pass, record the actual numbers in `docs/load-test-results/<date>/results.md`:

```markdown
# Load Test — <DATE>

Hardware: AMD EPYC 9354 32C/64T, 256 GB DDR5, 2× 4 TB NVMe SSD
Model: model.qor4b (Q4 quantization), 32 layers, hidden=2560
QORA-server: <git SHA>
slot_count: <value from config>

## Ramp results

| Concurrency | requests | errors | throughput | p50 | p95 | p99 |
|---|---|---|---|---|---|---|
| 10  | ... | ... | ... | ... | ... | ... |
| 50  | ... | ... | ... | ... | ... | ... |
| 100 | ... | ... | ... | ... | ... | ... |
| 150 | ... | ... | ... | ... | ... | ... |
| 200 | ... | ... | ... | ... | ... | ... |
| 300 | ... | ... | ... | ... | ... | ... |

## Soak

24 h, concurrency=100, max_tokens=200
- requests served: ...
- errors: ...
- RSS first: ... KB
- RSS last:  ... KB
- drift: ...%
- result: PASS/FAIL
```

These numbers become the baseline. Future hardware changes or scheduler tunings get compared against this baseline before launch.

## Step 6 — Cleanup

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

- The load test sets `temperature: 0.0` so completions are deterministic (greedy). Keep this — non-deterministic generations make latency comparisons noisier.
- `--max-tokens 500` sized to roughly match the median customer request. Adjust if your traffic pattern differs.
- The numbers above are targets, not guarantees. Real-world performance depends on prompt distribution, context lengths, and concurrent vision encodes (which serialize on the vision_semaphore).
- If you scale beyond 150 concurrent on a single node, you've outgrown one box — add a second EPYC behind a load balancer rather than chasing single-node optimizations past this point.
