#!/usr/bin/env bash
# 24-hour soak test for QORA API server.
# Runs load_test for 24h while sampling RSS every minute.
# Verifies <5% RSS drift (no memory leaks).
#
# Usage:
#   URL=https://api.example.com KEY=qora_yourkey CONCURRENCY=100 ./scripts/soak_24h.sh
#
# Environment variables:
#   URL          - API base URL (required)
#   KEY          - API key for load test (required)
#   CONCURRENCY  - Concurrent connections (default: 100)
#   MAX_TOKENS   - Tokens per request (default: 200)
#   DURATION     - Soak duration in seconds (default: 86400 = 24h)

set -euo pipefail

URL=${URL:?URL env var required}
KEY=${KEY:?KEY env var required}
CONCURRENCY=${CONCURRENCY:-100}
MAX_TOKENS=${MAX_TOKENS:-200}
DURATION=${DURATION:-86400}

LOGDIR=${LOGDIR:-/var/log/qora}
mkdir -p "$LOGDIR"
TS=$(date +%Y%m%d_%H%M)
LOG=$LOGDIR/soak_${TS}.log
MEMLOG=$LOGDIR/soak_${TS}.mem

# Find the qora-server PID. Adjust if the binary name differs.
PID=$(pgrep -x qora-server || pgrep -f 'qora-server' | head -1)
if [ -z "$PID" ]; then
    echo "qora-server is not running" >&2
    exit 1
fi
echo "Monitoring qora-server PID=$PID for ${DURATION}s ($((DURATION/3600))h)"
echo "Load: ${CONCURRENCY} concurrent, max_tokens=${MAX_TOKENS}"
echo "Logs: $LOG (load), $MEMLOG (memory)"

# Background memory sampler
(
    while sleep 60; do
        if ! kill -0 "$PID" 2>/dev/null; then
            echo "$(date -Iseconds) qora-server process died" >> "$MEMLOG"
            exit 1
        fi
        rss=$(ps -o rss= -p "$PID" 2>/dev/null || echo 0)
        echo "$(date -Iseconds) RSS=${rss}KB" >> "$MEMLOG"
    done
) &
MON=$!

trap "kill $MON 2>/dev/null || true" EXIT

# Run the load test
./target/release/load_test \
    --url "$URL" \
    --key "$KEY" \
    --concurrency "$CONCURRENCY" \
    --duration "$DURATION" \
    --max-tokens "$MAX_TOKENS" \
    | tee -a "$LOG"

kill $MON 2>/dev/null || true

# Analyze memory drift
if [ ! -s "$MEMLOG" ]; then
    echo "memory log empty — sampler did not run"
    exit 1
fi

first=$(head -1 "$MEMLOG" | awk -F'RSS=' '{print $2}' | tr -d 'KB')
last=$(tail -1 "$MEMLOG"  | awk -F'RSS=' '{print $2}' | tr -d 'KB')

if [ -z "$first" ] || [ -z "$last" ]; then
    echo "could not parse RSS values from $MEMLOG"
    exit 1
fi

drift=$(awk -v a="$first" -v b="$last" 'BEGIN { printf "%.2f", (b-a)/a*100 }')
echo
echo "=== Memory Analysis ==="
echo "first RSS: ${first} KB"
echo "last RSS:  ${last} KB"
echo "drift:     ${drift}%"

# Pass if drift is within ±5%
ok=$(awk -v d="$drift" 'BEGIN { exit !(d < 5 && d > -5) }' && echo PASS || echo FAIL)
echo "result:    $ok"

if [ "$ok" = "FAIL" ]; then
    echo
    echo "RSS drift exceeded 5% — likely memory leak. Review logs:"
    echo "  $LOG"
    echo "  $MEMLOG"
    exit 1
fi
