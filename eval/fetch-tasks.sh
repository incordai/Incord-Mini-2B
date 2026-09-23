#!/usr/bin/env bash
# Download the task benchmarks used by `incord-eval tasks` from the Hugging Face
# dataset rows API. The data is NOT committed; it lands in the given directory
# (default eval-data/tasks, which is git-ignored).
#
#   bash eval/fetch-tasks.sh [out_dir]
#
# Datasets (licenses): ARC-Challenge test (CC BY-SA 4.0), MMLU test (MIT),
# HellaSwag validation (MIT), WinoGrande-xl validation (CC BY), GSM8K test (MIT).
set -euo pipefail

OUT="${1:-eval-data/tasks}"
API="https://datasets-server.huggingface.co/rows"
PAGE=100

# name|dataset query|total rows
TASKS=(
  "arc|dataset=allenai/ai2_arc&config=ARC-Challenge&split=test|1172"
  "mmlu|dataset=cais/mmlu&config=all&split=test|14042"
  "hellaswag|dataset=Rowan/hellaswag&config=default&split=validation|10042"
  "winogrande|dataset=allenai/winogrande&config=winogrande_xl&split=validation|1267"
  "gsm8k|dataset=openai/gsm8k&config=main&split=test|1319"
)

fetch_page() { # url dest
  local url="$1" dest="$2" attempt
  for attempt in 1 2 3 4 5; do
    if curl -sf --max-time 60 "$url" -o "$dest.tmp" && grep -q '"rows"' "$dest.tmp"; then
      mv "$dest.tmp" "$dest"
      return 0
    fi
    sleep $((attempt * 10))
  done
  rm -f "$dest.tmp"
  echo "failed: $url" >&2
  return 1
}

for spec in "${TASKS[@]}"; do
  IFS='|' read -r name query total <<<"$spec"
  mkdir -p "$OUT/$name"
  for ((off = 0; off < total; off += PAGE)); do
    dest=$(printf '%s/%s/page-%05d.json' "$OUT" "$name" "$off")
    [[ -s "$dest" ]] && continue
    fetch_page "$API?$query&offset=$off&length=$PAGE" "$dest"
  done
  echo "$name: $(ls "$OUT/$name" | wc -l) pages"
done
