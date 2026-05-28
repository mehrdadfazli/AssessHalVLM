#!/usr/bin/env bash
set -euo pipefail
# Usage: ./scripts/run_mmhal.sh <config.json> <log_dir> <input.json> <images_root> [cache_dir]

CONFIG_FILE="${1:?config.json}"
LOG_DIR="${2:?log_dir}"
INPUT_JSON="${3:?MMHal response_template.json}"
IMAGES_ROOT="${4:?MMHal images directory}"
CACHE_DIR="${5:-./cache/huggingface_cache}"

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_DIR}"
export PYTHONPATH="${REPO_DIR}${PYTHONPATH:+:${PYTHONPATH}}"

cmd=(
  python run_MMHal.py --config "${CONFIG_FILE}" --input "${INPUT_JSON}" --images_root "${IMAGES_ROOT}"
  --log_dir "${LOG_DIR}" --cache_dir "${CACHE_DIR}"
)
[[ -n "${BENCH_NUM_EXAMPLES:-}" ]] && cmd+=(--limit "${BENCH_NUM_EXAMPLES}")
printf '%q ' "${cmd[@]}"
echo
exec "${cmd[@]}"
