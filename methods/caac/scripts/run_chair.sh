#!/usr/bin/env bash
set -euo pipefail
# Usage: ./scripts/run_chair.sh <config.json> <log_dir> <chair_data_path> [cache_dir]

CONFIG_FILE="${1:?config.json}"
LOG_DIR="${2:?log_dir}"
CHAIR_DATA="${3:?path to COCO val2014 directory}"
CACHE_DIR="${4:-./cache/huggingface_cache}"

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_DIR}"
export PYTHONPATH="${REPO_DIR}${PYTHONPATH:+:${PYTHONPATH}}"

cmd=(python run_CHAIR.py --config "${CONFIG_FILE}" --chair_data_path "${CHAIR_DATA}" --log_dir "${LOG_DIR}" --cache_dir "${CACHE_DIR}")
[[ -n "${BENCH_NUM_EXAMPLES:-}" ]] && cmd+=(--num_images "${BENCH_NUM_EXAMPLES}")
printf '%q ' "${cmd[@]}"
echo
exec "${cmd[@]}"
