#!/usr/bin/env bash
set -euo pipefail
# Usage: ./scripts/run_amber.sh <config.json> <log_dir> <amber_path> [cache_dir]

CONFIG_FILE="${1:?config.json}"
LOG_DIR="${2:?log_dir}"
AMBER_PATH="${3:?AMBER dataset root}"
CACHE_DIR="${4:-./cache/huggingface_cache}"

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_DIR}"
export PYTHONPATH="${REPO_DIR}${PYTHONPATH:+:${PYTHONPATH}}"

cmd=(python run_AMBER.py --config "${CONFIG_FILE}" --amber_path "${AMBER_PATH}" --log_dir "${LOG_DIR}" --cache_dir "${CACHE_DIR}")
[[ -n "${BENCH_NUM_EXAMPLES:-}" && -z "${FULL_BENCHMARK:-}" ]] && cmd+=(--num_items "${BENCH_NUM_EXAMPLES}")
[[ "${ONLY_GENERATIVE:-0}" == "1" ]] && cmd+=(--only_generative)
[[ "${ONLY_DISCRIMINATIVE:-0}" == "1" ]] && cmd+=(--only_discriminative)
printf '%q ' "${cmd[@]}"
echo
exec "${cmd[@]}"
