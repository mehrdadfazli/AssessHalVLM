#!/usr/bin/env bash
set -euo pipefail
# Usage: ./scripts/run_mmstar.sh <config.json> <log_dir> <mmstar_data_root> [mmstar_hf_cache] [cache_dir]
# mmstar_data_root should contain MMStar/images/*.png (created on first run from HF metadata).

CONFIG_FILE="${1:?config.json}"
LOG_DIR="${2:?log_dir}"
MMSTAR_ROOT="${3:?parent of MMStar/images}"
HF_CACHE="${4:-./cache/huggingface_cache}"
CACHE_DIR="${5:-./cache/huggingface_cache}"

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_DIR}"
export PYTHONPATH="${REPO_DIR}${PYTHONPATH:+:${PYTHONPATH}}"

cmd=(
  python run_MMStar.py --config "${CONFIG_FILE}" --mmstar_data_root "${MMSTAR_ROOT}" --log_dir "${LOG_DIR}"
  --cache_dir "${CACHE_DIR}" --mmstar_hf_cache "${HF_CACHE}"
)
[[ -n "${BENCH_NUM_EXAMPLES:-}" && -z "${FULL_BENCHMARK:-}" ]] && cmd+=(--limit "${BENCH_NUM_EXAMPLES}")
printf '%q ' "${cmd[@]}"
echo
exec "${cmd[@]}"
