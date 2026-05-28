#!/usr/bin/env bash
# score_all.sh — Score CHAIR / MMStar / MMHal for all six CAAC configs.
#                AMBER scoring is currently disabled (pending full run completion).
#
# Required:
#   COCO_ANNO_PATH   Directory with COCO 2014 annotation JSONs (captions + instances).
#   OPENAI_API_KEY   Used by MMHal and MMStar judges.
#
# Optional:
#   RESULTS_TAG      Subdirectory under results/ (default: full)
#   JUDGE_MODEL      GPT model for both judges (default: gpt-4o)
#   CHAIR_CACHE      Path for CHAIR evaluator pickle (default: results/<tag>/chair/chair_evaluator.pkl)
#   FORCE            Set to 1 to re-score even if a summary already exists
#
# Usage:
#   export COCO_ANNO_PATH=/path/to/coco/annotations
#   export OPENAI_API_KEY=sk-...
#   bash scripts/score_all.sh

set -euo pipefail
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_DIR}"
export PYTHONPATH="${REPO_DIR}${PYTHONPATH:+:${PYTHONPATH}}"

: "${COCO_ANNO_PATH:?[score_all] export COCO_ANNO_PATH}"
: "${OPENAI_API_KEY:?[score_all] export OPENAI_API_KEY}"

RESULTS_TAG="${RESULTS_TAG:-full}"
RESULTS_ROOT="${REPO_DIR}/results/${RESULTS_TAG}"
JUDGE_MODEL="${JUDGE_MODEL:-gpt-4o}"
FORCE="${FORCE:-0}"
CHAIR_CACHE="${CHAIR_CACHE:-${RESULTS_ROOT}/chair/chair_evaluator.pkl}"

log()  { echo "[score_all] $*"; }
skip() { echo "[score_all] SKIP $*"; }
ok()   { echo "[score_all] DONE $*"; }

jsonl_unique_count() {
    local file="$1" field="$2"
    python3 - "${file}" "${field}" << 'PY'
import json, sys
seen = set()
for line in open(sys.argv[1], encoding='utf-8'):
    line = line.strip()
    if not line: continue
    try:
        v = json.loads(line).get(sys.argv[2])
        if v is not None: seen.add(int(v))
    except Exception: pass
print(len(seen))
PY
}

json_list_count() {
    python3 - "$1" << 'PY'
import json, sys
try:
    d = json.load(open(sys.argv[1], encoding='utf-8'))
    print(len(d) if isinstance(d, list) else 0)
except Exception: print(0)
PY
}

RUN_NAMES=(
    instructblip_base
    instructblip_w_CAAC
    llava15_base
    llava15_w_CAAC
    llavanext_base
    llavanext_w_CAAC
)

# ═══════════════════════════════════════════════════════════════════════════════
# 1. CHAIR
# ═══════════════════════════════════════════════════════════════════════════════
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ CHAIR ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
CHAIR_EXPECTED=500

for run in "${RUN_NAMES[@]}"; do
    model_tag="${run%%_*}"
    run_dir="${RESULTS_ROOT}/chair/${run}"
    cap_file="${run_dir}/${model_tag}_chair.jsonl"
    summary="${run_dir}/summary_chair.json"

    if [[ ! -f "${cap_file}" ]]; then skip "chair/${run}: no output file"; continue; fi
    n="$(jsonl_unique_count "${cap_file}" "image_id")"
    if (( n < CHAIR_EXPECTED )); then skip "chair/${run}: ${n}/${CHAIR_EXPECTED} — incomplete"; continue; fi
    if [[ -f "${summary}" && "${FORCE}" != "1" ]]; then skip "chair/${run}: summary exists (FORCE=1 to re-score)"; continue; fi

    log "Scoring chair/${run} (${n} images)..."
    python eval/chair.py \
        --cap_file     "${cap_file}" \
        --caption_key  caption_512 \
        --coco_path    "${COCO_ANNO_PATH}" \
        --cache        "${CHAIR_CACHE}" \
        --summary_file "${summary}"
    ok "chair/${run} → ${summary}"
done

# ═══════════════════════════════════════════════════════════════════════════════
# 2. MMStar
# ═══════════════════════════════════════════════════════════════════════════════
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ MMSTAR ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
MMSTAR_EXPECTED=1500

for run in "${RUN_NAMES[@]}"; do
    model_tag="${run%%_*}"
    run_dir="${RESULTS_ROOT}/mmstar/${run}"
    cap_file="${run_dir}/${model_tag}_mmstar.jsonl"
    summary="${run_dir}/summary_mmstar.json"

    if [[ ! -f "${cap_file}" ]]; then skip "mmstar/${run}: no output file"; continue; fi
    n="$(jsonl_unique_count "${cap_file}" "index")"
    if (( n < MMSTAR_EXPECTED )); then skip "mmstar/${run}: ${n}/${MMSTAR_EXPECTED} — incomplete"; continue; fi
    if [[ -f "${summary}" && "${FORCE}" != "1" ]]; then skip "mmstar/${run}: summary exists (FORCE=1 to re-score)"; continue; fi

    log "Scoring mmstar/${run} (${n} samples, judge: ${JUDGE_MODEL})..."
    python eval/mmstar_eval.py \
        --cap_file     "${cap_file}" \
        --scoring      vlmeval \
        --judge-model  "${JUDGE_MODEL}" \
        --api-provider openai \
        --judge-sleep  0.3 \
        --summary_file "${summary}"
    ok "mmstar/${run} → ${summary}"
done

# ═══════════════════════════════════════════════════════════════════════════════
# 3. MMHal-Bench
# ═══════════════════════════════════════════════════════════════════════════════
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ MMHAL ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
MMHAL_EXPECTED=96

for run in "${RUN_NAMES[@]}"; do
    model_tag="${run%%_*}"
    run_dir="${RESULTS_ROOT}/mmhal/${run}"
    resp_file="${run_dir}/${model_tag}_mmhal.json"
    summary="${run_dir}/summary_mmhal.json"

    if [[ ! -f "${resp_file}" ]]; then skip "mmhal/${run}: no output file"; continue; fi
    n="$(json_list_count "${resp_file}")"
    if (( n < MMHAL_EXPECTED )); then skip "mmhal/${run}: ${n}/${MMHAL_EXPECTED} — incomplete"; continue; fi
    if [[ -f "${summary}" && "${FORCE}" != "1" ]]; then skip "mmhal/${run}: summary exists (FORCE=1 to re-score)"; continue; fi

    log "Scoring mmhal/${run} (${n} samples, judge: ${JUDGE_MODEL})..."
    python eval/eval_gpt4.py \
        --response     "${resp_file}" \
        --gpt-model    "${JUDGE_MODEL}" \
        --api-provider openai \
        --request-sleep 1.0 \
        --summary_file "${summary}"
    ok "mmhal/${run} → ${summary}"
done

# ═══════════════════════════════════════════════════════════════════════════════
# 4. Consolidated summary table
# ═══════════════════════════════════════════════════════════════════════════════
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ SUMMARY ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

SUMMARY_OUT="${RESULTS_ROOT}/summary_all.json"

python3 - "${RESULTS_ROOT}" "${SUMMARY_OUT}" << 'PY'
import json, os, sys, glob

results_root = sys.argv[1]
out_path     = sys.argv[2]

RUN_NAMES = [
    "instructblip_base", "instructblip_w_CAAC",
    "llava15_base",       "llava15_w_CAAC",
    "llavanext_base",     "llavanext_w_CAAC",
]

def load_json(path):
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None

all_results = {}

for run in RUN_NAMES:
    row = {}

    # CHAIR
    s = load_json(os.path.join(results_root, "chair", run, "summary_chair.json"))
    if s:
        pct = s.get("metrics_percent", {})
        row["chair_CHAIRs"] = pct.get("CHAIRs")
        row["chair_CHAIRi"] = pct.get("CHAIRi")
        row["chair_Recall"] = pct.get("Recall")

    # MMStar
    s = load_json(os.path.join(results_root, "mmstar", run, "summary_mmstar.json"))
    if s:
        row["mmstar_accuracy"] = round(s.get("overall_mean_hit", 0) * 100, 2)
        row["mmstar_n"]        = s.get("n")

    # MMHal
    s = load_json(os.path.join(results_root, "mmhal", run, "summary_mmhal.json"))
    if s:
        row["mmhal_avg_score"]       = s.get("average_score")
        row["mmhal_hallucination_rate"] = s.get("hallucination_rate")

    if row:
        all_results[run] = row

# Print table
col_w = 26
bench_keys = [
    ("chair_CHAIRs",         "CHAIR-S↓"),
    ("chair_CHAIRi",         "CHAIR-I↓"),
    ("chair_Recall",         "CHAIR-R↑"),
    ("mmstar_accuracy",      "MMStar-Acc↑"),
    ("mmhal_avg_score",      "MMHal-Score↑"),
    ("mmhal_hallucination_rate", "MMHal-Hallu↓"),
]

header = f"{'Config':{col_w}}" + "".join(f"{lbl:>14}" for _, lbl in bench_keys)
print(header)
print("─" * len(header))
for run in RUN_NAMES:
    row = all_results.get(run, {})
    line = f"{run:{col_w}}"
    for key, _ in bench_keys:
        val = row.get(key)
        line += f"{str(round(val,2)) if val is not None else '—':>14}"
    print(line)

# Save JSON
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(all_results, f, indent=2)
print(f"\nSaved → {out_path}")
PY

echo ""
echo "[score_all] All done. Per-run summaries: results/${RESULTS_TAG}/<bench>/<run>/summary_<bench>.json"
echo "[score_all] Consolidated table: ${RESULTS_ROOT}/summary_all.json"
