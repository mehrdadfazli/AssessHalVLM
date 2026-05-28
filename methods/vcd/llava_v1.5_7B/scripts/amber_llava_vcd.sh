#!/bin/bash
#SBATCH --partition=gpuq
#SBATCH --qos=gpu
#SBATCH --gres=gpu:A100.80gb:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --time=10:00:00
#SBATCH --job-name=amber-llava-vcd
#SBATCH --output=/scratch/smansou3/LVLM/lvlm-logs/AMBER/llava_VCD/slurm-%j.log

set -euo pipefail

mkdir -p /scratch/smansou3/LVLM/lvlm-logs/AMBER/llava_VCD

source /scratch/smansou3/avisc_env.sh
source /scratch/smansou3/avisc-env/bin/activate
cd /scratch/smansou3/LVLM/AvisC

LOG_PATH=/scratch/smansou3/LVLM/lvlm-logs/AMBER/llava_VCD
META=$LOG_PATH/job_metadata_${SLURM_JOB_ID}.txt
SENTINEL=/tmp/.sentinel_${SLURM_JOB_ID}

EXPECTED_CD=true
EXPECTED_M3ID=false
EXPECTED_AVISC=false

{
    echo "job_id=$SLURM_JOB_ID"
    echo "job_name=$SLURM_JOB_NAME"
    echo "start_time=$(date -Iseconds)"
    echo "host=$(hostname)"
    echo "model=llava-1.5-7b"
    echo "method=VCD"
    echo "benchmark=AMBER"
    echo "model_path=liuhaotian/llava-v1.5-7b"
    echo "data_path=/scratch/smansou3/LVLM/datasets/AMBER/image"
    echo "json_path=/scratch/smansou3/LVLM/AvisC/experiments/AMBER/data/query/query_generative.json"
    echo "log_path=$LOG_PATH"
    echo "seed=42"
    echo "max_token=512"
    echo "use_cd=$EXPECTED_CD"
    echo "use_m3id=$EXPECTED_M3ID"
    echo "use_avisc=$EXPECTED_AVISC"
} > "$META"

echo "=== nvidia-smi ==="
nvidia-smi
echo ""
echo "=== /home PRE ==="
du -sh /home/smansou3/.cache/ /home/smansou3/.cache/huggingface/ /home/smansou3/.cache/torch/ 2>/dev/null || true
echo ""
echo "=== HF_HOME=$HF_HOME ==="
echo "=== job starting at $(date) ==="

echo "=== exact python invocation ==="
echo "python experiments/cd_scripts/amber_eval_llava.py \\
    --model-path liuhaotian/llava-v1.5-7b \\
    --data_path /scratch/smansou3/LVLM/datasets/AMBER/image \\
    --json_path /scratch/smansou3/LVLM/AvisC/experiments/AMBER/data/query/query_generative.json \\
    --log_path $LOG_PATH \\
    --seed 42 \\
    --max_token 512 \\
    --gpu-id 0 \\
    --use_cd True --use_m3id False --use_avisc False"
echo ""

# Sentinel: touch BEFORE python so the post-run find for command_line_args.json
# only matches THIS run (not stale files from prior re-runs in the same dir).
touch "$SENTINEL"

EXIT_CODE=0
python experiments/cd_scripts/amber_eval_llava.py \
    --model-path liuhaotian/llava-v1.5-7b \
    --data_path /scratch/smansou3/LVLM/datasets/AMBER/image \
    --json_path /scratch/smansou3/LVLM/AvisC/experiments/AMBER/data/query/query_generative.json \
    --log_path "$LOG_PATH" \
    --seed 42 \
    --max_token 512 \
    --gpu-id 0 \
    --use_cd True --use_m3id False --use_avisc False || EXIT_CODE=$?

echo "=== job done at $(date), python exit=$EXIT_CODE ==="
echo ""
echo "=== /home POST ==="
du -sh /home/smansou3/.cache/ /home/smansou3/.cache/huggingface/ /home/smansou3/.cache/torch/ 2>/dev/null || true
echo ""
echo "=== /scratch caches POST ==="
du -sh /scratch/smansou3/.cache/huggingface/ /scratch/smansou3/.cache/torch/ 2>/dev/null
echo ""

# Flag verification: parse the command_line_args.json that THIS run wrote.
echo "=== flag verification ==="
ARGS_JSON=$(find "$LOG_PATH" -name "command_line_args.json" -newer "$SENTINEL" -type f 2>/dev/null | head -1)
if [ -n "${ARGS_JSON:-}" ] && [ -f "$ARGS_JSON" ]; then
    ACTUAL_CD=$(python -c "import json; print(str(json.load(open('$ARGS_JSON'))['use_cd']).lower())")
    ACTUAL_M3ID=$(python -c "import json; print(str(json.load(open('$ARGS_JSON'))['use_m3id']).lower())")
    ACTUAL_AVISC=$(python -c "import json; print(str(json.load(open('$ARGS_JSON'))['use_avisc']).lower())")
    if [ "$ACTUAL_CD" != "$EXPECTED_CD" ] || [ "$ACTUAL_M3ID" != "$EXPECTED_M3ID" ] || [ "$ACTUAL_AVISC" != "$EXPECTED_AVISC" ]; then
        echo "FLAG MISMATCH: expected use_cd=$EXPECTED_CD use_m3id=$EXPECTED_M3ID use_avisc=$EXPECTED_AVISC, got use_cd=$ACTUAL_CD use_m3id=$ACTUAL_M3ID use_avisc=$ACTUAL_AVISC"
        EXIT_CODE=3
    else
        echo "Flag verification PASSED: use_cd=$ACTUAL_CD use_m3id=$ACTUAL_M3ID use_avisc=$ACTUAL_AVISC"
    fi
else
    echo "Flag verification SKIPPED: command_line_args.json not found (path=$LOG_PATH)"
fi
echo ""

# Output verification: AMBER produces a single Amber_result.json (NOT a JSONL).
RESULT_JSON=$(find "$LOG_PATH" -name "Amber_result.json" -newer "$SENTINEL" -type f 2>/dev/null | head -1)
RESULT_COUNT=0
if [ -n "${RESULT_JSON:-}" ] && [ -f "$RESULT_JSON" ]; then
    RESULT_COUNT=$(python -c "import json; d=json.load(open('$RESULT_JSON')); print(len(d) if isinstance(d, list) else len(d.get('results', d)))" 2>/dev/null || echo 0)
fi

{
    echo "end_time=$(date -Iseconds)"
    echo "exit_code=$EXIT_CODE"
    echo "result_json=${RESULT_JSON:-(none)}"
    echo "result_count=$RESULT_COUNT"
    echo "args_json=${ARGS_JSON:-(none)}"
} >> "$META"

echo "=== output verification ==="
ls -la "$LOG_PATH"
echo "Result JSON: ${RESULT_JSON:-(none)}"
echo "Entry count: $RESULT_COUNT  (expected 1004)"

# Cleanup sentinel
rm -f "$SENTINEL"

exit $EXIT_CODE
