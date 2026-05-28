#!/bin/bash
#SBATCH --partition=gpuq
#SBATCH --qos=gpu
#SBATCH --gres=gpu:A100.80gb:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --time=01:00:00
#SBATCH --job-name=mmhal-next-vcd
#SBATCH --output=/scratch/smansou3/LVLM/lvlm-logs/MMHal-Bench/llava_next_VCD/slurm-%j.log

set -euo pipefail

mkdir -p /scratch/smansou3/LVLM/lvlm-logs/MMHal-Bench/llava_next_VCD

source /scratch/smansou3/avisc_next_env.sh
source /scratch/smansou3/avisc-next-env/bin/activate
cd /scratch/smansou3/LVLM/AvisC-next/cd_scripts_next

LOG_PATH=/scratch/smansou3/LVLM/lvlm-logs/MMHal-Bench/llava_next_VCD
META=$LOG_PATH/job_metadata_${SLURM_JOB_ID}.txt
SENTINEL=/tmp/.sentinel_${SLURM_JOB_ID}

EXPECTED_CD=true
EXPECTED_M3ID=false
EXPECTED_AVISC=false
EXPECTED_COUNT=96
EXPECTED_TRANSFORMERS=4.47.0

{
    echo "job_id=$SLURM_JOB_ID"
    echo "job_name=$SLURM_JOB_NAME"
    echo "start_time=$(date -Iseconds)"
    echo "host=$(hostname)"
    echo "model=llava-next-v1.6-vicuna-7b"
    echo "method=VCD"
    echo "benchmark=MMHal-Bench"
    echo "model_path=llava-hf/llava-v1.6-vicuna-7b-hf"
    echo "env=avisc-next-env"
    echo "transformers_version=$EXPECTED_TRANSFORMERS"
    echo "data_path=/scratch/smansou3/LVLM/datasets/MMHal-Bench/images"
    echo "jsonl_path=/scratch/smansou3/LVLM/datasets/MMHal-Bench/mmhal_inputs.jsonl"
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
echo ""

echo "=== env sanity: transformers version ==="
ACTUAL_TRANSFORMERS=$(python -c "import transformers; print(transformers.__version__)")
echo "transformers=$ACTUAL_TRANSFORMERS (expected=$EXPECTED_TRANSFORMERS)"
if [ "$ACTUAL_TRANSFORMERS" != "$EXPECTED_TRANSFORMERS" ]; then
    echo "ENV SANITY FAILED: wrong transformers version. Likely wrong venv activated."
    echo "end_time=$(date -Iseconds)" >> "$META"
    echo "exit_code=5" >> "$META"
    rm -f "$SENTINEL"
    exit 5
fi
echo ""

echo "=== job starting at $(date) ==="

echo "=== exact python invocation ==="
echo "python mmhal_eval_llava_next.py \\
    --model-path llava-hf/llava-v1.6-vicuna-7b-hf \\
    --jsonl_path /scratch/smansou3/LVLM/datasets/MMHal-Bench/mmhal_inputs.jsonl \\
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
python mmhal_eval_llava_next.py \
    --model-path llava-hf/llava-v1.6-vicuna-7b-hf \
    --jsonl_path /scratch/smansou3/LVLM/datasets/MMHal-Bench/mmhal_inputs.jsonl \
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

# Output verification: MMHal-Bench produces predictions.jsonl (one record per line, 96 total).
echo "=== output verification ==="
RESULT_FILE=$(find "$LOG_PATH" -name "predictions.jsonl" -newer "$SENTINEL" -type f 2>/dev/null | head -1)
RESULT_COUNT=0
if [ -n "${RESULT_FILE:-}" ] && [ -f "$RESULT_FILE" ]; then
    RESULT_COUNT=$(wc -l < "$RESULT_FILE")
    if [ "$RESULT_COUNT" -ne "$EXPECTED_COUNT" ]; then
        echo "OUTPUT VERIFICATION FAILED: predictions.jsonl has $RESULT_COUNT lines, expected $EXPECTED_COUNT"
        EXIT_CODE=4
    else
        echo "Output verification PASSED: predictions.jsonl has $RESULT_COUNT lines"
    fi
else
    echo "OUTPUT VERIFICATION FAILED: predictions.jsonl not found at $LOG_PATH"
    EXIT_CODE=4
fi

{
    echo "end_time=$(date -Iseconds)"
    echo "exit_code=$EXIT_CODE"
    echo "actual_transformers=$ACTUAL_TRANSFORMERS"
    echo "result_file=${RESULT_FILE:-(none)}"
    echo "result_count=$RESULT_COUNT"
    echo "args_json=${ARGS_JSON:-(none)}"
} >> "$META"

echo ""
echo "=== output directory listing ==="
ls -la "$LOG_PATH"
echo "Result file: ${RESULT_FILE:-(none)}"
echo "Entry count: $RESULT_COUNT  (expected $EXPECTED_COUNT)"

# Cleanup sentinel
rm -f "$SENTINEL"

exit $EXIT_CODE
