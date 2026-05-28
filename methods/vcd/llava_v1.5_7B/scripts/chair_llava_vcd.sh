#!/bin/bash
#SBATCH --partition=gpuq
#SBATCH --qos=gpu
#SBATCH --gres=gpu:A100.80gb:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --time=02:30:00
#SBATCH --job-name=chair-llava-vcd
#SBATCH --output=/scratch/smansou3/LVLM/lvlm-logs/CHAIR/llava_VCD/slurm-%j.log

set -euo pipefail

mkdir -p /scratch/smansou3/LVLM/lvlm-logs/CHAIR/llava_VCD

source /scratch/smansou3/avisc_env.sh
source /scratch/smansou3/avisc-env/bin/activate
cd /scratch/smansou3/LVLM/AvisC

LOGDIR=/scratch/smansou3/LVLM/lvlm-logs/CHAIR/llava_VCD
META=$LOGDIR/job_metadata_${SLURM_JOB_ID}.txt

{
    echo "job_id=$SLURM_JOB_ID"
    echo "job_name=$SLURM_JOB_NAME"
    echo "start_time=$(date -Iseconds)"
    echo "host=$(hostname)"
    echo "model=llava-1.5-7b"
    echo "method=VCD"
    echo "benchmark=CHAIR"
    echo "model_path=liuhaotian/llava-v1.5-7b"
    echo "data_path=/scratch/smansou3/LVLM/datasets/coco2014/val2014"
    echo "log_dir=$LOGDIR"
    echo "num_images=500"
    echo "seed=42"
    echo "max_token=512"
    echo "use_cd=True"
    echo "use_m3id=False"
    echo "use_avisc=False"
    echo "opera_results="
} > "$META"

echo "=== nvidia-smi ==="
nvidia-smi
echo ""
echo "=== /home PRE ==="
du -sh /home/smansou3/.cache/ /home/smansou3/.cache/huggingface/ /home/smansou3/.cache/torch/ 2>/dev/null || true
echo ""
echo "=== HF_HOME=$HF_HOME ==="
echo "=== job starting at $(date) ==="

EXIT_CODE=0
python experiments/cd_scripts/chair_eval_llava.py \
    --model-path liuhaotian/llava-v1.5-7b \
    --data_path /scratch/smansou3/LVLM/datasets/coco2014/val2014 \
    --opera_results "" \
    --log_dir "$LOGDIR" \
    --num_images 500 \
    --seed 42 \
    --max_token 512 \
    --use_cd True \
    --use_m3id False \
    --use_avisc False || EXIT_CODE=$?

echo "=== job done at $(date), python exit=$EXIT_CODE ==="
echo ""
echo "=== /home POST ==="
du -sh /home/smansou3/.cache/ /home/smansou3/.cache/huggingface/ /home/smansou3/.cache/torch/ 2>/dev/null || true
echo ""
echo "=== /scratch caches POST ==="
du -sh /scratch/smansou3/.cache/huggingface/ /scratch/smansou3/.cache/torch/ 2>/dev/null
echo ""

JSONL=$(find "$LOGDIR" -name "*_results.jsonl" -type f -printf '%T@ %p\n' 2>/dev/null | sort -rn | head -1 | cut -d' ' -f2-)
LINECOUNT=0
if [ -n "${JSONL:-}" ]; then
    LINECOUNT=$(wc -l < "$JSONL")
fi

{
    echo "end_time=$(date -Iseconds)"
    echo "exit_code=$EXIT_CODE"
    echo "output_jsonl=${JSONL:-(none)}"
    echo "line_count=$LINECOUNT"
} >> "$META"

echo "=== output verification ==="
ls -la "$LOGDIR"
echo "JSONL: ${JSONL:-(none)}"
echo "Line count: $LINECOUNT"
if [ -n "${JSONL:-}" ]; then
    echo "First line:"
    head -1 "$JSONL"
fi

exit $EXIT_CODE
