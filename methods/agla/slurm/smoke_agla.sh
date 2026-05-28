#!/bin/bash
#SBATCH --job-name=agla-smoke
#SBATCH --partition=gpuq
#SBATCH --qos=gpu
#SBATCH --gres=gpu:A100.80gb:1
#SBATCH --mem=48G
#SBATCH --cpus-per-task=4
#SBATCH --time=01:00:00
#SBATCH --output=/path/to/LVLM/lvlm-logs/agla_smoke_%j.out
#SBATCH --error=/path/to/LVLM/lvlm-logs/agla_smoke_%j.err

module load python
source /path/to/LVLM/envs/agla-env/bin/activate

export HF_HOME=/path/to/.cache/huggingface
export TORCH_HOME=/path/to/.cache/torch
export TMPDIR=/path/to/tmp
export PYTHONPATH=/path/to/LVLM/AGLA:$PYTHONPATH

cd /path/to/LVLM/AGLA/eval

# Create a 5-image subset for smoke test
head -5 ../data/CHAIR/chair_smoke.jsonl 2>/dev/null || head -5 ../data/CHAIR/chair.jsonl > ../data/CHAIR/chair_smoke.jsonl

python run_llava_chair.py \
    --model-path liuhaotian/llava-v1.5-7b \
    --image-folder /path/to/LVLM/datasets/coco2014/val2014 \
    --question-file ../data/CHAIR/chair_smoke.jsonl \
    --answers-file /path/to/LVLM/lvlm-logs/AGLA/smoke_test.jsonl \
    --use_agla \
    --alpha 2 \
    --beta 0.5 \
    --seed 42

echo "AGLA smoke test complete."
