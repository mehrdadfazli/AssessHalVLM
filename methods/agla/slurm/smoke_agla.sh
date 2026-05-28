#!/bin/bash
#SBATCH --job-name=agla-smoke
#SBATCH --partition=gpuq
#SBATCH --qos=gpu
#SBATCH --gres=gpu:A100.80gb:1
#SBATCH --mem=48G
#SBATCH --cpus-per-task=4
#SBATCH --time=01:00:00
#SBATCH --output=/scratch/mmarvani/LVLM/lvlm-logs/agla_smoke_%j.out
#SBATCH --error=/scratch/mmarvani/LVLM/lvlm-logs/agla_smoke_%j.err

module load python
source /scratch/mmarvani/LVLM/envs/agla-env/bin/activate

export HF_HOME=/scratch/mmarvani/.cache/huggingface
export TORCH_HOME=/scratch/mmarvani/.cache/torch
export TMPDIR=/scratch/mmarvani/tmp
export PYTHONPATH=/scratch/mmarvani/LVLM/AGLA:$PYTHONPATH

cd /scratch/mmarvani/LVLM/AGLA/eval

# Create a 5-image subset for smoke test
head -5 ../data/CHAIR/chair_smoke.jsonl 2>/dev/null || head -5 ../data/CHAIR/chair.jsonl > ../data/CHAIR/chair_smoke.jsonl

python run_llava_chair.py \
    --model-path liuhaotian/llava-v1.5-7b \
    --image-folder /scratch/mmarvani/LVLM/datasets/coco2014/val2014 \
    --question-file ../data/CHAIR/chair_smoke.jsonl \
    --answers-file /scratch/mmarvani/LVLM/lvlm-logs/AGLA/smoke_test.jsonl \
    --use_agla \
    --alpha 2 \
    --beta 0.5 \
    --seed 42

echo "AGLA smoke test complete."
