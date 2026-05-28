#!/bin/bash
#SBATCH --job-name=agla-iblip
#SBATCH --partition=gpuq
#SBATCH --qos=gpu
#SBATCH --gres=gpu:A100.80gb:1
#SBATCH --mem=48G
#SBATCH --cpus-per-task=4
#SBATCH --time=04:00:00
#SBATCH --output=/path/to/LVLM/lvlm-logs/agla_iblip_chair_%j.out
#SBATCH --error=/path/to/LVLM/lvlm-logs/agla_iblip_chair_%j.err

module load python
source /path/to/LVLM/envs/agla-env/bin/activate

export HF_HOME=/path/to/.cache/huggingface
export TORCH_HOME=/path/to/.cache/torch
export TMPDIR=/path/to/tmp
export PYTHONPATH=/path/to/LVLM/AGLA:$PYTHONPATH

cd /path/to/LVLM/AGLA/eval

echo "===== AGLA + InstructBLIP on CHAIR (500 images) ====="
python run_instructblip_chair.py \
    --image-folder /path/to/LVLM/datasets/coco2014/val2014 \
    --question-file ../data/CHAIR/chair_500.jsonl \
    --answers-file /path/to/LVLM/lvlm-logs/AGLA/instructblip_chair_agla_500.jsonl \
    --use_agla \
    --alpha 2 \
    --beta 0.5 \
    --seed 42

echo "===== Complete ====="
