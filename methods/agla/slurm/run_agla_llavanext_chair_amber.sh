#!/bin/bash
#SBATCH --job-name=agla-next
#SBATCH --partition=gpuq
#SBATCH --qos=gpu
#SBATCH --gres=gpu:A100.80gb:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --time=06:00:00
#SBATCH --output=/path/to/LVLM/lvlm-logs/agla_llavanext_%j.out
#SBATCH --error=/path/to/LVLM/lvlm-logs/agla_llavanext_%j.err

module load python
source /path/to/lvlm-env/bin/activate

export HF_HOME=/path/to/.cache/huggingface
export TRANSFORMERS_CACHE=/path/to/.cache/huggingface
export TORCH_HOME=/path/to/.cache/torch
export TMPDIR=/path/to/tmp
export PYTHONPATH=/path/to/LVLM/CAAC:$PYTHONPATH

cd /path/to/LVLM/CAAC

# ---- AGLA + LLaVA-NeXT on CHAIR (500 images) ----
echo "===== AGLA + LLaVA-NeXT on CHAIR ====="
python experiments/run_agla_llavanext.py \
    --question_file /path/to/LVLM/AGLA/data/CHAIR/chair_500.jsonl \
    --image_folder /path/to/LVLM/datasets/coco2014/val2014 \
    --augmented_folder /path/to/LVLM/lvlm-logs/AGLA/augmented_chair \
    --output_file /path/to/LVLM/lvlm-logs/AGLA/llavanext_chair_agla_500.jsonl \
    --alpha 2 --beta 0.5 --seed 42 --load_in_8bit
echo "===== CHAIR complete ====="

# ---- AGLA + LLaVA-NeXT on AMBER (1004 images) ----
echo "===== AGLA + LLaVA-NeXT on AMBER ====="
python experiments/run_agla_llavanext.py \
    --question_file /path/to/LVLM/AGLA/data/AMBER/amber_generative.jsonl \
    --image_folder /path/to/LVLM/datasets/AMBER/image \
    --augmented_folder /path/to/LVLM/lvlm-logs/AGLA/augmented_amber \
    --output_file /path/to/LVLM/lvlm-logs/AGLA/llavanext_amber_agla.jsonl \
    --alpha 2 --beta 0.5 --seed 42 --load_in_8bit
echo "===== AMBER complete ====="

echo "All AGLA LLaVA-NeXT runs complete."
