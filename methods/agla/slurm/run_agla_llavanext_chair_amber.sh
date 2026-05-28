#!/bin/bash
#SBATCH --job-name=agla-next
#SBATCH --partition=gpuq
#SBATCH --qos=gpu
#SBATCH --gres=gpu:A100.80gb:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --time=06:00:00
#SBATCH --output=/scratch/mmarvani/LVLM/lvlm-logs/agla_llavanext_%j.out
#SBATCH --error=/scratch/mmarvani/LVLM/lvlm-logs/agla_llavanext_%j.err

module load python
source /scratch/mmarvani/lvlm-env/bin/activate

export HF_HOME=/scratch/mmarvani/.cache/huggingface
export TRANSFORMERS_CACHE=/scratch/mmarvani/.cache/huggingface
export TORCH_HOME=/scratch/mmarvani/.cache/torch
export TMPDIR=/scratch/mmarvani/tmp
export PYTHONPATH=/scratch/mmarvani/LVLM/CAAC:$PYTHONPATH

cd /scratch/mmarvani/LVLM/CAAC

# ---- AGLA + LLaVA-NeXT on CHAIR (500 images) ----
echo "===== AGLA + LLaVA-NeXT on CHAIR ====="
python experiments/run_agla_llavanext.py \
    --question_file /scratch/mmarvani/LVLM/AGLA/data/CHAIR/chair_500.jsonl \
    --image_folder /scratch/mmarvani/LVLM/datasets/coco2014/val2014 \
    --augmented_folder /scratch/mmarvani/LVLM/lvlm-logs/AGLA/augmented_chair \
    --output_file /scratch/mmarvani/LVLM/lvlm-logs/AGLA/llavanext_chair_agla_500.jsonl \
    --alpha 2 --beta 0.5 --seed 42 --load_in_8bit
echo "===== CHAIR complete ====="

# ---- AGLA + LLaVA-NeXT on AMBER (1004 images) ----
echo "===== AGLA + LLaVA-NeXT on AMBER ====="
python experiments/run_agla_llavanext.py \
    --question_file /scratch/mmarvani/LVLM/AGLA/data/AMBER/amber_generative.jsonl \
    --image_folder /scratch/mmarvani/LVLM/datasets/AMBER/image \
    --augmented_folder /scratch/mmarvani/LVLM/lvlm-logs/AGLA/augmented_amber \
    --output_file /scratch/mmarvani/LVLM/lvlm-logs/AGLA/llavanext_amber_agla.jsonl \
    --alpha 2 --beta 0.5 --seed 42 --load_in_8bit
echo "===== AMBER complete ====="

echo "All AGLA LLaVA-NeXT runs complete."
