#!/bin/bash
#SBATCH --job-name=amber-next
#SBATCH --partition=gpuq
#SBATCH --qos=gpu
#SBATCH --gres=gpu:A100.80gb:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --time=1-12:00:00
#SBATCH --output=/scratch/mmarvani/LVLM/lvlm-logs/amber_llavanext_%j.out
#SBATCH --error=/scratch/mmarvani/LVLM/lvlm-logs/amber_llavanext_%j.err

module load python
source /scratch/mmarvani/lvlm-env/bin/activate

export HF_HOME=/scratch/mmarvani/.cache/huggingface
export TRANSFORMERS_CACHE=/scratch/mmarvani/.cache/huggingface
export TORCH_HOME=/scratch/mmarvani/.cache/torch
export TMPDIR=/scratch/mmarvani/tmp
export PYTHONPATH=/scratch/mmarvani/LVLM/CAAC:$PYTHONPATH
export NLTK_DATA=/scratch/mmarvani/.cache/nltk_data

cd /scratch/mmarvani/LVLM/CAAC

# ---- Vanilla ----
echo "===== LLaVA-NeXT AMBER Vanilla ====="
python experiments/run_amber_baselines.py \
    --model_type llava-next \
    --cache_dir /scratch/mmarvani/.cache/huggingface \
    --amber_path /scratch/mmarvani/LVLM/datasets/AMBER \
    --log_dir /scratch/mmarvani/LVLM/lvlm-logs/AMBER/baselines \
    --max_new_tokens 512 \
    --num_beams 1 \
    --load_in_8bit
echo "===== Vanilla complete ====="

# ---- VCD ----
echo "===== LLaVA-NeXT AMBER VCD ====="
python experiments/run_amber_baselines.py \
    --model_type llava-next \
    --cache_dir /scratch/mmarvani/.cache/huggingface \
    --amber_path /scratch/mmarvani/LVLM/datasets/AMBER \
    --log_dir /scratch/mmarvani/LVLM/lvlm-logs/AMBER/baselines \
    --max_new_tokens 512 \
    --num_beams 1 \
    --load_in_8bit \
    --use_VCD
echo "===== VCD complete ====="

# ---- M3ID ----
echo "===== LLaVA-NeXT AMBER M3ID ====="
python experiments/run_amber_baselines.py \
    --model_type llava-next \
    --cache_dir /scratch/mmarvani/.cache/huggingface \
    --amber_path /scratch/mmarvani/LVLM/datasets/AMBER \
    --log_dir /scratch/mmarvani/LVLM/lvlm-logs/AMBER/baselines \
    --max_new_tokens 512 \
    --num_beams 1 \
    --load_in_8bit \
    --use_M3ID
echo "===== M3ID complete ====="

echo "All LLaVA-NeXT AMBER runs complete."
