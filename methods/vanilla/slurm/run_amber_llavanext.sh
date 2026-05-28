#!/bin/bash
#SBATCH --job-name=amber-next
#SBATCH --partition=gpuq
#SBATCH --qos=gpu
#SBATCH --gres=gpu:A100.80gb:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --time=1-12:00:00
#SBATCH --output=/path/to/LVLM/lvlm-logs/amber_llavanext_%j.out
#SBATCH --error=/path/to/LVLM/lvlm-logs/amber_llavanext_%j.err

module load python
source /path/to/lvlm-env/bin/activate

export HF_HOME=/path/to/.cache/huggingface
export TRANSFORMERS_CACHE=/path/to/.cache/huggingface
export TORCH_HOME=/path/to/.cache/torch
export TMPDIR=/path/to/tmp
export PYTHONPATH=/path/to/LVLM/CAAC:$PYTHONPATH
export NLTK_DATA=/path/to/.cache/nltk_data

cd /path/to/LVLM/CAAC

# ---- Vanilla ----
echo "===== LLaVA-NeXT AMBER Vanilla ====="
python experiments/run_amber_baselines.py \
    --model_type llava-next \
    --cache_dir /path/to/.cache/huggingface \
    --amber_path /path/to/LVLM/datasets/AMBER \
    --log_dir /path/to/LVLM/lvlm-logs/AMBER/baselines \
    --max_new_tokens 512 \
    --num_beams 1 \
    --load_in_8bit
echo "===== Vanilla complete ====="

# ---- VCD ----
echo "===== LLaVA-NeXT AMBER VCD ====="
python experiments/run_amber_baselines.py \
    --model_type llava-next \
    --cache_dir /path/to/.cache/huggingface \
    --amber_path /path/to/LVLM/datasets/AMBER \
    --log_dir /path/to/LVLM/lvlm-logs/AMBER/baselines \
    --max_new_tokens 512 \
    --num_beams 1 \
    --load_in_8bit \
    --use_VCD
echo "===== VCD complete ====="

# ---- M3ID ----
echo "===== LLaVA-NeXT AMBER M3ID ====="
python experiments/run_amber_baselines.py \
    --model_type llava-next \
    --cache_dir /path/to/.cache/huggingface \
    --amber_path /path/to/LVLM/datasets/AMBER \
    --log_dir /path/to/LVLM/lvlm-logs/AMBER/baselines \
    --max_new_tokens 512 \
    --num_beams 1 \
    --load_in_8bit \
    --use_M3ID
echo "===== M3ID complete ====="

echo "All LLaVA-NeXT AMBER runs complete."
