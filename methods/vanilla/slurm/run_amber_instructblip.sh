#!/bin/bash
#SBATCH --job-name=amber-iblip
#SBATCH --partition=gpuq
#SBATCH --qos=gpu
#SBATCH --gres=gpu:A100.80gb:1
#SBATCH --mem=72G
#SBATCH --cpus-per-task=4
#SBATCH --time=6:00:00
#SBATCH --output=/path/to/LVLM/lvlm-logs/amber_iblip_%j.out
#SBATCH --error=/path/to/LVLM/lvlm-logs/amber_iblip_%j.err

module load python
source /path/to/lvlm-env/bin/activate

export HF_HOME=/path/to/.cache/huggingface
export TRANSFORMERS_CACHE=/path/to/.cache/huggingface
export TORCH_HOME=/path/to/.cache/torch
export TMPDIR=/path/to/tmp
export PYTHONPATH=/path/to/LVLM/CAAC:$PYTHONPATH
export NLTK_DATA=/path/to/.cache/nltk_data

cd /path/to/LVLM/CAAC

python experiments/run_amber_baselines.py \
    --model_type instructblip \
    --cache_dir /path/to/.cache/huggingface \
    --amber_path /path/to/LVLM/datasets/AMBER \
    --log_dir /path/to/LVLM/lvlm-logs/AMBER/baselines_instructblip \
    --max_new_tokens 512 \
    --num_beams 1 \
    --load_in_8bit
