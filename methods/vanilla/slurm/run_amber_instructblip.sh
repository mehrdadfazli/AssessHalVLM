#!/bin/bash
#SBATCH --job-name=amber-iblip
#SBATCH --partition=gpuq
#SBATCH --qos=gpu
#SBATCH --gres=gpu:A100.80gb:1
#SBATCH --mem=72G
#SBATCH --cpus-per-task=4
#SBATCH --time=6:00:00
#SBATCH --output=/scratch/mmarvani/LVLM/lvlm-logs/amber_iblip_%j.out
#SBATCH --error=/scratch/mmarvani/LVLM/lvlm-logs/amber_iblip_%j.err

module load python
source /scratch/mmarvani/lvlm-env/bin/activate

export HF_HOME=/scratch/mmarvani/.cache/huggingface
export TRANSFORMERS_CACHE=/scratch/mmarvani/.cache/huggingface
export TORCH_HOME=/scratch/mmarvani/.cache/torch
export TMPDIR=/scratch/mmarvani/tmp
export PYTHONPATH=/scratch/mmarvani/LVLM/CAAC:$PYTHONPATH
export NLTK_DATA=/scratch/mmarvani/.cache/nltk_data

cd /scratch/mmarvani/LVLM/CAAC

python experiments/run_amber_baselines.py \
    --model_type instructblip \
    --cache_dir /scratch/mmarvani/.cache/huggingface \
    --amber_path /scratch/mmarvani/LVLM/datasets/AMBER \
    --log_dir /scratch/mmarvani/LVLM/lvlm-logs/AMBER/baselines_instructblip \
    --max_new_tokens 512 \
    --num_beams 1 \
    --load_in_8bit
