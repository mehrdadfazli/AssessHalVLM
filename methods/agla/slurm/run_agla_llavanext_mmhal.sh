#!/bin/bash
#SBATCH --job-name=agla-nx-mm
#SBATCH --partition=gpuq
#SBATCH --qos=gpu
#SBATCH --gres=gpu:A100.80gb:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --time=02:00:00
#SBATCH --output=/path/to/LVLM/lvlm-logs/agla_nx_mmhal_%j.out
#SBATCH --error=/path/to/LVLM/lvlm-logs/agla_nx_mmhal_%j.err

module load python
source /path/to/lvlm-env/bin/activate

export HF_HOME=/path/to/.cache/huggingface
export TRANSFORMERS_CACHE=/path/to/.cache/huggingface
export TORCH_HOME=/path/to/.cache/torch
export TMPDIR=/path/to/tmp
export PYTHONPATH=/path/to/LVLM/CAAC:$PYTHONPATH

cd /path/to/LVLM/CAAC

python experiments/run_agla_llavanext_mmhal.py \
    --mmhal_path /path/to/LVLM/datasets/mmhal-bench \
    --augmented_folder /path/to/LVLM/lvlm-logs/AGLA/augmented_mmhal \
    --output_file /path/to/LVLM/lvlm-logs/MMHal/llavanext_agla.json \
    --alpha 2 --beta 0.5 --seed 42 --load_in_8bit

echo "Complete."
