#!/bin/bash
#SBATCH --job-name=agla-nx-ms
#SBATCH --partition=gpuq
#SBATCH --qos=gpu
#SBATCH --gres=gpu:A100.80gb:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --time=12:00:00
#SBATCH --output=/path/to/LVLM/lvlm-logs/agla_nx_mmstar_%j.out
#SBATCH --error=/path/to/LVLM/lvlm-logs/agla_nx_mmstar_%j.err

module load python
source /path/to/lvlm-env/bin/activate

export HF_HOME=/path/to/.cache/huggingface
export TRANSFORMERS_CACHE=/path/to/.cache/huggingface
export TORCH_HOME=/path/to/.cache/torch
export TMPDIR=/path/to/tmp
export PYTHONPATH=/path/to/LVLM/CAAC:$PYTHONPATH

cd /path/to/LVLM/CAAC

python experiments/run_agla_llavanext_mmstar.py \
    --input_file /path/to/LVLM/datasets/MMStar/mmstar_inputs.jsonl \
    --augmented_folder /path/to/LVLM/lvlm-logs/AGLA/augmented_mmstar \
    --output_file /path/to/LVLM/lvlm-logs/MMStar/llavanext_agla.jsonl \
    --alpha 2 --beta 0.5 --seed 42 --load_in_8bit

echo "Complete."
