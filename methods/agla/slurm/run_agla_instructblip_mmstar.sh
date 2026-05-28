#!/bin/bash
#SBATCH --job-name=agla-ib-ms2
#SBATCH --partition=gpuq
#SBATCH --qos=gpu
#SBATCH --gres=gpu:A100.80gb:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --time=12:00:00
#SBATCH --output=/scratch/mmarvani/LVLM/lvlm-logs/agla_ib_mmstar2_%j.out
#SBATCH --error=/scratch/mmarvani/LVLM/lvlm-logs/agla_ib_mmstar2_%j.err

module load python
source /scratch/mmarvani/lvlm-env/bin/activate

export HF_HOME=/scratch/mmarvani/.cache/huggingface
export TRANSFORMERS_CACHE=/scratch/mmarvani/.cache/huggingface
export TORCH_HOME=/scratch/mmarvani/.cache/torch
export TMPDIR=/scratch/mmarvani/tmp
export PYTHONPATH=/scratch/mmarvani/LVLM/CAAC:$PYTHONPATH

cd /scratch/mmarvani/LVLM/CAAC

python experiments/run_agla_instructblip_mmstar.py \
    --input_file /scratch/mmarvani/LVLM/datasets/MMStar/mmstar_inputs.jsonl \
    --augmented_folder /scratch/mmarvani/LVLM/lvlm-logs/AGLA/augmented_mmstar \
    --output_file /scratch/mmarvani/LVLM/lvlm-logs/MMStar/instructblip_agla_hf.jsonl \
    --alpha 2 --beta 0.5 --seed 42 --load_in_8bit

echo "Complete."
