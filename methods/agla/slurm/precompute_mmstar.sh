#!/bin/bash
#SBATCH --job-name=agla-aug-ms
#SBATCH --partition=gpuq
#SBATCH --qos=gpu
#SBATCH --gres=gpu:A100.80gb:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --time=04:00:00
#SBATCH --output=/scratch/mmarvani/LVLM/lvlm-logs/agla_aug_mmstar_%j.out
#SBATCH --error=/scratch/mmarvani/LVLM/lvlm-logs/agla_aug_mmstar_%j.err

module load python
source /scratch/mmarvani/LVLM/envs/agla-env/bin/activate

export HF_HOME=/scratch/mmarvani/.cache/huggingface
export TORCH_HOME=/scratch/mmarvani/.cache/torch
export TMPDIR=/scratch/mmarvani/tmp
export PYTHONPATH=/scratch/mmarvani/LVLM/AGLA:$PYTHONPATH

cd /scratch/mmarvani/LVLM/AGLA/eval

python precompute_augmentation.py \
    --question_file /scratch/mmarvani/LVLM/datasets/MMStar/mmstar_inputs.jsonl \
    --image_folder /scratch/mmarvani/LVLM/datasets/MMStar/images \
    --output_dir /scratch/mmarvani/LVLM/lvlm-logs/AGLA/augmented_mmstar \
    --image_key img_path \
    --question_key question

echo "Done."
