#!/bin/bash
#SBATCH --job-name=agla-aug-ms
#SBATCH --partition=gpuq
#SBATCH --qos=gpu
#SBATCH --gres=gpu:A100.80gb:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --time=04:00:00
#SBATCH --output=/path/to/LVLM/lvlm-logs/agla_aug_mmstar_%j.out
#SBATCH --error=/path/to/LVLM/lvlm-logs/agla_aug_mmstar_%j.err

module load python
source /path/to/LVLM/envs/agla-env/bin/activate

export HF_HOME=/path/to/.cache/huggingface
export TORCH_HOME=/path/to/.cache/torch
export TMPDIR=/path/to/tmp
export PYTHONPATH=/path/to/LVLM/AGLA:$PYTHONPATH

cd /path/to/LVLM/AGLA/eval

python precompute_augmentation.py \
    --question_file /path/to/LVLM/datasets/MMStar/mmstar_inputs.jsonl \
    --image_folder /path/to/LVLM/datasets/MMStar/images \
    --output_dir /path/to/LVLM/lvlm-logs/AGLA/augmented_mmstar \
    --image_key img_path \
    --question_key question

echo "Done."
