#!/bin/bash
#SBATCH --job-name=agla-mmhal
#SBATCH --partition=gpuq
#SBATCH --qos=gpu
#SBATCH --gres=gpu:A100.80gb:1
#SBATCH --mem=48G
#SBATCH --cpus-per-task=4
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/mmarvani/LVLM/lvlm-logs/agla_mmhal_%j.out
#SBATCH --error=/scratch/mmarvani/LVLM/lvlm-logs/agla_mmhal_%j.err

module load python
source /scratch/mmarvani/LVLM/envs/agla-env/bin/activate

export HF_HOME=/scratch/mmarvani/.cache/huggingface
export TORCH_HOME=/scratch/mmarvani/.cache/torch
export TMPDIR=/scratch/mmarvani/tmp
export PYTHONPATH=/scratch/mmarvani/LVLM/AGLA:$PYTHONPATH

cd /scratch/mmarvani/LVLM/AGLA/eval

# ---- AGLA + LLaVA-1.5 on MMHal ----
echo "===== AGLA + LLaVA-1.5 on MMHal ====="
python run_mmhal.py \
    --model_type llava \
    --model-path liuhaotian/llava-v1.5-7b \
    --mmhal_path /scratch/mmarvani/LVLM/datasets/mmhal-bench \
    --image-folder /scratch/mmarvani/LVLM/datasets/mmhal-bench/images \
    --output_file /scratch/mmarvani/LVLM/lvlm-logs/MMHal/llava_agla.json \
    --use_agla --alpha 2 --beta 0.5 --seed 42
echo "===== LLaVA-1.5 complete ====="

# ---- AGLA + InstructBLIP on MMHal ----
echo "===== AGLA + InstructBLIP on MMHal ====="
python run_mmhal.py \
    --model_type instructblip \
    --mmhal_path /scratch/mmarvani/LVLM/datasets/mmhal-bench \
    --image-folder /scratch/mmarvani/LVLM/datasets/mmhal-bench/images \
    --output_file /scratch/mmarvani/LVLM/lvlm-logs/MMHal/instructblip_agla.json \
    --use_agla --alpha 2 --beta 0.5 --seed 42
echo "===== InstructBLIP complete ====="

echo "All AGLA MMHal runs complete."
