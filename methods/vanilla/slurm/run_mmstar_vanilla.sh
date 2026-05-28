#!/bin/bash
#SBATCH --job-name=mmstar-van
#SBATCH --partition=gpuq
#SBATCH --qos=gpu
#SBATCH --gres=gpu:A100.80gb:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --time=06:00:00
#SBATCH --output=/scratch/mmarvani/LVLM/lvlm-logs/mmstar_vanilla_%j.out
#SBATCH --error=/scratch/mmarvani/LVLM/lvlm-logs/mmstar_vanilla_%j.err

module load python
source /scratch/mmarvani/lvlm-env/bin/activate

export HF_HOME=/scratch/mmarvani/.cache/huggingface
export TRANSFORMERS_CACHE=/scratch/mmarvani/.cache/huggingface
export TORCH_HOME=/scratch/mmarvani/.cache/torch
export TMPDIR=/scratch/mmarvani/tmp
export PYTHONPATH=/scratch/mmarvani/LVLM/CAAC:$PYTHONPATH

cd /scratch/mmarvani/LVLM/CAAC

echo "===== LLaVA-NeXT MMStar Vanilla ====="
python experiments/run_mmstar.py \
    --model_type llava-next \
    --output_file /scratch/mmarvani/LVLM/lvlm-logs/MMStar/llava-next_vanilla.jsonl \
    --load_in_8bit

echo "===== LLaVA-1.5 MMStar Vanilla ====="
python experiments/run_mmstar.py \
    --model_type llava \
    --output_file /scratch/mmarvani/LVLM/lvlm-logs/MMStar/llava_vanilla.jsonl \
    --load_in_8bit

echo "===== InstructBLIP MMStar Vanilla ====="
python experiments/run_mmstar.py \
    --model_type instructblip \
    --output_file /scratch/mmarvani/LVLM/lvlm-logs/MMStar/instructblip_vanilla.jsonl \
    --load_in_8bit

echo "All done."
