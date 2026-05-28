#!/bin/bash
#SBATCH --job-name=mmhal-van
#SBATCH --partition=gpuq
#SBATCH --qos=gpu
#SBATCH --gres=gpu:A100.80gb:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/mmarvani/LVLM/lvlm-logs/mmhal_vanilla_%j.out
#SBATCH --error=/scratch/mmarvani/LVLM/lvlm-logs/mmhal_vanilla_%j.err

module load python
source /scratch/mmarvani/lvlm-env/bin/activate

export HF_HOME=/scratch/mmarvani/.cache/huggingface
export TRANSFORMERS_CACHE=/scratch/mmarvani/.cache/huggingface
export TORCH_HOME=/scratch/mmarvani/.cache/torch
export TMPDIR=/scratch/mmarvani/tmp
export PYTHONPATH=/scratch/mmarvani/LVLM/CAAC:$PYTHONPATH

cd /scratch/mmarvani/LVLM/CAAC

# ---- LLaVA-NeXT Vanilla ----
echo "===== LLaVA-NeXT MMHal Vanilla ====="
python experiments/run_mmhal.py \
    --model_type llava-next \
    --mmhal_path /scratch/mmarvani/LVLM/datasets/mmhal-bench \
    --output_file /scratch/mmarvani/LVLM/lvlm-logs/MMHal/llava-next_vanilla.json \
    --load_in_8bit
echo "===== LLaVA-NeXT complete ====="

# ---- LLaVA-1.5 Vanilla ----
echo "===== LLaVA-1.5 MMHal Vanilla ====="
python experiments/run_mmhal.py \
    --model_type llava \
    --mmhal_path /scratch/mmarvani/LVLM/datasets/mmhal-bench \
    --output_file /scratch/mmarvani/LVLM/lvlm-logs/MMHal/llava_vanilla.json \
    --load_in_8bit
echo "===== LLaVA-1.5 complete ====="

# ---- InstructBLIP Vanilla ----
echo "===== InstructBLIP MMHal Vanilla ====="
python experiments/run_mmhal.py \
    --model_type instructblip \
    --mmhal_path /scratch/mmarvani/LVLM/datasets/mmhal-bench \
    --output_file /scratch/mmarvani/LVLM/lvlm-logs/MMHal/instructblip_vanilla.json \
    --load_in_8bit
echo "===== InstructBLIP complete ====="

echo "All MMHal vanilla runs complete."
