#!/bin/bash
#SBATCH --job-name=mmstar-van
#SBATCH --partition=gpuq
#SBATCH --qos=gpu
#SBATCH --gres=gpu:A100.80gb:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --time=06:00:00
#SBATCH --output=/path/to/LVLM/lvlm-logs/mmstar_vanilla_%j.out
#SBATCH --error=/path/to/LVLM/lvlm-logs/mmstar_vanilla_%j.err

module load python
source /path/to/lvlm-env/bin/activate

export HF_HOME=/path/to/.cache/huggingface
export TRANSFORMERS_CACHE=/path/to/.cache/huggingface
export TORCH_HOME=/path/to/.cache/torch
export TMPDIR=/path/to/tmp
export PYTHONPATH=/path/to/LVLM/CAAC:$PYTHONPATH

cd /path/to/LVLM/CAAC

echo "===== LLaVA-NeXT MMStar Vanilla ====="
python experiments/run_mmstar.py \
    --model_type llava-next \
    --output_file /path/to/LVLM/lvlm-logs/MMStar/llava-next_vanilla.jsonl \
    --load_in_8bit

echo "===== LLaVA-1.5 MMStar Vanilla ====="
python experiments/run_mmstar.py \
    --model_type llava \
    --output_file /path/to/LVLM/lvlm-logs/MMStar/llava_vanilla.jsonl \
    --load_in_8bit

echo "===== InstructBLIP MMStar Vanilla ====="
python experiments/run_mmstar.py \
    --model_type instructblip \
    --output_file /path/to/LVLM/lvlm-logs/MMStar/instructblip_vanilla.jsonl \
    --load_in_8bit

echo "All done."
