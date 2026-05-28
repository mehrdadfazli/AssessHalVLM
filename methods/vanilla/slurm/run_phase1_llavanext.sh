#!/bin/bash
#SBATCH --job-name=phase1-next
#SBATCH --partition=gpuq
#SBATCH --qos=gpu
#SBATCH --gres=gpu:A100.80gb:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --time=12:00:00
#SBATCH --output=/path/to/LVLM/lvlm-logs/phase1_llavanext_OUTPUT_2.out
#SBATCH --error=/path/to/LVLM/lvlm-logs/phase1_llavanext_ERROR_2.err

module load python
source /path/to/lvlm-env/bin/activate

export HF_HOME=/path/to/.cache/huggingface
export TRANSFORMERS_CACHE=/path/to/.cache/huggingface
export TORCH_HOME=/path/to/.cache/torch
export TMPDIR=/path/to/tmp
export PYTHONPATH=/path/to/LVLM/CAAC:$PYTHONPATH
export NLTK_DATA=/path/to/.cache/nltk_data

cd /path/to/LVLM/CAAC

# ---- Run 1: Vanilla baseline ----
echo "===== Starting LLaVA-NeXT vanilla baseline ====="
python experiments/run_chair_baselines.py \
    --model_type llava-next \
    --cache_dir /path/to/.cache/huggingface \
    --data_path /path/to/LVLM/datasets/coco2014/val2014 \
    --log_dir /path/to/LVLM/lvlm-logs/CHAIR/baselines \
    --num_images 500 \
    --max_new_tokens 512 \
    --num_beams 1 \
    --random_seed 42 \
    --load_in_8bit
echo "===== Vanilla baseline complete ====="

# ---- Run 2: VCD ----
echo "===== Starting LLaVA-NeXT VCD ====="
python experiments/run_chair_baselines.py \
    --model_type llava-next \
    --cache_dir /path/to/.cache/huggingface \
    --data_path /path/to/LVLM/datasets/coco2014/val2014 \
    --log_dir /path/to/LVLM/lvlm-logs/CHAIR/baselines \
    --num_images 500 \
    --max_new_tokens 512 \
    --num_beams 1 \
    --random_seed 42 \
    --load_in_8bit \
    --use_VCD
echo "===== VCD complete ====="

# ---- Run 3: M3ID ----
echo "===== Starting LLaVA-NeXT M3ID ====="
python experiments/run_chair_baselines.py \
    --model_type llava-next \
    --cache_dir /path/to/.cache/huggingface \
    --data_path /path/to/LVLM/datasets/coco2014/val2014 \
    --log_dir /path/to/LVLM/lvlm-logs/CHAIR/baselines \
    --num_images 500 \
    --max_new_tokens 512 \
    --num_beams 1 \
    --random_seed 42 \
    --load_in_8bit \
    --use_M3ID
echo "===== M3ID complete ====="

echo "All LlaVA-NeXT Phase 1 runs complete."
