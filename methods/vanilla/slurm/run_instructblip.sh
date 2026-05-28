#!/bin/bash
#SBATCH --job-name=iblip-chair
#SBATCH --partition=gpuq
#SBATCH --qos=gpu
#SBATCH --gres=gpu:A100.80gb:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --time=12:00:00
#SBATCH --output=/path/to/LVLM/lvlm-logs/iblip_chair_OUTPUT.out
#SBATCH --error=/path/to/LVLM/lvlm-logs/iblip_chair_ERROR.err

module load python
source /path/to/lvlm-env/bin/activate

export HF_HOME=/path/to/.cache/huggingface
export TRANSFORMERS_CACHE=/path/to/.cache/huggingface
export TORCH_HOME=/path/to/.cache/torch
export TMPDIR=/path/to/tmp
export PYTHONPATH=/path/to/LVLM/CAAC:$PYTHONPATH
export NLTK_DATA=/path/to/.cache/nltk_data

cd /path/to/LVLM/CAAC

# ---- Vanilla baseline ----
echo "===== InstructBLIP Vanilla ====="
python experiments/run_chair_baselines.py \
    --model_type instructblip \
    --cache_dir /path/to/.cache/huggingface \
    --data_path /path/to/LVLM/datasets/coco2014/val2014 \
    --log_dir /path/to/LVLM/lvlm-logs/CHAIR/baselines \
    --num_images 500 \
    --max_new_tokens 512 \
    --num_beams 1 \
    --random_seed 42 \
    --load_in_8bit
echo "===== Vanilla complete ====="

# ---- VCD ----
echo "===== InstructBLIP VCD ====="
python experiments/run_chair_baselines.py \
    --model_type instructblip \
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

# ---- M3ID ----
echo "===== InstructBLIP M3ID ====="
python experiments/run_chair_baselines.py \
    --model_type instructblip \
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

echo "All InstructBLIP CHAIR runs complete."w