#!/bin/bash
#SBATCH --job-name=iblip-chair
#SBATCH --partition=gpuq
#SBATCH --qos=gpu
#SBATCH --gres=gpu:A100.80gb:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --time=12:00:00
#SBATCH --output=/scratch/mmarvani/LVLM/lvlm-logs/iblip_chair_OUTPUT.out
#SBATCH --error=/scratch/mmarvani/LVLM/lvlm-logs/iblip_chair_ERROR.err

module load python
source /scratch/mmarvani/lvlm-env/bin/activate

export HF_HOME=/scratch/mmarvani/.cache/huggingface
export TRANSFORMERS_CACHE=/scratch/mmarvani/.cache/huggingface
export TORCH_HOME=/scratch/mmarvani/.cache/torch
export TMPDIR=/scratch/mmarvani/tmp
export PYTHONPATH=/scratch/mmarvani/LVLM/CAAC:$PYTHONPATH
export NLTK_DATA=/scratch/mmarvani/.cache/nltk_data

cd /scratch/mmarvani/LVLM/CAAC

# ---- Vanilla baseline ----
echo "===== InstructBLIP Vanilla ====="
python experiments/run_chair_baselines.py \
    --model_type instructblip \
    --cache_dir /scratch/mmarvani/.cache/huggingface \
    --data_path /scratch/mmarvani/LVLM/datasets/coco2014/val2014 \
    --log_dir /scratch/mmarvani/LVLM/lvlm-logs/CHAIR/baselines \
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
    --cache_dir /scratch/mmarvani/.cache/huggingface \
    --data_path /scratch/mmarvani/LVLM/datasets/coco2014/val2014 \
    --log_dir /scratch/mmarvani/LVLM/lvlm-logs/CHAIR/baselines \
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
    --cache_dir /scratch/mmarvani/.cache/huggingface \
    --data_path /scratch/mmarvani/LVLM/datasets/coco2014/val2014 \
    --log_dir /scratch/mmarvani/LVLM/lvlm-logs/CHAIR/baselines \
    --num_images 500 \
    --max_new_tokens 512 \
    --num_beams 1 \
    --random_seed 42 \
    --load_in_8bit \
    --use_M3ID
echo "===== M3ID complete ====="

echo "All InstructBLIP CHAIR runs complete."w