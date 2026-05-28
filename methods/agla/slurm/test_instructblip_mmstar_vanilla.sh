#!/bin/bash
#SBATCH --job-name=iblip-test
#SBATCH --partition=gpuq
#SBATCH --qos=gpu
#SBATCH --gres=gpu:A100.80gb:1
#SBATCH --mem=48G
#SBATCH --cpus-per-task=4
#SBATCH --time=01:00:00
#SBATCH --output=/scratch/mmarvani/LVLM/lvlm-logs/iblip_mmstar_test_%j.out
#SBATCH --error=/scratch/mmarvani/LVLM/lvlm-logs/iblip_mmstar_test_%j.err

module load python
source /scratch/mmarvani/LVLM/envs/agla-env/bin/activate

export HF_HOME=/scratch/mmarvani/.cache/huggingface
export TORCH_HOME=/scratch/mmarvani/.cache/torch
export TMPDIR=/scratch/mmarvani/tmp
export PYTHONPATH=/scratch/mmarvani/LVLM/AGLA:$PYTHONPATH

cd /scratch/mmarvani/LVLM/AGLA/eval

python -c "
import json, os, sys, torch
sys.path.append('..')
from lavis.models import load_model_and_preprocess
from PIL import Image
from sample import evolve_agla_sampling
evolve_agla_sampling()

device = 'cuda'
model, vis_processors, _ = load_model_and_preprocess(
    name='blip2_vicuna_instruct', model_type='vicuna7b', is_eval=True, device=device
)

with open('/scratch/mmarvani/LVLM/datasets/MMStar/mmstar_inputs.jsonl') as f:
    data = [json.loads(l) for l in f][:50]

aug_dir = '/scratch/mmarvani/LVLM/lvlm-logs/AGLA/augmented_mmstar'

configs = [
    {'name': 'Vanilla (no AGLA, max_length=512)', 'use_agla': False, 'max_len': 512},
    {'name': 'Vanilla (no AGLA, max_length=1024)', 'use_agla': False, 'max_len': 1024},
    {'name': 'AGLA (alpha=2, max_length=1024)', 'use_agla': True, 'max_len': 1024},
]

for cfg in configs:
    correct = 0
    empty = 0
    for item in data:
        img_path = item['img_path']
        raw_image = Image.open(img_path).convert('RGB')
        image_tensor = vis_processors['eval'](raw_image).unsqueeze(0).to(device)

        if cfg['use_agla']:
            aug_path = os.path.join(aug_dir, img_path.split('/')[-1])
            if os.path.exists(aug_path):
                aug_image = Image.open(aug_path).convert('RGB')
                image_tensor_cd = vis_processors['eval'](aug_image).unsqueeze(0).to(device)
            else:
                image_tensor_cd = None
        else:
            image_tensor_cd = None

        with torch.inference_mode():
            outputs = model.generate(
                {'image': image_tensor, 'prompt': item['prompt']},
                use_nucleus_sampling=False,
                num_beams=1,
                top_p=1.0,
                repetition_penalty=1,
                images_cd=image_tensor_cd if cfg['use_agla'] else None,
                cd_alpha=2.0 if cfg['use_agla'] else 0,
                cd_beta=0.5 if cfg['use_agla'] else 0,
                temperature=1.0,
                max_length=cfg['max_len']
            )

        resp = outputs[0].strip()
        if len(resp) < 5 or resp in ['<s>', '</s>']:
            empty += 1
        if item['answer'].upper() in resp[:10].upper():
            correct += 1
        torch.cuda.empty_cache()

    print(f'{cfg[\"name\"]}: {correct}/50 correct, {empty}/50 empty')
"
