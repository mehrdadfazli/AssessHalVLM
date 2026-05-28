#!/bin/bash
#SBATCH --job-name=iblip-trunc
#SBATCH --partition=gpuq
#SBATCH --qos=gpu
#SBATCH --gres=gpu:A100.80gb:1
#SBATCH --mem=48G
#SBATCH --cpus-per-task=4
#SBATCH --time=01:00:00
#SBATCH --output=/path/to/LVLM/lvlm-logs/iblip_trunc_test_%j.out
#SBATCH --error=/path/to/LVLM/lvlm-logs/iblip_trunc_test_%j.err

module load python
source /path/to/LVLM/envs/agla-env/bin/activate

export HF_HOME=/path/to/.cache/huggingface
export TORCH_HOME=/path/to/.cache/torch
export TMPDIR=/path/to/tmp
export PYTHONPATH=/path/to/LVLM/AGLA:$PYTHONPATH

cd /path/to/LVLM/AGLA/eval

python -c "
import json, os, sys, torch
sys.path.append('..')
from lavis.models import load_model_and_preprocess
from torchvision import transforms
from PIL import Image
from sample import evolve_agla_sampling
evolve_agla_sampling()

device = 'cuda'
model, vis_processors, _ = load_model_and_preprocess(
    name='blip2_vicuna_instruct', model_type='vicuna7b', is_eval=True, device=device
)

# Also load BLIP ITM for AGLA
model_itm, image_processors, text_processors = load_model_and_preprocess(
    'blip_image_text_matching', 'large', device=device, is_eval=True
)

with open('/path/to/LVLM/datasets/MMStar/mmstar_inputs.jsonl') as f:
    data = [json.loads(l) for l in f][:50]

aug_dir = '/path/to/LVLM/lvlm-logs/AGLA/augmented_mmstar'

def truncate_prompt(prompt, max_tokens=100):
    \"\"\"Keep only the last ~max_tokens tokens worth of text.\"\"\"
    words = prompt.split()
    # Approximate: 1 token ≈ 0.75 words
    max_words = int(max_tokens * 0.75)
    if len(words) > max_words:
        truncated = ' '.join(words[-max_words:])
        return truncated
    return prompt

# Test 3 configs
configs = [
    {'name': 'Full prompt, no AGLA', 'truncate': False, 'agla': False},
    {'name': 'Truncated 100 tokens, no AGLA', 'truncate': True, 'max_tok': 100, 'agla': False},
    {'name': 'Truncated 100 tokens, AGLA', 'truncate': True, 'max_tok': 100, 'agla': True},
    {'name': 'Truncated 150 tokens, no AGLA', 'truncate': True, 'max_tok': 150, 'agla': False},
    {'name': 'Truncated 150 tokens, AGLA', 'truncate': True, 'max_tok': 150, 'agla': True},
]

for cfg in configs:
    correct = 0
    empty = 0
    for item in data:
        img_path = item['img_path']
        raw_image = Image.open(img_path).convert('RGB')
        image_tensor = vis_processors['eval'](raw_image).unsqueeze(0).to(device)
        
        prompt = item['prompt']
        if cfg['truncate']:
            prompt = truncate_prompt(prompt, cfg['max_tok'])

        # AGLA augmented image
        if cfg['agla']:
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
                {'image': image_tensor, 'prompt': prompt},
                use_nucleus_sampling=True,
                num_beams=1,
                top_p=1.0,
                repetition_penalty=1,
                images_cd=image_tensor_cd if cfg['agla'] else None,
                cd_alpha=2.0 if cfg['agla'] else 0,
                cd_beta=0.5 if cfg['agla'] else 0,
                temperature=1.0,
                max_length=512
            )

        resp = outputs[0].strip()
        if len(resp) < 3 or resp in ['<s>', '</s>']:
            empty += 1
        gt = item['answer'].strip().upper()
        if gt in resp[:10].upper():
            correct += 1
        torch.cuda.empty_cache()

    print(f'{cfg[\"name\"]}: {correct}/50 correct, {empty}/50 empty')
"
