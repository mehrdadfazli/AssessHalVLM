#!/bin/bash
#SBATCH --job-name=iblip-trv2
#SBATCH --partition=gpuq
#SBATCH --qos=gpu
#SBATCH --gres=gpu:A100.80gb:1
#SBATCH --mem=48G
#SBATCH --cpus-per-task=4
#SBATCH --time=01:00:00
#SBATCH --output=/scratch/mmarvani/LVLM/lvlm-logs/iblip_trunc_v2_%j.out
#SBATCH --error=/scratch/mmarvani/LVLM/lvlm-logs/iblip_trunc_v2_%j.err

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

def shorten_prompt(question):
    \"\"\"Reformat MCQ to be shorter — just question + letter options on one line.\"\"\"
    if 'Options:' in question:
        q_part = question.split('Options:')[0].strip()
        opts_part = question.split('Options:')[1].strip()
        # Shorten options to just letters
        short_opts = []
        for opt in opts_part.split(','):
            opt = opt.strip()
            if opt:
                short_opts.append(opt.split(':')[0].strip() + ':' + ':'.join(opt.split(':')[1:]).strip()[:30])
        return q_part + ' ' + ', '.join(short_opts)
    return question

def letter_only_prompt(question):
    \"\"\"Just ask for the letter.\"\"\"
    if 'Options:' in question:
        return question + ' Answer with just the letter.'
    return question

configs = [
    {'name': 'Full + Answer letter suffix', 'transform': letter_only_prompt, 'agla': False},
    {'name': 'Shortened options', 'transform': shorten_prompt, 'agla': False},
    {'name': 'Shortened options + AGLA', 'transform': shorten_prompt, 'agla': True},
    {'name': 'Full + greedy + no sampling', 'transform': lambda x: x, 'agla': False, 'greedy': True},
    {'name': 'Shortened + greedy', 'transform': shorten_prompt, 'agla': False, 'greedy': True},
]

aug_dir = '/scratch/mmarvani/LVLM/lvlm-logs/AGLA/augmented_mmstar'

for cfg in configs:
    correct = 0
    empty = 0
    sample_responses = []

    for item in data:
        img_path = item['img_path']
        raw_image = Image.open(img_path).convert('RGB')
        image_tensor = vis_processors['eval'](raw_image).unsqueeze(0).to(device)

        prompt = cfg['transform'](item['prompt'])

        if cfg.get('agla'):
            aug_path = os.path.join(aug_dir, img_path.split('/')[-1])
            if os.path.exists(aug_path):
                aug_image = Image.open(aug_path).convert('RGB')
                image_tensor_cd = vis_processors['eval'](aug_image).unsqueeze(0).to(device)
            else:
                image_tensor_cd = None
        else:
            image_tensor_cd = None

        use_sampling = not cfg.get('greedy', False)

        with torch.inference_mode():
            outputs = model.generate(
                {'image': image_tensor, 'prompt': prompt},
                use_nucleus_sampling=use_sampling,
                num_beams=1,
                top_p=1.0,
                repetition_penalty=1,
                images_cd=image_tensor_cd if cfg.get('agla') else None,
                cd_alpha=2.0 if cfg.get('agla') else 0,
                cd_beta=0.5 if cfg.get('agla') else 0,
                temperature=1.0,
                max_length=512
            )

        resp = outputs[0].strip()
        if len(resp) < 3 or resp in ['<s>', '</s>']:
            empty += 1
        gt = item['answer'].strip().upper()
        if gt in resp[:10].upper():
            correct += 1
        
        if len(sample_responses) < 3:
            sample_responses.append(f'    [{gt}] -> \"{resp[:80]}\"')
        
        torch.cuda.empty_cache()

    print(f'{cfg[\"name\"]}: {correct}/50 correct, {empty}/50 empty')
    for s in sample_responses:
        print(s)
    print()
"
