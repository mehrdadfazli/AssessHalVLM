#!/bin/bash
#SBATCH --job-name=agla-aug-mm
#SBATCH --partition=gpuq
#SBATCH --qos=gpu
#SBATCH --gres=gpu:A100.80gb:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --time=01:00:00
#SBATCH --output=/path/to/LVLM/lvlm-logs/agla_aug_mmhal_%j.out
#SBATCH --error=/path/to/LVLM/lvlm-logs/agla_aug_mmhal_%j.err

module load python
source /path/to/LVLM/envs/agla-env/bin/activate

export HF_HOME=/path/to/.cache/huggingface
export TORCH_HOME=/path/to/.cache/torch
export TMPDIR=/path/to/tmp
export PYTHONPATH=/path/to/LVLM/AGLA:$PYTHONPATH

cd /path/to/LVLM/AGLA/eval

python -c "
import json, os, sys, torch
from PIL import Image
from tqdm import tqdm
from torchvision import transforms
sys.path.append('..')
from lavis.models import load_model_and_preprocess
from augmentation import augmentation

device = 'cuda'
output_dir = '/path/to/LVLM/lvlm-logs/AGLA/augmented_mmhal'
os.makedirs(output_dir, exist_ok=True)

print('Loading BLIP ITM...')
model_itm, image_processors, text_processors = load_model_and_preprocess(
    'blip_image_text_matching', 'large', device=device, is_eval=True
)
loader = transforms.Compose([transforms.ToTensor()])

with open('/path/to/LVLM/datasets/mmhal-bench/response_template.json') as f:
    data = json.load(f)

print(f'Processing {len(data)} images...')
for item in tqdm(data):
    filename = item['image_src'].split('/')[-1]
    image_path = f'/path/to/LVLM/datasets/mmhal-bench/images/{filename}'
    output_path = f'{output_dir}/{filename}'

    if os.path.exists(output_path):
        continue
    if not os.path.exists(image_path):
        continue

    raw_image = Image.open(image_path).convert('RGB')
    question = item['question']

    tensor_image = loader(raw_image.resize((384, 384)))
    image = image_processors['eval'](raw_image).unsqueeze(0).to(device)
    q_text = text_processors['eval'](question)
    tokenized_text = model_itm.tokenizer(q_text, padding='longest', truncation=True, return_tensors='pt').to(device)

    augmented_image = augmentation(image, q_text, tensor_image, model_itm, tokenized_text, raw_image)
    augmented_image.save(output_path)
    torch.cuda.empty_cache()

print(f'Done. Saved to {output_dir}')
"
