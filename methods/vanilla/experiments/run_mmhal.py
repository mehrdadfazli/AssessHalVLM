import os
import json
import torch
import argparse
import logging
from PIL import Image
from tqdm import tqdm
from datetime import datetime
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.model_utils import load_model_and_processor, process_inputs

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser(description="Generate responses for MMHal-Bench")
parser.add_argument("--model_type", type=str, choices=["llava", "llava-next", "instructblip"], required=True)
parser.add_argument("--cache_dir", type=str, default="/path/to/.cache/huggingface")
parser.add_argument("--mmhal_path", type=str, default="/path/to/LVLM/datasets/mmhal-bench")
parser.add_argument("--output_file", type=str, required=True)
parser.add_argument("--load_in_8bit", action="store_true", default=True)
parser.add_argument("--max_new_tokens", type=int, default=512)
parser.add_argument("--gpu_id", type=int, default=0)
args = parser.parse_args()

model_names = {
    "llava-next": "llava-hf/llava-v1.6-vicuna-7b-hf",
    "llava": "llava-hf/llava-1.5-7b-hf",
    "instructblip": "Salesforce/instructblip-vicuna-7b"
}

os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")

# Load model
logger.info(f"Loading model: {args.model_type}")
model, processor = load_model_and_processor(args.model_type, model_names, args.cache_dir, device, args.load_in_8bit)
model.eval()
logger.info("Model loaded successfully")

# Load MMHal-Bench data
with open(os.path.join(args.mmhal_path, "response_template.json")) as f:
    data = json.load(f)
logger.info(f"Loaded {len(data)} questions")

# Generate responses
for item in tqdm(data, desc="Generating responses"):
    image_filename = item["image_src"].split("/")[-1]
    image_path = os.path.join(args.mmhal_path, "images", image_filename)

    if not os.path.exists(image_path):
        logger.warning(f"Image {image_path} not found, skipping")
        item["model_answer"] = ""
        continue

    try:
        raw_image = Image.open(image_path).convert("RGB")
        question = item["question"]
        inputs = process_inputs(raw_image, question, processor, args.model_type)

        generated_ids = model.generate(
            **inputs,
            do_sample=False,
            max_new_tokens=args.max_new_tokens,
            num_beams=1
        )
        response = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()

        # Clean up response for LLaVA models
        if "ASSISTANT: " in response:
            response = response.split("ASSISTANT: ")[-1]

        item["model_answer"] = response
        torch.cuda.empty_cache()

    except Exception as e:
        logger.error(f"Error processing {image_filename}: {e}")
        item["model_answer"] = ""
        continue

# Save responses
os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
with open(args.output_file, "w") as f:
    json.dump(data, f, indent=2)

logger.info(f"Saved responses to {args.output_file}")
