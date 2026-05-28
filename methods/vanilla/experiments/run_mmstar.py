"""
Generate responses for MMStar benchmark.
Supports LLaVA-1.5, LLaVA-NeXT, and InstructBLIP.
"""
import os
import json
import torch
import argparse
import logging
from PIL import Image
from tqdm import tqdm
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.model_utils import load_model_and_processor, process_inputs

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser(description="Generate responses for MMStar")
parser.add_argument("--model_type", type=str, choices=["llava", "llava-next", "instructblip"], required=True)
parser.add_argument("--cache_dir", type=str, default="/scratch/mmarvani/.cache/huggingface")
parser.add_argument("--input_file", type=str, default="/scratch/mmarvani/LVLM/datasets/MMStar/mmstar_inputs.jsonl")
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

logger.info(f"Loading model: {args.model_type}")
model, processor = load_model_and_processor(args.model_type, model_names, args.cache_dir, device, args.load_in_8bit)
model.eval()
logger.info("Model loaded")

# Load inputs
with open(args.input_file) as f:
    data = [json.loads(l) for l in f]
logger.info(f"Loaded {len(data)} questions")

# Resume support
completed_indices = set()
if os.path.exists(args.output_file):
    with open(args.output_file) as f:
        for line in f:
            rec = json.loads(line)
            completed_indices.add(rec["index"])
    logger.info(f"Resuming: {len(completed_indices)} already done")

os.makedirs(os.path.dirname(args.output_file), exist_ok=True)

for item in tqdm(data, desc=f"MMStar {args.model_type}"):
    if item["index"] in completed_indices:
        continue

    img_path = item["img_path"]
    if not os.path.exists(img_path):
        logger.warning(f"Image {img_path} not found")
        continue

    try:
        raw_image = Image.open(img_path).convert("RGB")
        question = item["prompt"]
        inputs = process_inputs(raw_image, question, processor, args.model_type)

        generated_ids = model.generate(
            **inputs,
            do_sample=False,
            max_new_tokens=args.max_new_tokens,
            num_beams=1
        )

        response = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()

        # Model-specific cleanup
        if "ASSISTANT: " in response:
            # LLaVA / LLaVA-NeXT
            response = response.split("ASSISTANT: ")[-1]
        elif args.model_type == "instructblip":
            # InstructBLIP echoes the question — strip it
            q_text = item["question"].strip()
            if response.startswith(q_text):
                response = response[len(q_text):].strip()
            # Also try stripping prompt (may have suffix)
            p_text = item["prompt"].strip()
            if response.startswith(p_text):
                response = response[len(p_text):].strip()

        output_rec = {
            "index": item["index"],
            "question": item["question"],
            "answer": item["answer"],
            "response": response,
            "category": item.get("category", ""),
            "l2_category": item.get("l2_category", "")
        }

        with open(args.output_file, "a") as f:
            f.write(json.dumps(output_rec) + "\n")

        torch.cuda.empty_cache()

    except Exception as e:
        logger.error(f"Error on index {item['index']}: {e}")
        continue

logger.info(f"Done. Saved to {args.output_file}")