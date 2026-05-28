"""
AGLA + LLaVA-NeXT on MMHal-Bench using pre-computed augmented images.
"""
import os
import json
import torch
import argparse
import logging
import gc
from PIL import Image
from tqdm import tqdm
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.model_utils import load_model_and_processor, process_inputs

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser()
parser.add_argument("--cache_dir", type=str, default="/path/to/.cache/huggingface")
parser.add_argument("--mmhal_path", type=str, default="/path/to/LVLM/datasets/mmhal-bench")
parser.add_argument("--augmented_folder", type=str, required=True)
parser.add_argument("--output_file", type=str, required=True)
parser.add_argument("--alpha", type=float, default=2.0)
parser.add_argument("--beta", type=float, default=0.5)
parser.add_argument("--max_new_tokens", type=int, default=512)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--load_in_8bit", action="store_true", default=True)
args = parser.parse_args()

torch.manual_seed(args.seed)

model_names = {"llava-next": "llava-hf/llava-v1.6-vicuna-7b-hf"}
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

logger.info("Loading LLaVA-NeXT...")
model, processor = load_model_and_processor("llava-next", model_names, args.cache_dir, device, args.load_in_8bit)
model.eval()

with open(os.path.join(args.mmhal_path, "response_template.json")) as f:
    data = json.load(f)
logger.info(f"Loaded {len(data)} questions")

for item in tqdm(data, desc="AGLA LLaVA-NeXT MMHal"):
    image_filename = item["image_src"].split("/")[-1]
    image_path = os.path.join(args.mmhal_path, "images", image_filename)
    augmented_path = os.path.join(args.augmented_folder, image_filename)
    question = item["question"]

    if not os.path.exists(image_path):
        item["model_answer"] = ""
        continue

    try:
        raw_image = Image.open(image_path).convert("RGB")
        inputs = process_inputs(raw_image, question, processor, "llava-next")

        if os.path.exists(augmented_path):
            augmented_image = Image.open(augmented_path).convert("RGB")
            augmented_image = augmented_image.resize(raw_image.size)
            inputs_cd = process_inputs(augmented_image, question, processor, "llava-next")
        else:
            inputs_cd = None

        generated_tokens = []
        input_ids = inputs["input_ids"]
        attention_mask = inputs.get("attention_mask", None)
        pixel_values = inputs["pixel_values"]
        image_sizes = inputs.get("image_sizes", None)

        if inputs_cd is not None:
            pixel_values_cd = inputs_cd["pixel_values"]
            image_sizes_cd = inputs_cd.get("image_sizes", None)

        past_kv = None
        past_kv_cd = None

        for step in range(args.max_new_tokens):
            is_first = step == 0
            current_ids = input_ids if is_first else next_token_id

            with torch.no_grad():
                fwd_kwargs = dict(
                    input_ids=current_ids,
                    pixel_values=pixel_values if is_first else None,
                    attention_mask=attention_mask,
                    past_key_values=past_kv,
                    use_cache=True
                )
                if image_sizes is not None and is_first:
                    fwd_kwargs["image_sizes"] = image_sizes
                out = model(**fwd_kwargs)
                next_logits = out.logits[:, -1, :]
                past_kv = out.past_key_values

            if inputs_cd is not None:
                with torch.no_grad():
                    fwd_kwargs_cd = dict(
                        input_ids=current_ids,
                        pixel_values=pixel_values_cd if is_first else None,
                        attention_mask=attention_mask,
                        past_key_values=past_kv_cd,
                        use_cache=True
                    )
                    if image_sizes_cd is not None and is_first:
                        fwd_kwargs_cd["image_sizes"] = image_sizes_cd
                    out_cd = model(**fwd_kwargs_cd)
                    next_logits_cd = out_cd.logits[:, -1, :]
                    past_kv_cd = out_cd.past_key_values

                cutoff = torch.log(torch.tensor(args.beta).to(next_logits.device)) + next_logits.max(dim=-1, keepdim=True).values
                adjusted = next_logits + args.alpha * next_logits_cd
                adjusted = adjusted.masked_fill(next_logits < cutoff, -float("inf"))
                next_token_id = torch.argmax(adjusted, dim=-1, keepdim=True)
            else:
                next_token_id = torch.argmax(next_logits, dim=-1, keepdim=True)

            if next_token_id.item() == processor.tokenizer.eos_token_id:
                break

            generated_tokens.append(next_token_id)

            if attention_mask is not None:
                attention_mask = torch.cat([
                    attention_mask,
                    torch.ones((1, 1), dtype=attention_mask.dtype, device=attention_mask.device)
                ], dim=-1)

        if generated_tokens:
            gen_seq = torch.cat(generated_tokens, dim=-1)
            response = processor.tokenizer.decode(gen_seq.squeeze(), skip_special_tokens=True).strip()
        else:
            response = ""

        item["model_answer"] = response
        torch.cuda.empty_cache()
        gc.collect()

    except Exception as e:
        logger.error(f"Error: {e}")
        item["model_answer"] = ""
        continue

os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
with open(args.output_file, "w") as f:
    json.dump(data, f, indent=2)
logger.info(f"Saved to {args.output_file}")
