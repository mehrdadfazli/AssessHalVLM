"""
MMHal-Bench evaluation for LLaVA-NeXT (LlavaNextForConditionalGeneration).
Uses CAAC's hand-rolled VCD/M3ID loops from sampling_utils.py (Option B).

Notable decisions:
- 8-bit quant hardcoded (parity; CAAC's store_true+default=True is a bug, but all reference runs use 8-bit)
- M3ID lamda=0.2, beta=0.1 (parity per CAAC run_chair_baselines.py:31)
- 'lamda' (sic) is the kwarg name in sampling_utils.py — typo preserved
- inputs from LlavaNextProcessor flow directly into sampling_utils functions
- Output schema {**item, "model_answer": text} matches MMHal-Bench's expected field name
  (judge script reads `model_answer`; original response_template.json ships empty placeholder)
- max_token=512 (open-ended QA, matches CHAIR/AMBER convention; NOT MMStar's 64)
- Question fed verbatim (no MCQ suffix — MMHal-Bench is open-ended)
- Bug 10 fix retained: explicitly set processor.patch_size + vision_feature_select_strategy

Per-item append-mode write. Partial-line corruption theoretically possible if crash during write,
but JSON records are small (<5KB) and write is single syscall. Resume reads with try/except on
json.loads to skip any corrupt line, then dedup-by-index keeping last occurrence and rewrite
cleanly before appending further records.
"""

import os
import gc
import json
import argparse
import random

import numpy as np
import torch
from tqdm import tqdm
from PIL import Image

from transformers import (
    LlavaNextForConditionalGeneration,
    LlavaNextProcessor,
    BitsAndBytesConfig,
)

from sampling_utils import generate_VCD, generate_M3ID


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    if v.lower() in ("no", "false", "f", "n", "0"):
        return False
    raise argparse.ArgumentTypeError("Boolean value expected.")


def setup_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def parse_args():
    parser = argparse.ArgumentParser()

    # Standard (mirror mmstar_eval_llava_next.py)
    parser.add_argument("--model-path", type=str, default="llava-hf/llava-v1.6-vicuna-7b-hf")
    parser.add_argument("--jsonl_path", type=str, default="/path/to/LVLM/datasets/MMHal-Bench/mmhal_inputs.jsonl")
    parser.add_argument("--data_path", type=str, default="/path/to/LVLM/datasets/MMHal-Bench/images")
    parser.add_argument("--log_path", type=str, default="/path/to/LVLM/lvlm-logs/MMHal-Bench/llava_next")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_token", type=int, default=512)
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=1)

    # Method flags (kept in sync with our existing CLI surface; sbatch wrappers pass these)
    parser.add_argument("--use_cd", type=str2bool, default=False)
    parser.add_argument("--use_m3id", type=str2bool, default=False)
    parser.add_argument("--use_avisc", type=str2bool, default=False)

    # VCD knobs (map to sampling_utils.generate_VCD vcd_alpha/vcd_beta/vcd_noise_step)
    parser.add_argument("--cd_alpha", type=float, default=1.0)
    parser.add_argument("--cd_beta", type=float, default=0.1)
    parser.add_argument("--noise_step", type=int, default=500)

    # M3ID knobs (NEW; map to sampling_utils.generate_M3ID lamda/beta; defaults match CAAC run_chair_baselines.py:31)
    parser.add_argument("--m3id_lamb", type=float, default=0.2)
    parser.add_argument("--m3id_beta", type=float, default=0.1)

    # AvisC knobs (CLI parity only; not exercised here)
    parser.add_argument("--layer_gamma", type=float, default=0.5)
    parser.add_argument("--masking_scheme", type=str, default="zeros")
    parser.add_argument("--lamb", type=float, default=0.99)

    # Doc/parity
    parser.add_argument("--exp_description", type=str, default="..")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--top_k", type=int, default=None)

    return parser.parse_args()


def load_resume_state(out_path):
    """Read existing predictions.jsonl; drop malformed/partial lines; dedup by index (keep last);
    rewrite cleanly. Returns set of done indices."""
    if not os.path.exists(out_path):
        return set()

    valid_records = []
    with open(out_path) as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"Warning: skipping malformed line {line_no} in {out_path}: {e}")
                continue
            if "index" not in rec:
                print(f"Warning: skipping line {line_no} without 'index' key in {out_path}")
                continue
            valid_records.append(rec)

    seen = set()
    deduped = []
    for rec in reversed(valid_records):
        if rec["index"] not in seen:
            seen.add(rec["index"])
            deduped.append(rec)
    deduped.reverse()

    if len(deduped) != len(valid_records):
        print(f"Resume cleanup: {len(valid_records)} raw records -> {len(deduped)} after dedup")

    with open(out_path, "w") as f:
        for rec in deduped:
            f.write(json.dumps(rec) + "\n")

    return set(rec["index"] for rec in deduped)


def main():
    args = parse_args()

    os.makedirs(args.log_path, exist_ok=True)
    args_path = os.path.join(args.log_path, "command_line_args.json")
    with open(args_path, "w") as f:
        json.dump(vars(args), f, indent=2)
    print(f"Wrote {args_path}")

    setup_seeds(args.seed)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    quant_cfg = BitsAndBytesConfig(load_in_8bit=True, llm_int8_threshold=200.0)
    print(f"Loading model: {args.model_path}")
    model = LlavaNextForConditionalGeneration.from_pretrained(
        args.model_path,
        torch_dtype=torch.float16,
        attn_implementation="eager",
        quantization_config=quant_cfg,
        device_map="auto",
    )
    model.eval()
    processor = LlavaNextProcessor.from_pretrained(args.model_path)
    # Suppress deprecation warning + match CAAC convention: set patch_size and
    # vision_feature_select_strategy on the processor so image-token expansion
    # happens at processor time, not at model.forward (slightly faster + future-proof).
    processor.patch_size = model.config.vision_config.patch_size
    processor.vision_feature_select_strategy = model.config.vision_feature_select_strategy
    tokenizer = processor.tokenizer
    print("Model + processor loaded")

    with open(args.jsonl_path) as f:
        data_list = [json.loads(line) for line in f if line.strip()]
    print(f"Loaded {len(data_list)} items from {args.jsonl_path}")

    out_path = os.path.join(args.log_path, "predictions.jsonl")
    done_idx = load_resume_state(out_path)
    pending = [item for item in data_list if item["index"] not in done_idx]
    print(f"Total: {len(data_list)}, Done: {len(done_idx)}, Pending: {len(pending)}")

    for item in tqdm(pending, desc="MMHal-Bench LLaVA-NeXT"):
        img_path = item["img_path"]
        raw_image = Image.open(img_path).convert("RGB")
        query = item["question"]

        conversation = [{
            "role": "user",
            "content": [{"type": "image"}, {"type": "text", "text": query}],
        }]
        text_prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
        inputs = processor(
            images=raw_image, text=text_prompt, padding=True, return_tensors="pt",
        ).to(device, torch.float16)

        if args.use_cd:
            response = generate_VCD(
                model=model, tokenizer=tokenizer, inputs=inputs,
                max_new_tokens=args.max_token, do_sample=False, raw_image=raw_image,
                vcd_alpha=args.cd_alpha, vcd_beta=args.cd_beta, vcd_noise_step=args.noise_step,
            )
        elif args.use_m3id:
            response = generate_M3ID(
                model=model, tokenizer=tokenizer, inputs=inputs,
                max_new_tokens=args.max_token, do_sample=False, raw_image=raw_image,
                lamda=args.m3id_lamb, beta=args.m3id_beta,
            )
        else:
            generated_ids = model.generate(
                **inputs, do_sample=False, max_new_tokens=args.max_token, num_beams=1,
            )
            response = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
            response = response.split("ASSISTANT: ")[1] if "ASSISTANT: " in response else response

        response = response.strip()

        out_record = {**item, "model_answer": response}
        with open(out_path, "a") as f:
            f.write(json.dumps(out_record) + "\n")

        torch.cuda.empty_cache()
        gc.collect()

    print(f"Done. Output: {out_path}")


if __name__ == "__main__":
    main()
