#!/usr/bin/env python3
"""MMHal-Bench inference with CAAC."""
import argparse
import gc
import json
import logging
import os
from datetime import datetime

import torch
from tqdm import tqdm

from bench_io import benchmark_file_tag, finalize_hf_cache_args, load_config_defaults
from caac_config import add_caac_arguments, build_caac_exp_config
from caac_core import assert_caac_supported, compute_attention_factor, dynamic_generate_caac
from model_utils import load_model_and_processor, model_names, process_inputs


def load_image(image_file: str):
    from io import BytesIO

    import requests
    from PIL import Image

    if image_file.startswith("http://") or image_file.startswith("https://"):
        resp = requests.get(image_file, timeout=30)
        resp.raise_for_status()
        return Image.open(BytesIO(resp.content)).convert("RGB")
    return Image.open(image_file).convert("RGB")


def merge_mmhal_checkpoint(data, out_path):
    if not os.path.isfile(out_path):
        return data
    try:
        with open(out_path, "r", encoding="utf-8") as f:
            prev = json.load(f)
    except (json.JSONDecodeError, OSError):
        return data
    if not isinstance(prev, list) or len(prev) != len(data):
        return data
    for i, item in enumerate(data):
        pa = prev[i].get("model_answer")
        if isinstance(pa, str) and pa.strip() and pa.strip() != "Error":
            item["model_answer"] = pa
    return data


def _save_mmhal_out(out_path, data):
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
        f.flush()
        os.fsync(f.fileno())


def build_parser():
    p = argparse.ArgumentParser(description="Run CAAC on MMHal-Bench")
    p.add_argument("--config", type=str, default=None)
    p.add_argument("--input", type=str, default=None)
    p.add_argument("--images_root", type=str, default=None)
    p.add_argument("--output", type=str, default=None)
    p.add_argument("--log_dir", type=str, default=None)
    p.add_argument("--cache_dir", type=str, default=None)

    p.add_argument("--model_type", default="llava-next", choices=["instructblip", "llava", "llava-next"])
    p.add_argument("--gpu_id", type=int, default=0)
    p.add_argument("--load_in_8bit", action="store_true", default=True)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--limit", type=int, default=None)

    p.add_argument("--no_caac", action="store_true", help="Disable CAAC (vanilla generation)")
    p.add_argument("--do_sample", action="store_true", default=False)
    p.add_argument("--num_beams", type=int, default=1)
    p.add_argument("--max_new_tokens", type=int, default=512)

    add_caac_arguments(p)
    return p


def main():
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", type=str, default=None)
    pre_args, _ = pre.parse_known_args()

    parser = build_parser()
    parser.set_defaults(**load_config_defaults(pre_args.config))
    args = finalize_hf_cache_args(parser.parse_args())
    if args.limit is None and getattr(args, "num_images", None) is not None:
        args.limit = args.num_images

    if not args.input:
        raise SystemExit("--input (MMHal JSON) is required")
    if not args.images_root:
        raise SystemExit("--images_root is required")

    log_dir = args.log_dir or "./results/mmhal"
    os.makedirs(log_dir, exist_ok=True)

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")

    log_file = os.path.join(log_dir, f"log_mmhal_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    logging.basicConfig(filename=log_file, level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    logger = logging.getLogger(__name__)
    use_caac = bool(getattr(args, "use_CAAC", True)) and not args.no_caac


    model, processor = load_model_and_processor(
        args.model_type, model_names, args.cache_dir, device, args.load_in_8bit
    )
    model.eval()
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    tag = benchmark_file_tag(args.model_type)
    out_path = args.output or os.path.join(log_dir, f"{tag}_mmhal.json")
    with open(os.path.join(log_dir, "config_mmhal.json"), "w", encoding="utf-8") as f:
        json.dump({k: v for k, v in vars(args).items() if k != "config"}, f, indent=2)

    with open(args.input, "r", encoding="utf-8") as f:
        data = json.load(f)
    if args.limit is not None:
        data = data[: args.limit]
    data = merge_mmhal_checkpoint(data, out_path)

    exp_config = build_caac_exp_config(args, compute_attention_factor)

    for idx, item in enumerate(tqdm(data, desc="MMHal")):
        ma = item.get("model_answer")
        if isinstance(ma, str) and ma.strip() and ma.strip() != "Error":
            continue
        image_src = item.get("image_src", "")
        question = item.get("question", "")
        if image_src.startswith("http://") or image_src.startswith("https://"):
            image_path = image_src
        else:
            image_path = (
                image_src if os.path.isabs(image_src) else os.path.join(args.images_root, image_src)
            )

        try:
            raw_image = load_image(image_path)
        except OSError as e:
            logger.error("[%s] image error %s: %s", idx, image_path, e)
            item["model_answer"] = "Error"
            _save_mmhal_out(out_path, data)
            continue

        try:
            if use_caac:
                assert_caac_supported(args.model_type)
                gen_ids, n_repeats = dynamic_generate_caac(
                    raw_image,
                    question,
                    model=model,
                    processor=processor,
                    model_type=args.model_type,
                    exp_config=exp_config,
                    do_sample=args.do_sample,
                    num_beams=args.num_beams,
                    logger_=logger,
                )
                cap64 = processor.batch_decode(gen_ids[:, :64], skip_special_tokens=True)[0].strip()
                cap512 = processor.batch_decode(gen_ids[:, :512], skip_special_tokens=True)[0].strip()
                answer = cap512.strip() if cap512.strip() else cap64.strip()
                item["model_answer"] = answer if answer else " "
                item["n_repeats_forward"] = n_repeats
            else:
                inputs = process_inputs(raw_image, question, processor, args.model_type)
                with torch.no_grad():
                    outputs = model.generate(
                        **inputs,
                        do_sample=args.do_sample,
                        max_new_tokens=args.max_new_tokens,
                        num_beams=args.num_beams,
                    )
                gen_ids = outputs[:, inputs["input_ids"].shape[-1] :]
                cap64 = processor.batch_decode(gen_ids[:, :64], skip_special_tokens=True)[0].strip()
                cap512 = processor.batch_decode(gen_ids[:, :512], skip_special_tokens=True)[0].strip()
                answer = cap512.strip() if cap512.strip() else cap64.strip()
                item["model_answer"] = answer if answer else " "
                item["n_repeats_forward"] = 0.0
        except Exception as e:
            logger.error("[%s] gen: %s", idx, e)
            item["model_answer"] = "Error"
            item["n_repeats_forward"] = 0.0

        _save_mmhal_out(out_path, data)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

    logger.info("Done: %s", out_path)


if __name__ == "__main__":
    main()
