#!/usr/bin/env python3
"""CHAIR benchmark inference with CAAC."""
import argparse
import gc
import json
import logging
import os
import random
from datetime import datetime

import torch
from PIL import Image
from tqdm import tqdm

from bench_io import benchmark_file_tag, finalize_hf_cache_args, load_config_defaults, load_jsonl_int_field
from caac_config import add_caac_arguments, build_caac_exp_config
from caac_core import assert_caac_supported, compute_attention_factor, dynamic_generate_caac
from model_utils import load_model_and_processor, model_names, process_inputs


def load_opera_image_ids(opera_results_path):
    image_ids = []
    with open(opera_results_path, "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)
            image_ids.append(data["image_id"])
    return image_ids


def build_parser():
    p = argparse.ArgumentParser(description="Run CAAC on CHAIR benchmark")
    p.add_argument("--config", type=str, default=None)
    p.add_argument("--chair_data_path", type=str, default=None)
    p.add_argument("--log_dir", type=str, default=None)
    p.add_argument("--cache_dir", type=str, default=None)

    p.add_argument("--model_type", default="llava", choices=["instructblip", "llava", "llava-next"])
    p.add_argument("--gpu_id", type=int, default=0)
    p.add_argument("--load_in_8bit", action="store_true", default=True)
    p.add_argument("--no_caac", action="store_true", help="Disable CAAC (vanilla generation)")
    p.add_argument("--do_sample", action="store_true", default=False)
    p.add_argument("--num_beams", type=int, default=1)
    p.add_argument("--max_new_tokens", type=int, default=512)

    add_caac_arguments(p)

    p.add_argument("--opera_results", action="store_true", default=False)
    p.add_argument("--opera_results_path", type=str, default=None)
    p.add_argument("--num_images", type=int, default=500)
    p.add_argument("--random_seed", type=int, default=42)
    return p


def main():
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", type=str, default=None)
    pre_args, _ = pre.parse_known_args()

    parser = build_parser()
    parser.set_defaults(**load_config_defaults(pre_args.config))
    args = finalize_hf_cache_args(parser.parse_args())

    if not args.chair_data_path:
        raise SystemExit("--chair_data_path is required (or set in --config)")
    if not args.log_dir:
        raise SystemExit("--log_dir is required (or set in --config)")

    os.makedirs(args.log_dir, exist_ok=True)
    log_file = os.path.join(args.log_dir, f"log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    logging.basicConfig(filename=log_file, level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    logger = logging.getLogger(__name__)
    use_caac = bool(getattr(args, "use_CAAC", True)) and not args.no_caac


    if torch.cuda.is_available():
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("Using device: %s", device)
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    exp_config = build_caac_exp_config(args, compute_attention_factor)
    tag = benchmark_file_tag(args.model_type)
    config_path = os.path.join(args.log_dir, "config_chair.json")
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump({**{k: v for k, v in exp_config.items() if not callable(v)}, "chair_data_path": args.chair_data_path}, f, indent=4)

    output_file = os.path.join(args.log_dir, f"{tag}_chair.jsonl")
    done_ids = load_jsonl_int_field(output_file, "image_id")

    model, processor = load_model_and_processor(
        args.model_type, model_names, args.cache_dir, device, args.load_in_8bit
    )
    model.eval()

    if args.opera_results:
        if not args.opera_results_path:
            raise SystemExit("--opera_results_path required when --opera_results")
        image_ids = load_opera_image_ids(args.opera_results_path)
        image_list = [(f"COCO_val2014_{i:012d}.jpg", i) for i in image_ids]
    else:
        img_files = [f for f in os.listdir(args.chair_data_path) if f.endswith(".jpg")]
        image_list = [(f, int(f.split(".jpg")[0][-6:])) for f in img_files]
        random.seed(args.random_seed)
        random.shuffle(image_list)
        if args.num_images is not None:
            image_list = image_list[: args.num_images]

    image_list = [(f, i) for f, i in image_list if i not in done_ids]
    if done_ids:
        logger.info(
            "Resume CHAIR: skipping %d images already in %s; %d remaining",
            len(done_ids),
            output_file,
            len(image_list),
        )

    query = "Describe this image."

    for img_file, img_id in tqdm(image_list, desc="CHAIR"):
        img_path = os.path.join(args.chair_data_path, img_file)
        raw_image = Image.open(img_path).convert("RGB")

        if use_caac:
            assert_caac_supported(args.model_type)
            gen_ids, n_repeats = dynamic_generate_caac(
                raw_image,
                query,
                model=model,
                processor=processor,
                model_type=args.model_type,
                exp_config=exp_config,
                do_sample=args.do_sample,
                num_beams=args.num_beams,
                logger_=logger,
            )
            caption_64 = processor.batch_decode(gen_ids[:, :64], skip_special_tokens=True)[0].strip()
            caption_512 = processor.batch_decode(gen_ids[:, :512], skip_special_tokens=True)[0].strip()
            result = {
                "image_id": img_id,
                "caption_64": caption_64,
                "caption_512": caption_512,
                "n_repeats_forward": n_repeats,
            }
        else:
            inputs = process_inputs(raw_image, query, processor, args.model_type)
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    do_sample=args.do_sample,
                    max_new_tokens=args.max_new_tokens,
                    num_beams=args.num_beams,
                )
            gen_ids = outputs[:, inputs["input_ids"].shape[-1] :]
            caption_64 = processor.batch_decode(gen_ids[:, :64], skip_special_tokens=True)[0].strip()
            caption_512 = processor.batch_decode(gen_ids[:, :512], skip_special_tokens=True)[0].strip()
            result = {"image_id": img_id, "caption_64": caption_64, "caption_512": caption_512}

        with open(output_file, "a", encoding="utf-8") as f:
            json.dump(result, f)
            f.write("\n")
            f.flush()
            os.fsync(f.fileno())

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

    logger.info("Done. Results: %s", output_file)


if __name__ == "__main__":
    main()
