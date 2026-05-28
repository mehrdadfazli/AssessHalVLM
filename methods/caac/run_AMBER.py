#!/usr/bin/env python3
"""AMBER benchmark inference with CAAC."""
import argparse
import gc
import json
import logging
import os
from datetime import datetime

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from bench_io import (
    apply_full_benchmark_scope,
    benchmark_file_tag,
    finalize_hf_cache_args,
    load_config_defaults,
)
from caac_config import add_caac_arguments, build_caac_exp_config
from caac_core import assert_caac_supported, compute_attention_factor, dynamic_generate_caac
from model_utils import load_model_and_processor, model_names, process_inputs


def recorder(out):
    neg_words = ["No", "not", "no", "NO"]
    out = out.replace(".", "").replace(",", "")
    words = out.split()
    if any(w in neg_words for w in words) or any(w.endswith("n't") for w in words):
        return "No"
    return "Yes"


AMBER_QUERY_ALL = "data/query/query_all.json"
AMBER_QUERY_GENERATIVE = "data/query/query_generative.json"
AMBER_QUERY_DISCRIMINATIVE = "data/query/query_discriminative.json"


def uses_generative_output_format(item, queries_json_path, args):
    """Generative: multi-length captions; discriminative: short Yes/No via recorder()."""
    if args.only_generative:
        return True
    if args.only_discriminative:
        return False
    base = os.path.basename(queries_json_path).lower()
    if base == "query_generative.json":
        return True
    if base.startswith("query_discriminative") and base.endswith(".json"):
        return False
    return int(item["id"]) <= 1004


def resolve_amber_queries_json(args):
    """Split jobs load dataset query files directly (no id filtering of query_all)."""
    if args.only_generative:
        return AMBER_QUERY_GENERATIVE
    if args.only_discriminative:
        return AMBER_QUERY_DISCRIMINATIVE
    return args.queries_json


def build_parser():
    p = argparse.ArgumentParser(description="Run CAAC on AMBER benchmark")
    p.add_argument("--config", type=str, default=None)
    p.add_argument("--amber_path", type=str, default=None)
    p.add_argument("--log_dir", type=str, default=None)
    p.add_argument("--cache_dir", type=str, default=None)

    p.add_argument("--queries_json", default=AMBER_QUERY_ALL)
    p.add_argument("--annotations_json", default="data/annotations.json")
    p.add_argument("--image_dirname", default="image")

    p.add_argument("--model_type", default="llava", choices=["instructblip", "llava", "llava-next"])
    p.add_argument("--gpu_id", type=int, default=0)
    p.add_argument("--load_in_8bit", action="store_true", default=True)
    p.add_argument("--no_caac", action="store_true", help="Disable CAAC (vanilla generation)")
    p.add_argument("--do_sample", action="store_true", default=False)
    p.add_argument("--num_beams", type=int, default=1)
    p.add_argument("--max_new_tokens", type=int, default=512)

    add_caac_arguments(p)

    p.add_argument("--only_generative", action="store_true", default=False)
    p.add_argument("--only_discriminative", action="store_true", default=False)
    p.add_argument("--num_items", type=int, default=None)
    p.add_argument("--random_seed", type=int, default=42)
    return p


def main():
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", type=str, default=None)
    pre_args, _ = pre.parse_known_args()

    parser = build_parser()
    parser.set_defaults(**load_config_defaults(pre_args.config))
    args = apply_full_benchmark_scope(finalize_hf_cache_args(parser.parse_args()))
    if getattr(args, "num_items", None) is None and getattr(args, "num_images", None) is not None:
        args.num_items = args.num_images
    if args.only_generative and args.only_discriminative:
        raise SystemExit("Use only one of --only_generative or --only_discriminative")
    if not args.amber_path:
        raise SystemExit("--amber_path is required")
    if not args.log_dir:
        raise SystemExit("--log_dir is required")

    os.makedirs(args.log_dir, exist_ok=True)
    log_file = os.path.join(args.log_dir, f"log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    logging.basicConfig(filename=log_file, level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    logger = logging.getLogger(__name__)
    use_caac = bool(getattr(args, "use_CAAC", True)) and not args.no_caac


    if torch.cuda.is_available() and args.gpu_id is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    exp_config = build_caac_exp_config(args, compute_attention_factor)
    tag = benchmark_file_tag(args.model_type)
    with open(os.path.join(args.log_dir, "config_amber.json"), "w", encoding="utf-8") as f:
        json.dump({k: v for k, v in exp_config.items() if not callable(v)}, f, indent=4)

    output_file = os.path.join(args.log_dir, f"{tag}_amber.json")

    model, processor = load_model_and_processor(
        args.model_type, model_names, args.cache_dir, device, args.load_in_8bit
    )
    model.eval()

    queries_rel = resolve_amber_queries_json(args)
    json_query_path = os.path.join(args.amber_path, queries_rel)
    image_dir = os.path.join(args.amber_path, args.image_dirname)
    with open(json_query_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    items = list(data)
    if args.num_items is not None:
        rng = np.random.default_rng(args.random_seed)
        idxs = rng.permutation(len(items))[: args.num_items]
        items = [items[i] for i in idxs]

    responses = []
    if os.path.exists(output_file):
        try:
            with open(output_file, "r", encoding="utf-8") as f:
                responses = json.load(f)
        except json.JSONDecodeError:
            responses = []
    processed_ids = {r.get("id") for r in responses}
    split = "generative" if args.only_generative else "discriminative" if args.only_discriminative else "all"
    remaining = sum(1 for it in items if int(it["id"]) not in processed_ids)
    logger.info(
        "AMBER queries=%s split=%s: %d items, %d ids already in output, %d remaining — %s",
        queries_rel,
        split,
        len(items),
        len(processed_ids),
        remaining,
        output_file,
    )

    for item in tqdm(items, desc=f"AMBER-{split}"):
        question_id = int(item["id"])
        if question_id in processed_ids:
            continue
        img_path = os.path.join(image_dir, item["image"])
        try:
            raw_image = Image.open(img_path).convert("RGB")
        except OSError as e:
            logger.error("Skip %s: %s", img_path, e)
            continue

        query = item.get("query", "Describe this image.")
        gen_fmt = uses_generative_output_format(item, queries_rel, args)
        max_tok = args.max_new_tokens if gen_fmt else 10
        exp_loop = {**exp_config, "max_new_tokens": max_tok}

        if use_caac:
            assert_caac_supported(args.model_type)
            generated_ids, _ratio = dynamic_generate_caac(
                raw_image,
                query,
                model=model,
                processor=processor,
                model_type=args.model_type,
                exp_config=exp_loop,
                do_sample=args.do_sample,
                num_beams=args.num_beams,
                logger_=logger,
            )
            if gen_fmt:
                response_text_64 = processor.batch_decode(
                    generated_ids[:, :64], skip_special_tokens=True
                )[0].strip()
                response_text_128 = processor.batch_decode(
                    generated_ids[:, :128], skip_special_tokens=True
                )[0].strip()
                response_text_256 = processor.batch_decode(
                    generated_ids[:, :256], skip_special_tokens=True
                )[0].strip()
                response_text_512 = processor.batch_decode(
                    generated_ids[:, :512], skip_special_tokens=True
                )[0].strip()
                result = {
                    "id": question_id,
                    "response_64": response_text_64,
                    "response_128": response_text_128,
                    "response_256": response_text_256,
                    "response_512": response_text_512,
                    "response_length": int(generated_ids.shape[-1]),
                }
            else:
                response_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
                result = {
                    "id": question_id,
                    "response": recorder(response_text),
                    "response_length": int(generated_ids.shape[-1]),
                }
        else:
            inputs = process_inputs(raw_image, query, processor, args.model_type)
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    do_sample=args.do_sample,
                    max_new_tokens=max_tok,
                    num_beams=args.num_beams,
                )
            gen_ids = outputs[:, inputs["input_ids"].shape[-1] :]
            if gen_fmt:
                result = {
                    "id": question_id,
                    "response_64": processor.batch_decode(gen_ids[:, :64], skip_special_tokens=True)[0].strip(),
                    "response_128": processor.batch_decode(gen_ids[:, :128], skip_special_tokens=True)[0].strip(),
                    "response_256": processor.batch_decode(gen_ids[:, :256], skip_special_tokens=True)[0].strip(),
                    "response_512": processor.batch_decode(gen_ids[:, :512], skip_special_tokens=True)[0].strip(),
                    "response_length": int(gen_ids.shape[-1]),
                }
            else:
                response_text = processor.batch_decode(gen_ids, skip_special_tokens=True)[0].strip()
                result = {
                    "id": question_id,
                    "response": recorder(response_text),
                    "response_length": int(gen_ids.shape[-1]),
                }

        responses.append(result)
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(responses, f, indent=4)
            f.flush()
            os.fsync(f.fileno())
        processed_ids.add(question_id)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

    logger.info("Done. Results: %s", output_file)


if __name__ == "__main__":
    main()
