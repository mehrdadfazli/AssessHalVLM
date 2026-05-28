#!/usr/bin/env python3
"""MMStar benchmark: HF metadata + local PNG cache; CAAC inference; JSONL for eval/mmstar_eval.py."""
import argparse
import gc
import json
import logging
import os
from datetime import datetime
from io import BytesIO

import torch
from datasets import load_dataset
from PIL import Image
from tqdm import tqdm

from bench_io import (
    apply_full_benchmark_scope,
    benchmark_file_tag,
    finalize_hf_cache_args,
    load_config_defaults,
    load_jsonl_int_field,
)
from caac_config import add_caac_arguments, build_caac_exp_config
from caac_core import assert_caac_supported, compute_attention_factor, dynamic_generate_caac
from model_utils import load_model_and_processor, model_names, process_inputs


def process_data_mmstar(
    data_root,
    hf_dataset_id="Lin-Chen/MMStar",
    hf_config="val",
    split="val",
    cache_dir=None,
    max_samples=None,
):
    kwargs = {"trust_remote_code": True}
    if cache_dir:
        kwargs["cache_dir"] = cache_dir
    ds = load_dataset(hf_dataset_id, hf_config, split=split, **kwargs)
    img_root = os.path.join(data_root, "MMStar", "images")
    os.makedirs(img_root, exist_ok=True)

    data_list = []
    for i, row in enumerate(ds):
        if max_samples is not None and i >= max_samples:
            break
        idx = int(row["index"])
        img_path = os.path.join(img_root, f"{idx}.png")
        if not os.path.isfile(img_path):
            im = row["image"]
            if isinstance(im, Image.Image):
                im = im.convert("RGB")
            elif isinstance(im, dict):
                if im.get("bytes"):
                    im = Image.open(BytesIO(im["bytes"])).convert("RGB")
                elif im.get("path"):
                    im = Image.open(im["path"]).convert("RGB")
                else:
                    raise ValueError("MMStar image dict missing bytes/path")
            else:
                im = Image.open(im).convert("RGB")
            im.save(img_path)

        question = row["question"]
        prompt = question.rstrip()
        if "Please select the correct answer" not in prompt:
            prompt = prompt + "\nPlease select the correct answer from the options above.\n"

        new_result = {
            "index": idx,
            "img_url": img_path,
            "prompt": prompt,
            "lan": "mmstar",
            "type": "choice",
            "answer": str(row["answer"]).strip(),
            "question": question,
        }
        if row.get("category") is not None:
            new_result["category"] = row["category"]
        if row.get("l2_category") is not None:
            new_result["l2_category"] = row["l2_category"]
        data_list.append(new_result)
    return data_list


def build_parser():
    p = argparse.ArgumentParser(description="Run CAAC on MMStar")
    p.add_argument("--config", type=str, default=None)
    p.add_argument("--mmstar_data_root", type=str, default=None)
    p.add_argument("--mmstar_hf_cache", type=str, default=None)
    p.add_argument("--mmstar_hf_id", type=str, default="Lin-Chen/MMStar")
    p.add_argument("--mmstar_hf_config", type=str, default="val")
    p.add_argument("--mmstar_split", type=str, default="val")
    p.add_argument("--mmstar_max_samples", type=int, default=None)
    p.add_argument("--log_dir", type=str, default=None)
    p.add_argument("--output", type=str, default=None)
    p.add_argument("--cache_dir", type=str, default=None)

    p.add_argument("--model_type", default="llava-next", choices=["instructblip", "llava", "llava-next"])
    p.add_argument("--gpu_id", type=int, default=0)
    p.add_argument("--load_in_8bit", action="store_true", default=True)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--limit", type=int, default=None)

    p.add_argument("--no_caac", action="store_true", help="Disable CAAC (vanilla generation)")
    p.add_argument("--do_sample", action="store_true", default=False)
    p.add_argument("--num_beams", type=int, default=1)
    p.add_argument("--max_new_tokens", type=int, default=32)

    add_caac_arguments(p)
    return p


def main():
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", type=str, default=None)
    pre_args, _ = pre.parse_known_args()

    parser = build_parser()
    parser.set_defaults(**load_config_defaults(pre_args.config))
    args = apply_full_benchmark_scope(finalize_hf_cache_args(parser.parse_args()))
    if args.limit is None and getattr(args, "num_images", None) is not None:
        args.limit = args.num_images
    if args.mmstar_max_samples is None and args.limit is not None:
        args.mmstar_max_samples = args.limit

    if not args.mmstar_data_root:
        raise SystemExit("--mmstar_data_root is required")
    if not args.log_dir:
        raise SystemExit("--log_dir is required")

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")

    os.makedirs(args.log_dir, exist_ok=True)
    log_file = os.path.join(args.log_dir, f"log_mmstar_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    logging.basicConfig(filename=log_file, level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    logger = logging.getLogger(__name__)
    use_caac = bool(getattr(args, "use_CAAC", True)) and not args.no_caac


    model, processor = load_model_and_processor(
        args.model_type, model_names, args.cache_dir, device, args.load_in_8bit
    )
    model.eval()
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    tag = benchmark_file_tag(args.model_type)
    with open(os.path.join(args.log_dir, "config_mmstar.json"), "w", encoding="utf-8") as f:
        json.dump({k: v for k, v in vars(args).items() if k != "config"}, f, indent=2)

    out_path = args.output or os.path.join(args.log_dir, f"{tag}_mmstar.jsonl")
    hf_cache = args.mmstar_hf_cache or args.cache_dir

    exp_config = build_caac_exp_config(args, compute_attention_factor)

    data = process_data_mmstar(
        args.mmstar_data_root,
        hf_dataset_id=args.mmstar_hf_id,
        hf_config=args.mmstar_hf_config,
        split=args.mmstar_split,
        cache_dir=hf_cache,
        max_samples=args.mmstar_max_samples,
    )
    if args.limit is not None:
        data = data[: args.limit]

    done_idx = load_jsonl_int_field(out_path, "index")
    data = [row for row in data if int(row["index"]) not in done_idx]
    if done_idx:
        logger.info(
            "Resume MMStar: skipping %d rows already in %s; %d remaining",
            len(done_idx),
            out_path,
            len(data),
        )

    if not data:
        logger.info("MMStar: nothing to run (all indices already in %s).", out_path)
        return

    with open(out_path, "a", encoding="utf-8", buffering=1) as fout:
        for idx, item in enumerate(tqdm(data, desc="MMStar")):
            img_path = item["img_url"]
            prompt = item["prompt"]
            try:
                raw_image = Image.open(img_path).convert("RGB")
            except OSError as e:
                logger.error("[%s] %s", idx, e)
                fout.write(json.dumps({**item, "response": "Error"}, default=str) + "\n")
                fout.flush()
                os.fsync(fout.fileno())
                continue

            try:
                if use_caac:
                    assert_caac_supported(args.model_type)
                    gen_ids, _nrep = dynamic_generate_caac(
                        raw_image,
                        prompt,
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
                    record = {**item, "response": answer if answer else " "}
                else:
                    inputs = process_inputs(raw_image, prompt, processor, args.model_type)
                    with torch.no_grad():
                        outputs = model.generate(
                            **inputs,
                            do_sample=args.do_sample,
                            max_new_tokens=args.max_new_tokens,
                            num_beams=args.num_beams,
                        )
                        gen_ids = outputs[:, inputs["input_ids"].shape[-1] :]
                    answer = processor.batch_decode(gen_ids, skip_special_tokens=True)[0].strip()
                    record = {**item, "response": answer if answer else " "}
            except Exception as e:
                logger.error("[%s] gen: %s", idx, e)
                record = {**item, "response": "Error"}

            fout.write(json.dumps(record, default=str) + "\n")
            fout.flush()
            os.fsync(fout.fileno())
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()

    logger.info("Done: %s", out_path)


if __name__ == "__main__":
    main()
