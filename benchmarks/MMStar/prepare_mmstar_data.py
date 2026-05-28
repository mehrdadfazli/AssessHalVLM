#!/usr/bin/env python3
"""
Download MMStar from Hugging Face, cache images locally, emit a JSONL manifest.

Output layout (under --data_root):
  MMStar/images/{index}.png
  MMStar/mmstar_inputs.jsonl

Each line of mmstar_inputs.jsonl is one example with ground-truth fields and paths.
Collaborators run their model, append a ``response`` field per line (or write a new
JSONL with the same rows plus ``response``), then run mmstar_eval.py.
"""
from __future__ import annotations

import argparse
import json
import os
from io import BytesIO

from datasets import load_dataset
from PIL import Image


def main() -> None:
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    p.add_argument(
        "--data_root",
        type=str,
        default="./data",
        help="Parent directory; creates MMStar/images/ and MMStar/mmstar_inputs.jsonl",
    )
    p.add_argument("--hf_dataset_id", type=str, default="Lin-Chen/MMStar")
    p.add_argument("--hf_config", type=str, default="val")
    p.add_argument("--split", type=str, default="val")
    p.add_argument("--cache_dir", type=str, default=None, help="HF datasets cache")
    p.add_argument("--max_samples", type=int, default=None, help="Debug: first N rows only")
    p.add_argument(
        "--add_vlmeval_suffix",
        action="store_true",
        help=(
            "Add trailing 'Please select the correct answer...' to field ``prompt`` "
            "(matches AFTER / VLMEvalKit-style wording). Field ``question`` stays raw HF."
        ),
    )
    args = p.parse_args()

    kwargs = {"trust_remote_code": True}
    if args.cache_dir:
        kwargs["cache_dir"] = args.cache_dir
    ds = load_dataset(args.hf_dataset_id, args.hf_config, split=args.split, **kwargs)

    img_root = os.path.join(args.data_root, "MMStar", "images")
    os.makedirs(img_root, exist_ok=True)
    out_jsonl = os.path.join(args.data_root, "MMStar", "mmstar_inputs.jsonl")

    n = 0
    with open(out_jsonl, "w", encoding="utf-8") as out:
        for i, row in enumerate(ds):
            if args.max_samples is not None and i >= args.max_samples:
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
            if args.add_vlmeval_suffix and "Please select the correct answer" not in prompt:
                prompt = (
                    prompt
                    + "\nPlease select the correct answer from the options above.\n"
                )

            rec = {
                "index": idx,
                "question": question,
                "prompt": prompt,
                "answer": str(row["answer"]).strip(),
                "img_path": os.path.abspath(img_path),
            }
            if row.get("category") is not None:
                rec["category"] = row["category"]
            if row.get("l2_category") is not None:
                rec["l2_category"] = row["l2_category"]

            out.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n += 1

    print(f"Wrote {n} rows to {out_jsonl}")
    print(f"Images under {img_root}")


if __name__ == "__main__":
    main()
