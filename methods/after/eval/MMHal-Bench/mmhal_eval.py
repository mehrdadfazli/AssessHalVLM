"""
Prepare and optionally score MMHal-Bench results.

This script bridges AFTER's JSONL outputs (from `inference_editing.py`) into
MMHal-Bench's expected `response_template.json` format (with `model_answer`
filled), and can optionally invoke the official GPT-4 scoring script.
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, List

_DEFAULT_EVAL_SCRIPT = str(Path(__file__).resolve().parent / "eval_gpt4_openai_v1.py")


def load_jsonl_results(path: str) -> Dict[int, str]:
    """Map item index -> model response text from AFTER outputs.

    Supports:
    - JSONL (one JSON object per line), typically from inference_editing.py
    - JSON array (e.g. an already-filled MMHal response file)
    """
    def _extract_answer(row: dict) -> str:
        # AFTER conventions: typically `response`; fall back to `prediction`/`answer`.
        ans = row.get("response", None)
        if ans is None:
            ans = row.get("prediction", None)
        if ans is None:
            ans = row.get("answer", None)
        if ans is None:
            # Filled MMHal template / merged outputs often use this key.
            ans = row.get("model_answer", "")
        return str(ans)

    idx_to_answer: Dict[int, str] = {}
    with open(path, "r") as f:
        content = f.read()

    stripped = content.lstrip()
    if not stripped:
        return idx_to_answer

    # Case 1: JSON array/object file.
    if stripped[0] in "[{":
        parsed = json.loads(content)
        if isinstance(parsed, list):
            for i, row in enumerate(parsed):
                if not isinstance(row, dict):
                    continue
                idx = row.get("index", i)
                idx_to_answer[int(idx)] = _extract_answer(row)
            return idx_to_answer

        if isinstance(parsed, dict):
            idx = parsed.get("index", 0)
            idx_to_answer[int(idx)] = _extract_answer(parsed)
            return idx_to_answer

    # Case 2: JSONL file.
    for line_no, line in enumerate(content.splitlines(), start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(
                    f"Failed to parse JSONL line {line_no} in {path}: {e}"
                ) from e
            idx = row.get("index", None)
            if idx is None:
                continue
            idx_to_answer[int(idx)] = _extract_answer(row)
    return idx_to_answer


def fill_template(
    template_json: str,
    idx_to_answer: Dict[int, str],
    *,
    strict_len_96: bool = True,
) -> List[dict]:
    with open(template_json, "r") as f:
        records = json.load(f)

    if strict_len_96:
        assert len(records) == 96, f"Expected 96 MMHal items, got {len(records)}"

    for i, rec in enumerate(records):
        rec["model_answer"] = idx_to_answer.get(i, rec.get("model_answer", "")) or ""
    return records


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Convert AFTER MMHal JSONL to MMHal-Bench JSON and optionally run GPT scoring.",
    )
    p.add_argument(
        "--results_jsonl",
        type=str,
        required=True,
        help="Path to AFTER JSONL outputs (results/MMHal-Bench/*.jsonl).",
    )
    p.add_argument(
        "--template_json",
        type=str,
        required=True,
        help="Path to MMHal-Bench response_template.json (from the dataset download).",
    )
    p.add_argument(
        "--output_json",
        type=str,
        default=None,
        help="Where to write filled JSON. Default: alongside results_jsonl with .json extension.",
    )
    p.add_argument(
        "--run_gpt_eval",
        action="store_true",
        help="If set, run MMHal-Bench eval_gpt4.py after writing output_json.",
    )
    p.add_argument(
        "--eval_script",
        type=str,
        default=_DEFAULT_EVAL_SCRIPT,
        help=(
            "Path to GPT judge script. Default uses eval_gpt4_openai_v1.py "
            "(OpenAI SDK >= 1). For legacy openai<1, pass the dataset eval_gpt4.py."
        ),
    )
    p.add_argument(
        "--api_key",
        type=str,
        default=None,
        help="OpenAI API key for eval_gpt4.py. If omitted, uses OPENAI_API_KEY env var.",
    )
    p.add_argument(
        "--gpt_model",
        type=str,
        default="gpt-4o-mini",
        help="Chat judge model passed to the GPT scorer (e.g. gpt-4o-mini).",
    )
    p.add_argument(
        "--evaluation_json",
        type=str,
        default=None,
        help="Where to save GPT evaluation raw outputs. Default: output_json with _gpt_eval.json suffix.",
    )
    return p


def main():
    args = build_parser().parse_args()

    if args.output_json is None:
        base, _ = os.path.splitext(args.results_jsonl)
        args.output_json = base + ".json"
    elif os.path.abspath(args.output_json) == os.path.abspath(args.results_jsonl):
        raise ValueError(
            "--output_json must be different from --results_jsonl. "
            "Use a .json output path (e.g. replace .jsonl with .json)."
        )

    idx_to_answer = load_jsonl_results(args.results_jsonl)
    filled = fill_template(args.template_json, idx_to_answer, strict_len_96=True)

    os.makedirs(os.path.dirname(os.path.abspath(args.output_json)), exist_ok=True)
    with open(args.output_json, "w") as f:
        json.dump(filled, f, indent=2)

    if not args.run_gpt_eval:
        return

    api_key = args.api_key or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("No OpenAI API key provided. Pass --api_key or set OPENAI_API_KEY.")

    if args.evaluation_json is None:
        base, _ = os.path.splitext(args.output_json)
        args.evaluation_json = base + "_gpt_eval.json"

    cmd = [
        sys.executable,
        args.eval_script,
        "--response",
        args.output_json,
        "--evaluation",
        args.evaluation_json,
        "--api-key",
        api_key,
        "--gpt-model",
        args.gpt_model,
    ]
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()

