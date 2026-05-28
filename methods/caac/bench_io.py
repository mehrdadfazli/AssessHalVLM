"""Stable benchmark output filenames and JSONL resume checkpoints."""
import json
import os

from caac_config import ensure_caac_args
from hf_paths import resolve_cache_dir, setup_hf_cache_env


def full_benchmark_enabled() -> bool:
    return os.environ.get("FULL_BENCHMARK", "").strip().lower() in ("1", "true", "yes")


def apply_full_benchmark_scope(args):
    """Drop config subset caps (num_images / num_items / limit) for full AMBER & MMStar runs."""
    if not full_benchmark_enabled():
        return args
    if hasattr(args, "num_items"):
        args.num_items = None
    if hasattr(args, "num_images"):
        args.num_images = None
    if hasattr(args, "limit"):
        args.limit = None
    if hasattr(args, "mmstar_max_samples"):
        args.mmstar_max_samples = None
    return args


def load_config_defaults(config_path):
    """JSON config values for argparse.set_defaults (CLI flags override these)."""
    if not config_path:
        return {}
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    defaults = {}
    for k, v in cfg.items():
        if k.startswith("_"):
            continue
        if k == "cache_dir" and (v is None or v == ""):
            continue
        defaults[k] = v
    if "use_CAAC" not in cfg:
        defaults["use_CAAC"] = True
    return defaults


def finalize_hf_cache_args(args):
    """Apply scratch HF env vars, CAAC defaults, and resolved cache_dir after config merge."""
    setup_hf_cache_env()
    args.cache_dir = resolve_cache_dir(getattr(args, "cache_dir", None))
    mmstar_hf = getattr(args, "mmstar_hf_cache", None)
    if mmstar_hf is None or str(mmstar_hf).strip() == "":
        args.mmstar_hf_cache = args.cache_dir
    ensure_caac_args(args)
    return args


def benchmark_file_tag(model_type: str) -> str:
    """Short tag for result files, e.g. llava15_chair.jsonl."""
    return {
        "instructblip": "instructblip",
        "llava": "llava15",
        "llava-next": "llavanext",
    }.get(model_type, model_type.replace("-", "_"))


def load_jsonl_int_field(path: str, field: str) -> set:
    """Read JSONL; collect int(obj[field]) for each parseable line (skips bad lines)."""
    done = set()
    if not os.path.isfile(path):
        return done
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                done.add(int(json.loads(line)[field]))
            except (json.JSONDecodeError, KeyError, ValueError, TypeError):
                continue
    return done
