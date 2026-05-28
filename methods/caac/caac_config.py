"""CAAC hyperparameters: shared defaults, CLI args, and exp_config for inference."""
from __future__ import annotations

import argparse
from typing import Any, Callable, Dict

# Default CAAC hyperparameters (LLaVA / LLaVA-NeXT; 32 decoder layers).
CAAC_HYPERPARAM_DEFAULTS: Dict[str, Any] = {
    "img_txt_cal_layers": list(range(32)),
    "img_cal_layers": [0, 1, 2, 3, 4],
    "min_lamb": 1.0,
    "max_lamb": 1.5,
    "confidence_threshold": 0.25,
    "input_token_idx_calibration": [-1, -2, -3],
    "ref_image": "white",
    "beta": 0.5,
}

CAAC_CONFIG_KEYS = frozenset(CAAC_HYPERPARAM_DEFAULTS.keys())


def add_caac_arguments(parser: argparse.ArgumentParser) -> None:
    """Register CAAC hyperparameter CLI flags (defaults match CAAC_HYPERPARAM_DEFAULTS)."""
    d = CAAC_HYPERPARAM_DEFAULTS
    parser.add_argument("--img_txt_cal_layers", type=int, nargs="+", default=d["img_txt_cal_layers"])
    parser.add_argument("--img_cal_layers", type=int, nargs="+", default=d["img_cal_layers"])
    parser.add_argument("--min_lamb", type=float, default=d["min_lamb"])
    parser.add_argument("--max_lamb", type=float, default=d["max_lamb"])
    parser.add_argument("--confidence_threshold", type=float, default=d["confidence_threshold"])
    parser.add_argument(
        "--input_token_idx_calibration",
        type=int,
        nargs="+",
        default=d["input_token_idx_calibration"],
    )
    parser.add_argument(
        "--ref_image",
        default=d["ref_image"],
        choices=["self", "white", "black", "noise"],
    )
    parser.add_argument("--beta", type=float, default=d["beta"])


def ensure_caac_args(namespace) -> None:
    """Fill missing CAAC fields on the parsed namespace (e.g. base JSON without CAAC keys)."""
    for key, default in CAAC_HYPERPARAM_DEFAULTS.items():
        val = getattr(namespace, key, None)
        if val is None:
            setattr(namespace, key, default)


def build_caac_exp_config(namespace, compute_attention_factor: Callable) -> Dict[str, Any]:
    """Build exp_config passed to dynamic_generate_caac from resolved CLI/config args."""
    ensure_caac_args(namespace)
    exp = {key: getattr(namespace, key) for key in CAAC_CONFIG_KEYS}
    exp["compute_attention_factor"] = compute_attention_factor
    exp["max_new_tokens"] = namespace.max_new_tokens
    return exp
