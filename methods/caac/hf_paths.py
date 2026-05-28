"""Hugging Face cache paths (repo-local default)."""
import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
HF_CACHE_DIR = str(REPO_ROOT / "cache" / "huggingface_cache")


def setup_hf_cache_env() -> str:
    """Point HF/Transformers cache env vars at the default cache directory."""
    os.makedirs(HF_CACHE_DIR, exist_ok=True)
    for key in ("HF_HOME", "HF_HUB_CACHE", "TRANSFORMERS_CACHE", "HUGGINGFACE_HUB_CACHE"):
        os.environ[key] = HF_CACHE_DIR
    return HF_CACHE_DIR


def resolve_cache_dir(cache_dir) -> str:
    """Non-empty cache_dir wins; otherwise use HF_CACHE_DIR."""
    if cache_dir is not None and str(cache_dir).strip():
        return str(cache_dir).strip()
    return HF_CACHE_DIR
