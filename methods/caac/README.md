# Confidence Aware Attention Calibration (CAAC)

Official code for **CAAC** — a training-free, inference-time method to mitigate object hallucination by upscaling image–text attention on uncertain decoding steps and applying attention calibration on early layers (WACV 2026).

## Supported models (7B)

- InstructBLIP (Vicuna-7B) — vanilla inference only (CAAC requires LLaVA-style attention hooks)
- LLaVA-1.5
- LLaVA-NeXT

## Benchmarks

| Benchmark | Runner | Evaluation |
|-----------|--------|------------|
| CHAIR | `run_CHAIR.py` | `eval/chair.py` |
| AMBER | `run_AMBER.py` | `eval/amber.py` |
| MMHal-Bench | `run_MMHal.py` | `eval/eval_gpt4.py` |
| MMStar | `run_MMStar.py` | `eval/mmstar_eval.py` |

## Installation

```bash
conda create -n caac python=3.10 -y
conda activate caac
pip install -r requirements.txt
mkdir -p cache/huggingface_cache
```

Default Hugging Face cache: `./cache/huggingface_cache` (override with `--cache_dir` or the last argument to `scripts/run_*.sh`).

## Quick start

Run from the repository root.

**CHAIR:**

```bash
bash scripts/run_chair.sh configs/llavanext_w_CAAC.json ./outputs/chair_run \
  /path/to/coco/val2014 ./cache/huggingface_cache
```

**AMBER:**

```bash
bash scripts/run_amber.sh configs/llavanext_w_CAAC.json ./outputs/amber_run \
  /path/to/AMBER ./cache/huggingface_cache
```

**MMStar:**

```bash
bash scripts/run_mmstar.sh configs/llavanext_w_CAAC.json ./outputs/mmstar_run \
  /path/to/data_root ./cache/huggingface_cache ./cache/huggingface_cache
```

**Scoring** (batch helper; set `COCO_ANNO_PATH` and `OPENAI_API_KEY`):

```bash
export COCO_ANNO_PATH=/path/to/coco/annotations
export OPENAI_API_KEY=...
bash scripts/score_all.sh
```

## Config keys

See `configs/*.json` for `use_CAAC`, `img_txt_cal_layers`, `img_cal_layers`, `min_lamb`, `max_lamb`, `confidence_threshold`, `ref_image`, `beta`, and `max_new_tokens`.

## Layout

```
├── caac_core.py
├── caac_config.py
├── model_utils.py
├── bench_io.py
├── run_*.py
├── configs/
├── scripts/
├── eval/
└── requirements.txt
```

## Citation

If you use this code, please cite the WACV 2026 CAAC paper (bibtex when available).
