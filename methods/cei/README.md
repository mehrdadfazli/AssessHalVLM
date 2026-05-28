# Context Embedding Injection (CEI)

Official code for **Context Embedding Injection (CEI)** — a training-free, inference-time method to mitigate hallucination in large vision-language models. CEI extracts a context embedding from one forward pass, then injects it during autoregressive generation using a dynamic two-pass schedule.

## Supported models (7B)

- InstructBLIP (Vicuna-7B)
- LLaVA-1.5
- LLaVA-NeXT

## Benchmarks

| Benchmark | Runner | Evaluation |
|-----------|--------|------------|
| CHAIR | `run_CHAIR.py` | `eval/chair.py` |
| AMBER | `run_AMBER.py` | `eval/amber.py` |
| MMHal-Bench | `run_MMHal.py` | `eval/eval_gpt4.py` |
| POPE | `run_POPE.py` | external POPE scorer |
| MMStar | `run_MMStar.py` | `eval/mmstar_eval.py` |

## Installation

```bash
conda create -n cei python=3.10 -y
conda activate cei
pip install -r requirements.txt
```

Requires a CUDA GPU and Hugging Face access for model weights.

## Quick start

Run from the repository root (`PYTHONPATH=.` or `cd` into this directory).

**CHAIR:**

```bash
bash scripts/run_chair.sh configs/llava_w_CEI.json ./outputs/chair_run \
  /path/to/coco/val2014 /path/to/hf_cache
```

**AMBER:**

```bash
bash scripts/run_amber.sh configs/llava_w_CEI.json ./outputs/amber_run \
  /path/to/AMBER /path/to/hf_cache
```

**MMStar** (parent of `MMStar/images/`):

```bash
bash scripts/run_mmstar.sh configs/llavanext_w_CEI.json ./outputs/mmstar_run \
  /path/to/data_root /path/to/hf_datasets_cache /path/to/hf_cache
```

**Scoring** (example CHAIR):

```bash
python eval/chair.py \
  --cap_file ./outputs/chair_run/llava15_chair.jsonl \
  --caption_key caption_512 \
  --coco_path /path/to/coco/annotations \
  --summary_file ./outputs/chair_run/chair_summary.json
```

Dataset paths are passed via CLI or shell wrappers (not stored in JSON configs). Hyperparameters live under `configs/`.

## Layout

```
├── cei_core.py
├── model_utils.py
├── run_*.py
├── configs/
├── scripts/
├── eval/
└── requirements.txt
```

## Citation

If you use this code, please cite the ACL Findings CEI paper (bibtex when available).
