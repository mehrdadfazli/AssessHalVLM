# AFTER

Training-free activation editing for mitigating object hallucination in large vision-language models (LVLMs). This release contains inference, feature extraction, QAO training, and benchmark evaluation code.

**Paper:** [AFTER (arXiv:2601.01957)](https://arxiv.org/abs/2601.01957)

## Supported models (7B)

- LLaVA-1.5
- InstructBLIP (Vicuna-7B)
- LLaVA-NeXT

## Benchmarks

| Benchmark | Inference | Scoring |
|-----------|-----------|---------|
| CHAIR | `inference_editing.py` | `eval/CHAIR/chair.py` |
| AMBER | `inference_editing.py` | `eval/AMBER/amber_eval.py` |
| MMStar | `inference_editing.py` | `eval/MMStar/mmstar_eval.py` |
| MMHal-Bench | `inference_editing.py` | `eval/MMHal-Bench/mmhal_eval.py` |

## Installation

```bash
conda create -n after python=3.10 -y
conda activate after
pip install -r requirements.txt
mkdir -p features probes cache results
```

Set model weights via `--model_dir` (or update paths in `inference_editing.py`). Use `--cache_dir ./cache/huggingface_cache` for Hugging Face downloads.

## Quick start

**1. Extract activations** (three modes on AMBER train split):

```bash
python get_activations.py --model_name <MODEL_KEY> --dataset_name AMBER_train_I+Q \
  --mode I+Q --data_root <AMBER_ROOT> --features_root ./features --cache_dir ./cache/huggingface_cache
# Repeat for T+Q_query and T+Q_best
```

**2. Train offset estimator (QAO):**

```bash
python train_estimator.py --model <MODEL_KEY> \
  --query_set AMBER_train_I+Q --caption_set AMBER_train_T+Q_query --vector_set AMBER_train_T+Q_best \
  --path ./features --save_path probes/<MODEL_KEY>_offset_generator
```

**3. Run inference** (example: MMStar with AFTER):

```bash
python inference_editing.py \
  --model <MODEL_KEY> --model_dir <HF_OR_LOCAL_MODEL> \
  --validate_dataset MMStar --data_root ./data \
  --probe_dataset AMBER_train --pos_mode T+Q_best --neg_mode I+Q \
  --num_heads 64 --alpha 7 \
  --offset_name <MODEL_KEY>_offset_generator_q_10 \
  --subfix "_I+Q;T+Q_best" \
  --features_root ./features --probes_root ./probes --results_root ./results \
  --cache_dir ./cache/huggingface_cache
```

Vanilla baseline: add `--no_intervention --subfix "_vanilla"`.

**4. Score** (example: MMStar exact match):

```bash
python eval/MMStar/mmstar_eval.py \
  --cap_file results/MMStar/<MODEL_KEY>_64_7_I+Q;T+Q_best.jsonl \
  --scoring exact
```

Dataset roots (`--data_root`, `--mmhal_data_root`, COCO paths for CHAIR) must be passed explicitly; they are not bundled in this archive.

## Layout

```
├── get_activations.py
├── train_estimator.py
├── inference_editing.py
├── utils.py
├── models/
├── eval/
├── benchmarks/MMStar/   # shareable MMStar kit
├── configs/             # (via scripts/)
└── requirements.txt
```

## Citation

```bibtex
@article{wang2026after,
  title={AFTER: Mitigating the Object Hallucination of LVLM via Adaptive Factual-Guided Activation Editing},
  author={Wang, Tianbo and Ma, Yuqing and Liao, Kewei and Zhang, Zhange and Li, Simin and Guo, Jinyang and Liu, Xianglong},
  journal={arXiv preprint arXiv:2601.01957},
  year={2026}
}
```
