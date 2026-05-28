# Environments

The seven methods cannot share one Python environment — they require conflicting `transformers` versions and, in AGLA's case, the LAVIS stack. Set up the four environments below. All work was done on the GMU Hopper cluster (NVIDIA A100 80GB, `gpuq` partition, `--qos=gpu`).

> **Always set the HuggingFace cache to scratch** before running anything, or model downloads will fill your home-directory quota:
> ```bash
> export HF_HOME=/path/to/scratch/.cache/huggingface
> export TRANSFORMERS_CACHE=$HF_HOME
> export TORCH_HOME=/path/to/scratch/.cache/torch
> export TMPDIR=/path/to/scratch/tmp
> ```

---

## 1. `lvlm` — main environment

Used for: Vanilla baselines, VCD/M3ID on LLaVA-NeXT (CAAC harness), AGLA LLaVA-NeXT contrastive decoding, and **all offline scoring** (CHAIR, AMBER, MMStar) plus MMHal OpenRouter scoring.

```bash
conda create -n lvlm python=3.10 -y
conda activate lvlm
pip install -r methods/vanilla/requirements.txt
# torch 2.5.1, transformers 4.47.0
pip install openai            # for MMHal OpenRouter scoring
```

## 2. `agla` — AGLA for LLaVA-1.5 / InstructBLIP + augmentation

Used for: AGLA on LLaVA-1.5 and InstructBLIP (CHAIR, AMBER, MMHal), and pre-computing augmented images for all models.

```bash
conda create -n agla python=3.10 -y
conda activate agla
pip install -r methods/agla/requirements.txt
# torch 2.0.1, transformers 4.34.0, salesforce-lavis 1.0.2
# salesforce-lavis installed with --no-deps; then numpy<2, diffusers==0.21.0, huggingface-hub<1.0
```

Then clone the official AGLA repo and apply our patches — see `methods/agla/README.md`.

## 3. `caac` / `cei` — Mehrdad's calibration/injection methods

```bash
conda create -n caac python=3.10 -y
conda activate caac
pip install -r methods/caac/requirements.txt   # same reqs work for cei
# transformers >=4.46,<4.50
```

## 4. `after` — Mehrdad's activation editing

```bash
conda create -n after python=3.10 -y
conda activate after
pip install -r methods/after/requirements.txt
# transformers 4.46.3
```

---

## Environment → method → cell map

| Environment | Methods | Models | Benchmarks |
|-------------|---------|--------|------------|
| `lvlm` | Vanilla | all 3 | all 4 |
| `lvlm` | VCD, M3ID | LLaVA-NeXT | CHAIR, AMBER |
| `lvlm` | AGLA | LLaVA-NeXT | all 4 |
| `lvlm` | (scoring) | — | CHAIR, AMBER, MMStar, MMHal |
| `agla` | AGLA | LLaVA-1.5, InstructBLIP | CHAIR, AMBER, MMHal |
| `agla` | AGLA augmentation precompute | all 3 | all 4 |
| `caac` | CAAC | all 3 | all 4 |
| `cei` | CEI | all 3 | all 4 |
| `after` | AFTER | all 3 | all 4 |

VCD/M3ID on LLaVA-1.5 and InstructBLIP use Sina's AvisC-based scripts (`methods/vcd/llava_v1.5_7B/`, `methods/vcd/instructblip_7B/`), which carry their own environment notes in `methods/vcd/README.md`.
