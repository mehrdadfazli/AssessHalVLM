# AssessHalVLM

**Beyond the Benchmarks: Assessing Hallucination Mitigation Methods in Large Vision-Language Models**

Official evaluation framework for our systematic diagnostic suite spanning **3 base LVLMs × 7 inference-time configurations × 4 benchmarks**. This repository lets anyone reproduce every cell of our results matrix end-to-end: generate model responses, score them, and recover the reported numbers.

---

## Overview

We evaluate whether hallucination-mitigation methods genuinely improve vision-language models or merely game hallucination benchmarks at the cost of general capability. Two hypotheses are tested:

1. **Overfitting** — methods reduce benchmark hallucination scores without improving real grounding.
2. **Capability tax** — reducing hallucination degrades performance on general multimodal tasks.

### Models (7B)

| Key | HuggingFace ID |
|-----|----------------|
| LLaVA-1.5 | `llava-hf/llava-1.5-7b-hf` (also `liuhaotian/llava-v1.5-7b` for AGLA/AFTER) |
| InstructBLIP | `Salesforce/instructblip-vicuna-7b` |
| LLaVA-NeXT | `llava-hf/llava-v1.6-vicuna-7b-hf` |

### Methods

| Method | Family | Directory |
|--------|--------|-----------|
| Vanilla | baseline | `methods/vanilla/` |
| VCD | contrastive decoding | `methods/vcd/` |
| M3ID | contrastive decoding | `methods/m3id/` |
| AGLA | attention / augmentation | `methods/agla/` |
| CAAC | attention calibration | `methods/caac/` | 
| CEI | representation enhancement | `methods/cei/` | 
| AFTER | representation enhancement | `methods/after/` |

### Benchmarks

| Benchmark | Task | Size | Scoring |
|-----------|------|------|---------|
| CHAIR | open-ended captioning | 500 COCO val2014 | `eval/chair.py` (no API) |
| AMBER | generative hallucination | 1,004 (gen split) | `eval/amber.py` (no API) |
| MMHal-Bench | hallucination QA | 96 | `eval/eval_mmhal_openrouter.py` (GPT-4o judge) |
| MMStar | general capability (MCQ) | 1,500 | `eval/mmstar_eval.py` (no API) |

---

## Repository layout

```
AssessHalVLM/
├── README.md                   # this file
├── docs/
│   ├── REPRODUCTION.md          # every results-matrix cell -> exact command
│   ├── EXPERIMENTAL_SETTINGS.md # full hyperparameter appendix (all 7 methods)
│   └── ENVIRONMENTS.md          # conda environment setup per method
├── eval/                        # shared canonical scoring scripts
│   ├── chair.py
│   ├── amber.py
│   ├── eval_gpt4.py             # original OpenAI GPT-4 MMHal judge
│   ├── eval_mmhal_openrouter.py # OpenRouter GPT-4o MMHal judge (used for our runs)
│   ├── test_mmhal_single_call.py
│   └── mmstar_eval.py
├── benchmarks/
│   └── MMStar/                  # MMStar data download + prep
├── data/
│   ├── chair_500.jsonl          # the 500 sampled COCO image IDs + prompts
│   └── amber_generative.jsonl   # AMBER generative split (1,004 items)
└── methods/
    ├── vanilla/                 # CAAC-harness baselines (all 3 models, all 4 benchmarks)
    ├── vcd/                     # LLaVA-1.5, InstructBLIP (AvisC) + LLaVA-NeXT CAAC harness
    ├── m3id/                    # LLaVA-1.5, InstructBLIP (AvisC) + LLaVA-NeXT CAAC harness
    ├── agla/                    # full AGLA pipeline + LLaVA-NeXT split approach
    ├── caac/                    # config-driven
    ├── cei/                     # config-driven
    └── after/                   # activation editing
```

Each method directory is **self-contained** and bundles its own copy of the relevant eval scripts, because methods run in **different conda environments** (see `docs/ENVIRONMENTS.md`). The top-level `eval/` holds the canonical scoring scripts referenced in `docs/REPRODUCTION.md`.

---

## Quick start

### 1. Clone and pick an environment

Methods require different `transformers` versions and cannot share a single environment. See `docs/ENVIRONMENTS.md` for the four environments:

| Environment | transformers | Methods |
|-------------|--------------|---------|
| `lvlm` | 4.47 | Vanilla, VCD/M3ID (LLaVA-NeXT), AGLA (LLaVA-NeXT decoding), all scoring |
| `agla` | 4.34 + LAVIS | AGLA (LLaVA-1.5, InstructBLIP) + augmentation precompute |
| `caac` / `cei` | 4.46–4.49 | CAAC, CEI |
| `after` | 4.46.3 | AFTER |

### 2. Download data

```bash
# MMStar (1,500 images + questions from HuggingFace)
python benchmarks/MMStar/prepare_mmstar_data.py --data_root ./data

# CHAIR: COCO val2014 images + annotations (external)
# AMBER: AMBER image set + annotations (external, see methods/*/README)
# MMHal-Bench: HuggingFace Shengcao1006/MMHal-Bench
```

### 3. Run a method on a benchmark

Each method's README documents its exact interface. Example — AGLA on CHAIR with LLaVA-1.5:

```bash
conda activate agla
cd methods/agla
python eval/run_llava_chair.py --use_agla --alpha 2 --beta 0.5 --seed 42 \
  --output_file outputs/llava_chair_agla.jsonl
```

### 4. Score

```bash
conda activate lvlm
python eval/chair.py \
  --coco_path /path/to/coco/annotations \
  --cap_file methods/agla/outputs/llava_chair_agla_converted.jsonl
```

**The full cell-by-cell command list is in [`docs/REPRODUCTION.md`](docs/REPRODUCTION.md).**

---

## Reproducing the full results matrix

`docs/REPRODUCTION.md` maps every cell of the paper's results table to the exact command that produces it, including:

- Which environment to activate
- The generation command (model + method + benchmark)
- Any output-format conversion needed before scoring
- The scoring command
- The expected metric value

MMHal-Bench is the only benchmark requiring a paid API (GPT-4o judge via OpenRouter, ~$0.45 per model/method, ~$7–8 for the full matrix). All other benchmarks score offline.

---

## Notes and known issues

- **AGLA vendored dependencies.** The AGLA method wraps the official AGLA repo (https://github.com/Lackel/AGLA), which bundles its own `llava/` and `lavis/`. We ship our patches (`methods/agla/lavis_patches/`) and run scripts, not the full vendored tree — clone the AGLA repo and apply the patches per `methods/agla/README.md`.
- **AGLA + InstructBLIP + MMStar.** LAVIS-based InstructBLIP returns empty responses on MMStar's long MCQ prompts (200–300 tokens). This is a LAVIS limitation, not method-specific (vanilla through LAVIS fails identically). Our reported number uses a HuggingFace InstructBLIP pipeline with left-truncated prompts.
- **VCD/M3ID on LLaVA-NeXT.** CHAIR and AMBER for LLaVA-NeXT were run through the CAAC baselines harness (`methods/vcd/llava_next_7B_caac/`, `methods/m3id/llava_next_7B_caac/`), not the AvisC-based scripts (which cover LLaVA-1.5 and InstructBLIP).

---

## Citation

If you use this code, please cite our paper (details to be added) and the original method papers: VCD (Leng et al., 2024), M3ID (Favero et al., 2024), AGLA (An et al., 2025), CAAC, CEI, and AFTER.