# AGLA (Assembly of Global and Local Attention)

Our integration of **AGLA** (An et al., CVPR 2025; https://github.com/Lackel/AGLA) across all 3 models and all 4 benchmarks. Training-free; uses a BLIP Image-Text-Matching (ITM) model to produce a prompt-relevant augmented image, then contrastively decodes against it.

Hyperparameters: `alpha = 2.0`, `beta = 0.5`, BLIP-ITM GradCAM `block_num = 6`, masking ratio `= 1 - itc/2`. Contrast is **additive**: `logits + alpha * logits_cd`.

## What is in this directory

```
agla/
├── eval/                       # AGLA generation scripts for LLaVA-1.5 / InstructBLIP
│   ├── run_llava_chair.py
│   ├── run_instructblip_chair.py
│   ├── run_mmhal.py            # --model_type {llava, instructblip}
│   ├── run_mmstar_agla.py
│   ├── run_llava_pope.py       # original AGLA POPE scripts (templates)
│   ├── run_instructblip_pope.py
│   ├── augmentation.py         # BLIP-ITM GradCAM augmentation
│   ├── precompute_augmentation.py  # save augmented images to disk (for the split pipeline)
│   └── sample.py               # evolve_agla_sampling() — contrastive decode hook
├── lavis_patches/              # our patches to AGLA's vendored lavis/
│   ├── lavis_init.py           # -> lavis/__init__.py
│   └── lavis_models_init.py    # -> lavis/models/__init__.py
├── llavanext/                  # LLaVA-NeXT split pipeline (runs in lvlm env)
│   ├── run_agla_llavanext.py            # CHAIR / AMBER
│   ├── run_agla_llavanext_mmhal.py
│   ├── run_agla_llavanext_mmstar.py
│   └── run_agla_instructblip_mmstar.py  # HF InstructBLIP + augmented imgs (MMStar workaround)
├── slurm/                      # SLURM submission scripts for every run
├── agla_integration_guide.py   # drop-in generate_with_agla() for HF InstructBLIP
└── requirements.txt
```

## Setup

1. Clone the official AGLA repo:
   ```bash
   git clone https://github.com/Lackel/AGLA.git
   cd AGLA
   ```
2. Build the `agla` env (`docs/ENVIRONMENTS.md`) and install `requirements.txt`. Install `salesforce-lavis` with `--no-deps`, then pin `numpy<2`, `diffusers==0.21.0`, `huggingface-hub<1.0`.
3. Apply our patches (they prevent LAVIS from importing unused models with unmet dependencies):
   ```bash
   cp lavis_patches/lavis_init.py        AGLA/lavis/__init__.py
   cp lavis_patches/lavis_models_init.py AGLA/lavis/models/__init__.py
   ```
4. Copy our `eval/*.py` scripts into `AGLA/eval/`. They are adapted from the original POPE scripts (the `run_*_pope.py` templates are included for reference) with the `"answer with one word"` suffix removed for open-ended benchmarks.

## LLaVA-1.5 / InstructBLIP (env: `agla`)

Run directly — see commands in `docs/REPRODUCTION.md`.

## LLaVA-NeXT (two-environment split)

AGLA's vendored `llava/` is 1.5-only and pins `transformers==4.34`; LLaVA-NeXT needs `>=4.40`. So:

1. **Pre-compute augmented images** in the `agla` env (`precompute_augmentation.py`) → saves augmented images to disk.
2. **Contrastively decode** in the `lvlm` env (`llavanext/run_agla_llavanext*.py`), loading augmented images from disk. The augmented image is resized to match the original's dimensions so LLaVA-NeXT AnyRes produces matching image-token counts.

## Known issue: InstructBLIP + MMStar

LAVIS-based InstructBLIP returns empty responses on MMStar's long MCQ prompts (confirmed even for vanilla, not AGLA-specific). Use `llavanext/run_agla_instructblip_mmstar.py`, which loads InstructBLIP via HuggingFace (no KV-cache; full forward per step) with augmented images from disk, mirroring AFTER's left-truncation handling. See `agla_integration_guide.py` for the standalone `generate_with_agla()` function.
