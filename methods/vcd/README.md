# VCD — Visual Contrastive Decoding

Inference code for **VCD** (Leng et al., CVPR 2024) applied to three 7B LVLMs, for the paper *"Hallucination Mitigation or Conservative Decoding?"*. Training-free, inference-time only.

> **One-line idea.** Contrast the model's logits on the original image against its logits on a Gaussian-noised reference image, subtract the noised distribution, and keep only tokens above an adaptive plausibility threshold.

## Layout

```
sina_VCD/
├── README.md
├── llava_v1.5_7B/          # LLaVA-1.5 7B — all 4 benchmarks
│   ├── chair_eval_llava.py
│   ├── amber_eval_llava.py
│   ├── mmstar_eval_llava.py
│   ├── mmhal_eval_llava.py
│   └── scripts/            # SLURM sbatch wrappers (the per-run config of record)
│       ├── chair_llava_vcd.sh
│       ├── amber_llava_vcd.sh
│       ├── mmstar_llava_vcd.sh
│       └── mmhal_llava_vcd.sh
├── instructblip_7B/        # InstructBLIP-Vicuna 7B — all 4 benchmarks
│   ├── chair_eval_instructblip.py
│   ├── amber_eval_instructblip.py
│   ├── mmstar_eval_instructblip.py
│   ├── mmhal_eval_instructblip.py
│   └── scripts/
│       ├── chair_ib_vcd.sh
│       ├── amber_ib_vcd.sh
│       ├── mmstar_ib_vcd.sh
│       └── mmhal_ib_vcd.sh
└── llava_next_7B/          # LLaVA-NeXT 7B — MMStar + MMHal-Bench only
    ├── mmstar_eval_llava_next.py
    ├── mmhal_eval_llava_next.py
    ├── sampling_utils.py    # hand-rolled VCD/M3ID token loops (CAAC origin)
    └── scripts/
        ├── mmstar_next_vcd.sh
        └── mmhal_next_vcd.sh
```

## Cells covered

| Model | CHAIR | AMBER | MMStar | MMHal-Bench |
|---|:---:|:---:|:---:|:---:|
| LLaVA-1.5 7B | ✅ | ✅ | ✅ | ✅ |
| InstructBLIP 7B | ✅ | ✅ | ✅ | ✅ |
| LLaVA-NeXT 7B | — (CAAC harness run) | — (CAAC harness run) | ✅ | ✅ |

**10 VCD cells** in this package. LLaVA-NeXT × CHAIR/AMBER were run by a teammate (CAAC harness) and are not included here.

## Hyperparameters (VCD paper defaults, Leng et al. 2024 §4.1)

| Knob | Value |
|---|---:|
| `cd_alpha` (contrast strength) | 1.0 |
| `cd_beta` (plausibility threshold) | 0.1 |
| `noise_step` (DDPM step) | 500 |

Applied **uniformly across all 3 models** (no per-model tuning). `seed=42` everywhere.

## Decoding & token budgets

| Model | Decoding | CHAIR/AMBER `max_new_tokens` | MMStar `max_new_tokens` | MMHal `max_new_tokens` |
|---|---|---:|---:|---:|
| LLaVA-1.5 | multinomial sampling (CHAIR T=0.1, others T=1.0; top_p=1.0) | 512 | 64 | 512 |
| InstructBLIP | nucleus sampling (`use_nucleus_sampling=True`, top_p=1.0) | 512 | 64 | 512 |
| LLaVA-NeXT | **greedy** (`do_sample=False`) | n/a | 64 | 512 |

## How a cell runs

Each `scripts/*.sh` is a SLURM sbatch wrapper that sources the right environment, `cd`s into the harness, and calls the matching eval `.py` with the method flags. The VCD flag set is always:

```
--use_cd True --use_m3id False --use_avisc False
```

VCD-specific values (`--cd_alpha 1.0 --cd_beta 0.1 --noise_step 500`) are the eval script's argparse defaults; the wrappers do not override them.

Example (LLaVA-1.5 × MMStar × VCD), from `llava_v1.5_7B/scripts/mmstar_llava_vcd.sh`:
```bash
python experiments/cd_scripts/mmstar_eval_llava.py \
    --model-path liuhaotian/llava-v1.5-7b \
    --jsonl_path <MMStar inputs jsonl> \
    --log_path <out dir> \
    --seed 42 --max_token 64 --gpu-id 0 \
    --use_cd True --use_m3id False --use_avisc False
```

## Environments

| Models | venv | transformers | torch | quantization |
|---|---|---|---|---|
| LLaVA-1.5, InstructBLIP | `avisc-env` | 4.31.0 | 2.0.1 | 8-bit (bitsandbytes 0.41) |
| LLaVA-NeXT | `avisc-next-env` | 4.47.0 | 2.5.1 | 8-bit (bitsandbytes 0.45) |

## Dependency note (important for whoever assembles the repo)

- The **LLaVA-1.5 + InstructBLIP** eval scripts (`*_eval_llava.py`, `*_eval_instructblip.py`) assume the **AvisC harness tree** around them — they do `sys.path` appends and `import` vendored `llava/`, `lavis/`, and `avisc_utils/` (which contains `vcd_add_noise` and the `evolve_avisc_sampling()` patch that implements the VCD/M3ID/AvisC decoding). Those vendored packages are **not bundled here** (they're large and shared). Drop these scripts back under `AvisC/experiments/cd_scripts/` in the merged repo, or vendor the `avisc_utils/`, `llava/`, `lavis/` packages alongside.
- The **LLaVA-NeXT** scripts are more self-contained: they import only `sampling_utils.py` (bundled here) plus stock `transformers` (`LlavaNextForConditionalGeneration` / `LlavaNextProcessor`). No AvisC vendoring needed.
- The eval `.py` files are **method-agnostic** (they accept both `--use_cd` and `--use_m3id`); they are identical to the copies in the M3ID package. The method is selected entirely by the wrapper's flags.

## Scoring (not bundled — shared team kit)

Generation here produces `predictions.jsonl` (CHAIR/MMStar/MMHal) or `Amber_result.json` (AMBER). Scoring uses the shared team scorers (`chair.py`, `amber_eval.py`, `mmstar_eval.py`, MMHal GPT-4o judge) — same files your teammates use; not duplicated in this method package.
