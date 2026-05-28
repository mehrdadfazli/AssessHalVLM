# M3ID — Multi-Modal Hallucination Control (Sina's implementation)

Inference code for **M3ID** (Favero et al., CVPR 2024) applied to three 7B LVLMs, for the paper *"Hallucination Mitigation or Conservative Decoding?"*. Training-free, inference-time only.

> **One-line idea.** Amplify image-conditioned logits relative to text-only (unconditional) logits via a λ coefficient, keeping the VCD-style plausibility-threshold filter.

## Layout

```
sina_M3ID/
├── README.md
├── llava_v1.5_7B/          # LLaVA-1.5 7B — all 4 benchmarks
│   ├── chair_eval_llava.py
│   ├── amber_eval_llava.py
│   ├── mmstar_eval_llava.py
│   ├── mmhal_eval_llava.py
│   └── scripts/            # SLURM sbatch wrappers (the per-run config of record)
│       ├── chair_llava_m3id.sh
│       ├── amber_llava_m3id.sh
│       ├── mmstar_llava_m3id.sh
│       └── mmhal_llava_m3id.sh
├── instructblip_7B/        # InstructBLIP-Vicuna 7B — all 4 benchmarks
│   ├── chair_eval_instructblip.py
│   ├── amber_eval_instructblip.py
│   ├── mmstar_eval_instructblip.py
│   ├── mmhal_eval_instructblip.py
│   └── scripts/
│       ├── chair_ib_m3id.sh
│       ├── amber_ib_m3id.sh
│       ├── mmstar_ib_m3id.sh
│       └── mmhal_ib_m3id.sh
└── llava_next_7B/          # LLaVA-NeXT 7B — MMStar + MMHal-Bench only
    ├── mmstar_eval_llava_next.py
    ├── mmhal_eval_llava_next.py
    ├── sampling_utils.py    # hand-rolled VCD/M3ID token loops (CAAC origin)
    └── scripts/
        ├── mmstar_next_m3id.sh
        └── mmhal_next_m3id.sh
```

## Cells covered

| Model | CHAIR | AMBER | MMStar | MMHal-Bench |
|---|:---:|:---:|:---:|:---:|
| LLaVA-1.5 7B | ✅ | ✅ | ✅ | ✅ |
| InstructBLIP 7B | ✅ | ✅ | ✅ | ✅ |
| LLaVA-NeXT 7B | — (Mohit's CAAC run) | — (Mohit's CAAC run) | ✅ | ✅ |

**10 M3ID cells** in this package. LLaVA-NeXT × CHAIR/AMBER were run by a teammate (CAAC harness) and are not included here.

## Hyperparameters

| Knob | Value | Provenance |
|---|---:|---|
| `m3id_lamb` (λ, amplification) | 0.2 | **CAAC harness runner default** — NOT the M3ID paper default. Kept uniform across cells for fair comparison; not retuned. |
| `m3id_beta` (plausibility threshold) | 0.1 | M3ID paper (Favero et al. 2024) |

> **λ provenance — read before citing.** The `λ=0.2` value is the runner-level default of the CAAC baselines harness, which differs from the original M3ID paper's default. We deliberately did not retune it. Cite Favero et al. 2024 for the *method*, but flag λ's source as the harness default. (Note: the LLaVA-NeXT `sampling_utils.py` ships a function-level default of `lamda=0.02`, but the eval scripts override it to `0.2` via `--m3id_lamb`.)

Applied **uniformly across all 3 models**. `seed=42` everywhere.

## Decoding & token budgets

| Model | Decoding | CHAIR/AMBER `max_new_tokens` | MMStar `max_new_tokens` | MMHal `max_new_tokens` |
|---|---|---:|---:|---:|
| LLaVA-1.5 | multinomial sampling (CHAIR T=0.1, others T=1.0; top_p=1.0) | 512 | 64 | 512 |
| InstructBLIP | nucleus sampling (`use_nucleus_sampling=True`, top_p=1.0) | 512 | 64 | 512 |
| LLaVA-NeXT | **greedy** (`do_sample=False`) | n/a | 64 | 512 |

## How a cell runs

Each `scripts/*.sh` is a SLURM sbatch wrapper that sources the right environment, `cd`s into the harness, and calls the matching eval `.py` with the method flags. The M3ID flag set is always:

```
--use_cd False --use_m3id True --use_avisc False
```

M3ID values (`--m3id_lamb 0.2 --m3id_beta 0.1`) are the eval script's argparse defaults on the LLaVA-NeXT path; for LLaVA-1.5 / InstructBLIP, M3ID's mixing is implemented inside AvisC's vendored sampler (toggled by `--use_m3id True`).

Example (LLaVA-1.5 × MMStar × M3ID), from `llava_v1.5_7B/scripts/mmstar_llava_m3id.sh`:
```bash
python experiments/cd_scripts/mmstar_eval_llava.py \
    --model-path liuhaotian/llava-v1.5-7b \
    --jsonl_path <MMStar inputs jsonl> \
    --log_path <out dir> \
    --seed 42 --max_token 64 --gpu-id 0 \
    --use_cd False --use_m3id True --use_avisc False
```

## Environments

| Models | venv | transformers | torch | quantization |
|---|---|---|---|---|
| LLaVA-1.5, InstructBLIP | `avisc-env` | 4.31.0 | 2.0.1 | 8-bit (bitsandbytes 0.41) |
| LLaVA-NeXT | `avisc-next-env` | 4.47.0 | 2.5.1 | 8-bit (bitsandbytes 0.45) |

## Dependency note (important for whoever assembles the repo)

- The **LLaVA-1.5 + InstructBLIP** eval scripts assume the **AvisC harness tree**: they `sys.path`-append and import vendored `llava/`, `lavis/`, and `avisc_utils/` (the latter holds `vcd_add_noise` and the `evolve_avisc_sampling()` patch implementing the M3ID/VCD/AvisC decoding). Those vendored packages are **not bundled here**. Drop these scripts back under `AvisC/experiments/cd_scripts/` in the merged repo, or vendor `avisc_utils/`, `llava/`, `lavis/` alongside.
- The **LLaVA-NeXT** scripts import only `sampling_utils.py` (bundled here) + stock `transformers`. No AvisC vendoring needed.
- The eval `.py` files are **method-agnostic** (accept both `--use_cd` and `--use_m3id`); they are identical to the copies in the VCD package. The method is selected entirely by the wrapper flags.

## Scoring (not bundled — shared team kit)

Generation here produces `predictions.jsonl` (CHAIR/MMStar/MMHal) or `Amber_result.json` (AMBER). Scoring uses the shared team scorers (`chair.py`, `amber_eval.py`, `mmstar_eval.py`, MMHal GPT-4o judge) — same files your teammates use; not duplicated in this method package.
