# Vanilla Baselines

Unmodified greedy-decoding baselines for all 3 models across all 4 benchmarks, run through the CAAC baselines harness (`--method none`). These are the reference points against which every mitigation method's gain/loss is measured.

## Layout

```
vanilla/
├── experiments/    # generation runners
│   ├── run_chair_baselines.py    # CHAIR (also drives VCD/M3ID via --method)
│   ├── run_amber_baselines.py    # AMBER
│   ├── run_mmhal.py              # MMHal-Bench
│   ├── run_mmstar.py             # MMStar
│   └── run_*_llavanext.py / run_chair.py / run_amber.py  # variants
├── src/            # shared model + sampling utilities (CAAC origin)
│   ├── model_utils.py            # adds LLaVA-NeXT + process_inputs
│   ├── sampling_utils.py         # unified generation loop, kwarg filtering
│   └── CAAC_utils.py
├── evals/          # CHAIR + AMBER scorers used by the harness
│   ├── chair.py
│   └── inference.py              # AMBER scorer
├── configs/
└── slurm/          # SLURM submission scripts
```

## Decoding

Greedy (`do_sample=false`, `num_beams=1`), `max_new_tokens=512`, 8-bit loading, `seed=42`. See `docs/REPRODUCTION.md` for per-benchmark commands and `docs/EXPERIMENTAL_SETTINGS.md` §A.5 for the full settings.

## Note

This harness also runs **VCD and M3ID for LLaVA-NeXT** (CHAIR/AMBER) via `--method vcd|m3id`; those same runners are mirrored under `methods/vcd/llava_next_7B_caac/` and `methods/m3id/llava_next_7B_caac/`.
