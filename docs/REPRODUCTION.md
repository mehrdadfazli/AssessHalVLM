# Reproduction Guide

Every cell of the results matrix → the exact command that produces it. The workflow for any cell is always:

1. **Activate** the right environment (`docs/ENVIRONMENTS.md`).
2. **Generate** model responses (GPU).
3. **Convert** output format if needed (AGLA outputs need conversion before CHAIR/AMBER scoring).
4. **Score** (offline, except MMHal which uses the GPT-4o judge).

Paths below assume a repo root of `$REPO` and a scratch workspace for data/outputs. Replace dataset paths with your own. Seeds are fixed at `42`.

---

## Results matrix (reported values)

### CHAIR — CHAIRs ↓ / CHAIRi ↓ / Recall ↑ / Len

| Model | Vanilla | VCD | M3ID | AGLA |
|-------|---------|-----|------|------|
| LLaVA-NeXT | 33.0 / 9.1 / 64.7 / 173.0 | 42.6 / 10.7 / 69.9 / 170.6 | 40.8 / 12.3 / 65.2 / 189.6 | 29.6 / 8.2 / 59.4 / 173.2 |
| LLaVA-1.5 | 46.6 / 14.8 / 79.6 / 88.1 | 53.8 / 16.0 / 80.0 / 102.0 | 56.2 / 15.7 / 82.2 / 94.4 | 55.2 / 14.7 / 78.2 / 101.3 |
| InstructBLIP | 50.8 / 15.9 / 75.7 / 102.1 | 59.2 / 17.1 / 74.8 / 104.1 | 69.8 / 21.3 / 74.0 / 102.0 | 51.0 / 13.5 / 72.4 / 106.1 |

### AMBER — CHAIR ↓ / Cover ↑ / Hal ↓ / Cog ↓ / Len

| Model | Vanilla | VCD | M3ID | AGLA |
|-------|---------|-----|------|------|
| LLaVA-NeXT | 9.2 / 61.1 / 50.3 / 5.0 | 11.1 / 63.6 / 59.2 / 5.3 | 13.4 / 60.4 / 61.3 / 5.5 | 8.8 / 59.9 / 50.3 / 4.5 |
| LLaVA-1.5 | 7.4 / 48.7 / 31.3 / 3.4 | 10.2 / 51.0 / 44.0 / 4.3 | 8.0 / 56.9 / 43.5 / 3.2 | 7.6 / 51.0 / 36.7 / 3.9 |
| InstructBLIP | 8.4 / 54.1 / 38.2 / 4.1 | 9.3 / 54.3 / 43.1 / 4.3 | 10.3 / 51.1 / 46.2 / 4.7 | 7.5 / 54.0 / 35.1 / 3.8 |

### MMHal-Bench — Avg Score ↑ / Hallucination Rate ↓

| Model | Vanilla | AGLA |
|-------|---------|------|
| LLaVA-NeXT | 2.64 / 0.57 | 2.83 / 0.51 |
| LLaVA-1.5 | 2.19 / 0.60 | 2.20 / 0.60 |
| InstructBLIP | 1.73 / 0.67 | 1.90 / 0.62 |

### MMStar — Overall accuracy

| Model | Vanilla | AGLA |
|-------|---------|------|
| LLaVA-NeXT | 32.27% | 32.87% |
| LLaVA-1.5 | 32.13% | 31.33% |
| InstructBLIP | 33.8% (HF pipeline) | (HF pipeline, left-truncated prompts) |

VCD/M3ID values for MMHal and MMStar, and all CAAC/CEI/AFTER values, follow the same workflow — see those method directories.

---

## Vanilla baselines (env: `lvlm`)

Generation harness: `methods/vanilla/experiments/`. `<MODEL>` ∈ `{llava, llava-next, instructblip}`.

**CHAIR:**
```bash
python methods/vanilla/experiments/run_chair_baselines.py \
  --model_type <MODEL> --method none \
  --image_folder /path/to/coco/val2014 \
  --question_file data/chair_500.jsonl \
  --output_file outputs/<MODEL>_chair_vanilla.jsonl --load_in_8bit
# score:
python eval/chair.py --coco_path /path/to/coco/annotations \
  --cap_file outputs/<MODEL>_chair_vanilla.jsonl
```

**AMBER:**
```bash
python methods/vanilla/experiments/run_amber_baselines.py \
  --model_type <MODEL> --method none \
  --image_folder /path/to/AMBER/image \
  --output_file outputs/<MODEL>_amber_vanilla.json --load_in_8bit
# score:
python eval/amber.py --inference_data outputs/<MODEL>_amber_vanilla.json \
  --evaluation_type g --annotation /path/to/AMBER/data/annotations.json \
  --word_association /path/to/AMBER/data/relation.json \
  --safe_words /path/to/AMBER/data/safe_words.txt \
  --metrics /path/to/AMBER/data/metrics.txt
```

**MMHal-Bench** (generation offline, scoring uses GPT-4o):
```bash
python methods/vanilla/experiments/run_mmhal.py \
  --model_type <MODEL> --mmhal_path /path/to/mmhal-bench \
  --output_file outputs/<MODEL>_mmhal_vanilla.json --load_in_8bit
# score (USES API — ~$0.45):
python eval/eval_mmhal_openrouter.py \
  --response outputs/<MODEL>_mmhal_vanilla.json \
  --evaluation outputs/<MODEL>_mmhal_vanilla_eval.json \
  --api-key $OPENROUTER_API_KEY --model-name "<MODEL> Vanilla"
```

**MMStar:**
```bash
python methods/vanilla/experiments/run_mmstar.py \
  --model_type <MODEL> \
  --input_file data/MMStar/mmstar_inputs.jsonl \
  --output_file outputs/<MODEL>_mmstar_vanilla.jsonl --load_in_8bit
# score (offline):
python eval/mmstar_eval.py --cap_file outputs/<MODEL>_mmstar_vanilla.jsonl --scoring exact
```
> InstructBLIP MMStar: see "Known issue" in the top-level README — use the HF left-truncation pipeline.

---

## VCD / M3ID

**LLaVA-1.5 and InstructBLIP** (env per `methods/vcd/README.md`):
```bash
# example: VCD on LLaVA-1.5, CHAIR
bash methods/vcd/llava_v1.5_7B/scripts/chair_llava_vcd.sh
# scripts cover chair/amber/mmstar/mmhal per model
```
VCD: `cd_alpha=1.0`, `cd_beta=0.1`, `noise_step=500`. M3ID: `m3id_lamb=0.2`, `m3id_beta=0.1`. Seed 42.

**LLaVA-NeXT** (env: `lvlm`, CAAC harness — CHAIR + AMBER only):
```bash
python methods/vcd/llava_next_7B_caac/run_chair_baselines.py \
  --model_type llava-next --method vcd \
  --image_folder /path/to/coco/val2014 --question_file data/chair_500.jsonl \
  --output_file outputs/llava-next_chair_vcd.jsonl --load_in_8bit
# (method m3id for M3ID; run_amber_baselines.py for AMBER)
```
LLaVA-NeXT MMStar + MMHal for VCD/M3ID use hand-rolled token loops in `methods/vcd/llava_next_7B/sampling_utils.py`.

---

## AGLA

### LLaVA-1.5 and InstructBLIP (env: `agla`)

CHAIR / MMHal run directly through the AGLA repo:
```bash
# CHAIR (LLaVA-1.5)
python methods/agla/eval/run_llava_chair.py --use_agla --alpha 2 --beta 0.5 --seed 42 \
  --output_file outputs/llava_chair_agla.jsonl
# MMHal (both models via run_mmhal.py --model_type {llava,instructblip})
python methods/agla/eval/run_mmhal.py --model_type instructblip --use_agla \
  --alpha 2 --beta 0.5 --output_file outputs/instructblip_mmhal_agla.json
# MMStar (LLaVA-1.5 / InstructBLIP via run_mmstar_agla.py)
python methods/agla/eval/run_mmstar_agla.py --model_type llava --use_agla \
  --alpha 2 --beta 0.5 --augmented_folder outputs/augmented_mmstar \
  --output_file outputs/llava_mmstar_agla.jsonl
```

### LLaVA-NeXT (split pipeline)

**Step 1 — pre-compute augmented images (env: `agla`):**
```bash
python methods/agla/eval/precompute_augmentation.py \
  --question_file data/chair_500.jsonl --image_folder /path/to/coco/val2014 \
  --output_dir outputs/augmented_chair --image_key image --question_key text
```

**Step 2 — contrastive decoding (env: `lvlm`):**
```bash
python methods/agla/llavanext/run_agla_llavanext.py \
  --question_file data/chair_500.jsonl --image_folder /path/to/coco/val2014 \
  --augmented_folder outputs/augmented_chair \
  --output_file outputs/llavanext_chair_agla.jsonl --alpha 2 --beta 0.5 --seed 42 --load_in_8bit
# variants: run_agla_llavanext_mmhal.py, run_agla_llavanext_mmstar.py
```

### AGLA output conversion before CHAIR/AMBER scoring

AGLA emits `{question_id, text, image}`. Convert before scoring:

```python
# CHAIR: AGLA jsonl -> {image_id, caption}
import json
rows = [json.loads(l) for l in open("outputs/llavanext_chair_agla.jsonl")]
out = [{"image_id": int(r["image"].split(".jpg")[0][-6:]), "caption": r["text"]} for r in rows]
with open("outputs/llavanext_chair_agla_converted.jsonl", "w") as f:
    for o in out: f.write(json.dumps(o) + "\n")
```
```python
# AMBER: AGLA jsonl -> [{id, response, response_length}]
import json
rows = [json.loads(l) for l in open("outputs/llavanext_amber_agla.jsonl")]
out = [{"id": r["question_id"], "response": r["text"], "response_length": len(r["text"].split())} for r in rows]
json.dump(out, open("outputs/llavanext_amber_agla_eval.json", "w"), indent=4)
```
MMStar and MMHal AGLA outputs need no conversion.

---

## CAAC / CEI / AFTER

These are config/CLI-driven and fully documented in their own READMEs:

- **CAAC** — `methods/caac/README.md` (`bash scripts/run_<bench>.sh configs/<model>_w_CAAC.json ...`)
- **CEI** — `methods/cei/README.md` (same pattern, `configs/<model>_w_CEI.json`)
- **AFTER** — `methods/after/README.md` (4-step: extract activations → train QAO → `inference_editing.py` → score). Key params `K=64`, `alpha=7`.

---

## MMHal API cost

GPT-4o judge via OpenRouter: ~1,728 calls for the full method × model matrix (96 questions each), ~$0.0043/call ≈ **$7–8 total**. Run `eval/test_mmhal_single_call.py` first to verify the key and measure exact per-call cost before a full run.
