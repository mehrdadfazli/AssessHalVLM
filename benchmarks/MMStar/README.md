# MMStar benchmark kit (shareable)

Self-contained folder for collaborators who want to run **other baselines** on [MMStar](https://huggingface.co/datasets/Lin-Chen/MMStar) and score predictions the same way as AFTER (VLMEvalKit-style parsing + optional GPT judge).

The full AFTER pipeline still lives at repo root (`inference_editing.py`, `eval/MMStar/mmstar_eval.py`). This directory is meant to be **copied or zipped** without dragging in probes, SLURM, or intervention code.

## Contents

| File | Purpose |
|------|---------|
| `prepare_mmstar_data.py` | Download HF data, save `MMStar/images/{index}.png`, write `mmstar_inputs.jsonl` |
| `mmstar_eval.py` | Score a JSONL of predictions (`exact` or `vlmeval`) |
| `requirements-mmstar.txt` | Minimal Python dependencies |

## 1) Install

```bash
cd benchmarks/MMStar
pip install -r requirements-mmstar.txt
```

For Hugging Face gated datasets or rate limits, run `huggingface-cli login` if needed.

## 2) Download data and build the manifest

```bash
python prepare_mmstar_data.py --data_root ./data
```

Optional:

- `--cache_dir /path/to/hf_cache` — HF download cache
- `--max_samples 50` — smoke test
- `--add_vlmeval_suffix` — add the same trailing instruction AFTER uses for VLMEvalKit-style prompts (raw `question` is always the HF field used by scoring)

Outputs:

- `./data/MMStar/images/{index}.png` — one image per example  
- `./data/MMStar/mmstar_inputs.jsonl` — one JSON object per line (see below)

## 3) JSONL schema for scoring

`mmstar_eval.py` expects **one JSON object per line** with at least:

| Field | Required | Description |
|-------|----------|-------------|
| `question` | Yes | Full MCQ text as in HF, including the `Options: A: ...` block |
| `answer` | Yes | Ground-truth letter, e.g. `A`–`D` |
| `response` | Yes | **Model output** (free text) |
| `index` | Recommended | Integer id; should match image filename `{index}.png` |
| `category`, `l2_category` | Optional | Used only for breakdown tables in the summary |

**Parity with AFTER:** Runs from `inference_editing.py` write the same fields; `question` is the raw HF question, and the model sees `prompt` with an extra line (`Please select the correct answer...`) when using default AFTER settings. For identical behavior, use `prepare_mmstar_data.py --add_vlmeval_suffix` and feed **`prompt`** to your model while keeping **`question`** in the JSONL for scoring.

Workflow:

1. Start from `mmstar_inputs.jsonl` (no `response` yet).  
2. For each line, load `img_path`, run your LVLM with `prompt` or `question` as you prefer.  
3. Write **`predictions.jsonl`**: same records plus `"response": "<model text>"`.

Order of lines does not affect accuracy as long as each row is self-consistent; the evaluator does not sort by `index`.

## 4) Score

### Offline (heuristics only, no API)

```bash
python mmstar_eval.py \
  --cap_file predictions.jsonl \
  --summary_csv ./summaries/mmstar_exact.csv \
  --scoring exact
```

### VLMEvalKit-style (GPT judge when heuristics fail)

```bash
export OPENAI_API_KEY=...
# optional: export OPENAI_BASE_URL=https://...
python mmstar_eval.py \
  --cap_file predictions.jsonl \
  --summary_csv ./summaries/mmstar_vlmeval.csv \
  --scoring vlmeval \
  --judge-model gpt-4o-mini
```

Optional: `--judge-retries 3`, `--judge-sleep 0.5`.

The summary CSV is written to `--summary_csv`; a per-example table is saved alongside with suffix `_per_sample.csv`.

## 5) Syncing with the main AFTER repo

`mmstar_eval.py` here is a copy of `eval/MMStar/mmstar_eval.py`. If you change scoring logic, update the canonical file under `eval/MMStar/` and refresh this copy:

```bash
cp ../../eval/MMStar/mmstar_eval.py ./mmstar_eval.py
```

(or maintain a single source of truth and only ship this folder after copying).

## 6) Citation

Use the MMStar dataset citation from the [dataset card](https://huggingface.co/datasets/Lin-Chen/MMStar). For AFTER’s use of the same evaluator behavior, cite AFTER and VLMEvalKit / OpenCompass as appropriate for your paper.
