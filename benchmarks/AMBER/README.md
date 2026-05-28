# AMBER Benchmark

Generative hallucination benchmark (generative split, 1,004 images).

## Data required (external)

Clone AMBER from https://github.com/junyangwang0410/AMBER and obtain:
- `image/` — the AMBER image set
- `data/annotations.json`, `data/relation.json`, `data/safe_words.txt`, `data/metrics.txt`

## Query list

The generative split queries (ids 1–1004) are in `data/amber_generative.jsonl` at the repo root.

## Scoring

```bash
python eval/amber.py --inference_data <predictions.json> --evaluation_type g \
  --annotation /path/to/AMBER/data/annotations.json \
  --word_association /path/to/AMBER/data/relation.json \
  --safe_words /path/to/AMBER/data/safe_words.txt \
  --metrics /path/to/AMBER/data/metrics.txt
```

Predictions must be a JSON array of `{id, response, response_length}`.
