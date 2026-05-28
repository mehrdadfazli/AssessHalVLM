# CHAIR Benchmark

Open-ended captioning hallucination metric on 500 COCO val2014 images.

## Data required (external)

- **Images:** COCO val2014 (`val2014/` directory, ~40K images). Download from https://cocodataset.org/#download
- **Annotations:** COCO `instances_val2014.json` and `captions_val2014.json` (the `annotations/` directory).

## Sampled image list

The exact 500 images we evaluate on (with prompts) are in `data/chair_500.jsonl` at the repo root. Seed = 42.

## Scoring

```bash
python eval/chair.py --coco_path /path/to/coco/annotations --cap_file <predictions.jsonl>
```

Predictions must be JSONL with fields `{image_id, caption}`.
