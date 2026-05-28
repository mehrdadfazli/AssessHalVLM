# MMHal-Bench

Hallucination QA benchmark (96 questions, GPT-4o judge).

## Data required

Download from HuggingFace:

```python
from huggingface_hub import snapshot_download
snapshot_download(repo_id='Shengcao1006/MMHal-Bench', repo_type='dataset',
                  local_dir='data/mmhal-bench')
```

Provides `response_template.json` (96 questions) and `images/` (96 images).

## Scoring (USES PAID API)

GPT-4o judge via OpenRouter (~$0.45 per model/method):

```bash
python eval/eval_mmhal_openrouter.py \
  --response <predictions.json> --evaluation <out_eval.json> \
  --api-key $OPENROUTER_API_KEY --model-name "<MODEL> <METHOD>"
```

Run `eval/test_mmhal_single_call.py` first to verify the key and measure exact per-call cost.
The original OpenAI GPT-4 judge is in `eval/eval_gpt4.py`.
