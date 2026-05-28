# Appendix: Experimental Settings — VCD, M3ID, AGLA, and Vanilla Baselines

*Draft for paper refinement. Values are taken from the CAAC codebase (VCD/M3ID/Vanilla) and the official AGLA repository (https://github.com/Lackel/AGLA) as of the reported runs. Adjust wording to match your main-text notation.*

---

## A.1 Shared Protocol

All methods evaluated here are **training-free at benchmark time** (inference-time interventions only; no fine-tuning of LVLM weights). Unless noted otherwise:

- **Backbones (7B):** LLaVA-1.5 (`llava-hf/llava-1.5-7b-hf` or `liuhaotian/llava-v1.5-7b`), InstructBLIP-Vicuna (`Salesforce/instructblip-vicuna-7b` or `blip2_vicuna_instruct` via LAVIS), and LLaVA-NeXT (`llava-hf/llava-v1.6-vicuna-7b-hf`).
- **Precision / loading:** 8-bit quantization (`load_in_8bit = true`, BitsAndBytes) with FP16 compute for all CAAC codebase runs (Vanilla, VCD, M3ID). AGLA runs on LLaVA-1.5 and InstructBLIP use FP16 without quantization (AGLA repo default via LAVIS / original LLaVA loader).
- **Hardware:** Single NVIDIA A100 80GB GPU (GMU Hopper cluster, `gpuq` partition, `--qos=gpu`). AGLA runs require ~48GB GPU memory (two models loaded simultaneously: target LVLM + BLIP ITM). Non-AGLA runs use ~20–25GB.
- **Reproducibility:** Fixed random seed (`seed = 42`) for all runs including CHAIR image subsampling (500 COCO val2014 images).

Benchmarks: **CHAIR** (500 images, open-ended captioning), **AMBER** (generative split, 1,004 images), **MMHal-Bench** (96 samples, GPT-4o judge via OpenRouter), and **MMStar** (1,500 multiple-choice questions, Hugging Face `Lin-Chen/MMStar`, `val` split).

---

## A.2 VCD (Visual Contrastive Decoding)

**Reference:** Leng et al., "Mitigating Object Hallucinations in Large Vision-Language Models through Visual Contrastive Decoding," CVPR 2024.
**Codebase used:** the CAAC repository. VCD logic unchanged from original paper; only model-loading patches applied to support LLaVA-NeXT.
**Scope:** LLaVA-NeXT on CHAIR and AMBER (LLaVA-1.5 and InstructBLIP VCD runs performed by collaborator via AvisC codebase).

### A.2.1 VCD Hyperparameters

All values follow the original paper, Section 4.1 Implementation Details. Identical hyperparameters are used across all backbone models.

| Parameter | Symbol | Value | Source |
|-----------|--------|-------|--------|
| Contrast strength | $\alpha$ | 1.0 | VCD paper §4.1 |
| Plausibility threshold | $\beta$ | 0.1 | VCD paper §4.1 |
| DDPM noise step | $t$ | 500 | VCD paper §4.1 |

**Contrastive formula:**

$$\hat{p}_t = \mathrm{softmax}\bigl((1+\alpha)\,\log p_\theta(y_t \mid v, x, y_{<t}) - \alpha\,\log p_\theta(y_t \mid v', x, y_{<t})\bigr)$$

where $v'$ is the noise-corrupted image obtained by applying $t = 500$ DDPM forward-process steps to $v$. Tokens with $\log p_\theta(y_t \mid v, x, y_{<t}) < \log \beta + \max_k \log p_\theta(y_t = k \mid v, x, y_{<t})$ are masked to $-\infty$.

### A.2.2 Decoding Settings

| Setting | Value |
|---------|-------|
| `do_sample` | `false` (greedy) |
| `max_new_tokens` | 512 |
| `num_beams` | 1 |
| `temperature` | N/A |
| `top_p` | N/A |
| `repetition_penalty` | 1.0 |
| `seed` | 42 |

---

## A.3 M3ID (Multi-Modal Mutual Information Decoding)

**Reference:** Favero et al., "Multi-Modal Hallucination Control by Visual Information Grounding," CVPR 2024.
**Codebase used:** CAAC repository. M3ID logic unchanged from original paper.
**Scope:** LLaVA-NeXT on CHAIR and AMBER (LLaVA-1.5 and InstructBLIP M3ID runs performed by collaborator via AvisC codebase).

### A.3.1 M3ID Hyperparameters

| Parameter | Symbol | Value | Source |
|-----------|--------|-------|--------|
| Amplification coefficient | $\lambda$ | 0.2 | CAAC codebase default / M3ID paper |
| Plausibility threshold | $\beta$ | 0.1 | CAAC codebase default / M3ID paper |

**Contrastive formula:**

$$\hat{p}_t = \mathrm{softmax}\bigl((1+\lambda)\,\log p_\theta(y_t \mid v, x, y_{<t}) - \lambda\,\log p_\theta(y_t \mid x, y_{<t})\bigr)$$

where the second term is the text-only forward pass (image tokens removed). Same plausibility cutoff as VCD.

### A.3.2 Decoding Settings

Identical to VCD (Section A.2.2): greedy, `max_new_tokens = 512`, `seed = 42`.

---

## A.4 AGLA (Assembly of Global and Local Attention)

**Reference:** An et al., "AGLA: Mitigating Object Hallucinations in Large Vision-Language Models with Assembly of Global and Local Attention," CVPR 2025.
**Repository:** Official AGLA repo (https://github.com/Lackel/AGLA), with patches described below.
**Scope:** All 3 backbone models on all 4 benchmarks.

### A.4.1 Models and Inference Stack

AGLA requires **two models** simultaneously: the target LVLM and a BLIP Image-Text Matching (ITM) model for prompt-guided image augmentation.

| Component | Model ID / Loader | Notes |
|-----------|-------------------|-------|
| LLaVA-1.5 | `liuhaotian/llava-v1.5-7b` via AGLA's vendored `llava/` | AGLA repo's bundled LLaVA code (transformers 4.34) |
| InstructBLIP | `blip2_vicuna_instruct` `vicuna7b` via LAVIS | AGLA repo's bundled `lavis/` |
| LLaVA-NeXT | `llava-hf/llava-v1.6-vicuna-7b-hf` via HuggingFace | Split pipeline (see §A.4.5) |
| BLIP ITM (augmentation) | `blip_image_text_matching` `large` via LAVIS | Shared across all backbones |

**Environment split:** AGLA's vendored `llava/` and `lavis/` require `transformers == 4.34`. LLaVA-NeXT (`LlavaNextForConditionalGeneration`) requires `transformers >= 4.40`. Two separate Python environments were used:
- `agla-env`: Python 3.10, torch 2.0.1, transformers 4.34.0, salesforce-lavis 1.0.2 (for LLaVA-1.5 + InstructBLIP AGLA, and augmented image pre-computation)
- `lvlm-env`: Python 3.10, torch 2.5.1, transformers 4.47.0 (for LLaVA-NeXT AGLA contrastive decoding)

### A.4.2 AGLA Hyperparameters

All values follow the AGLA paper and official evaluation scripts (`eval/llava1.5_pope.bash`).

| Parameter | Symbol | Value | Source |
|-----------|--------|-------|--------|
| Global/local mixing ratio | $\alpha$ | 2.0 | AGLA paper / eval scripts |
| Plausibility threshold | $\beta$ | 0.5 | AGLA paper / eval scripts |
| GradCAM block number | — | 6 | `augmentation.py` default |
| Image resize for ITM | — | 384 × 384 | `augmentation.py` default |

**Masking ratio computation:** $\text{ratio} = 1 - \text{ITC\_score} / 2$, clamped to $< 1 - 10^{-5}$. GradCAM attention maps from the BLIP ITM model (block 6) are thresholded at the `ratio`-th percentile to create a binary mask. Masked pixels are zeroed, producing the augmented image.

**Contrastive formula (additive, not subtractive like VCD):**

$$\hat{p}_t = \mathrm{softmax}\bigl(\log p_\theta(y_t \mid v, x, y_{<t}) + \alpha \cdot \log p_\theta(y_t \mid v', x, y_{<t})\bigr)$$

where $v'$ is the augmented (prompt-relevant masked) image. Tokens with $\log p_\theta(y_t \mid v, x, y_{<t}) < \log \beta + \max_k \log p_\theta(y_t = k \mid v, x, y_{<t})$ are masked to $-\infty$.

### A.4.3 Decoding Settings

| Setting | AGLA repo (LLaVA-1.5 / InstructBLIP) | AGLA split pipeline (LLaVA-NeXT) |
|---------|---------------------------------------|-----------------------------------|
| `do_sample` | `true` (nucleus sampling) | `false` (greedy, argmax) |
| `temperature` | 1.0 | N/A |
| `top_p` | 1.0 | N/A |
| `max_new_tokens` | 512 (captioning), 20 (MMStar MCQ) | 512 (captioning), 20 (MCQ) |
| `num_beams` | 1 | N/A (token-by-token) |
| `repetition_penalty` | 1.0 | N/A |
| `seed` | 42 | 42 |

Note: AGLA's original evaluation uses `do_sample = true` with `temperature = 1.0` (stochastic decoding), whereas VCD/M3ID use greedy decoding. This follows the AGLA paper's default configuration.

### A.4.4 Benchmark-Specific Generation

| Benchmark | `max_new_tokens` | Prompt | Notes |
|-----------|------------------|--------|-------|
| CHAIR | 512 | `"Please describe this image in detail."` | 500 COCO val2014 images, seed 42 |
| AMBER | 512 | Per-image queries from `query_generative.json` | 1,004 images |
| MMHal-Bench | 512 | Per-image questions from `response_template.json` | 96 images |
| MMStar | 20 | Full MCQ prompt from `mmstar_inputs.jsonl` | 1,500 questions |

### A.4.5 LLaVA-NeXT Split Pipeline (Implementation Detail)

AGLA's vendored `llava/` code supports only LLaVA-1.5 (no AnyRes, no `LlavaNextForConditionalGeneration`). To extend AGLA to LLaVA-NeXT without modifying AGLA's core method, a **two-step split pipeline** was used:

**Step 1 — Pre-compute augmented images** (`agla-env`):
- `precompute_augmentation.py` loads the BLIP ITM model, runs GradCAM-based augmentation on each image+question pair, and saves augmented images as JPEG/PNG files to disk.
- This step is model-agnostic (does not load the target LVLM).
- For MMStar, 788/1,500 images were successfully augmented; 712 fell back to saving the original image due to tensor dimension mismatches on unusual aspect ratios. Fallback images effectively disable AGLA for those samples (contrastive decoding amplifies vanilla logits).

**Step 2 — Contrastive decoding** (`lvlm-env`):
- `run_agla_llavanext.py` loads LLaVA-NeXT via HuggingFace, performs token-by-token generation with two parallel forward passes (original image and augmented image loaded from disk), and applies the AGLA contrastive formula at each step.
- Augmented images are resized to match original image dimensions before processing (required for LLaVA-NeXT's AnyRes to produce matching image token counts).
- KV caching is used for LLaVA-NeXT (supported by `LlavaNextForConditionalGeneration`).

**AGLA + InstructBLIP + MMStar:** Run via a HuggingFace-based InstructBLIP pipeline with left-truncated prompts. LAVIS-based InstructBLIP cannot handle MMStar's long MCQ prompts (200–300 tokens), producing empty responses regardless of AGLA. This is a LAVIS limitation, not an AGLA-specific issue.

### A.4.6 Patches Applied to AGLA Repository

1. `lavis/__init__.py`: Replaced wildcard imports (`from lavis.models import *`) with specific imports (`from lavis.models import load_model_and_preprocess, load_model, load_preprocess`) to avoid pulling in unused models with unmet dependencies (e.g., `blip_diffusion` requiring `diffusers`).
2. `lavis/models/__init__.py`: Commented out ~15 unused model imports (`BlipDiffusion`, `Img2PromptVQA`, `AlbefClassification`, etc.) that required `spacy`, `diffusers`, or other packages not needed for AGLA.
3. `eval/run_llava_chair.py` and `eval/run_instructblip_chair.py`: Created from POPE evaluation scripts by removing the `" Please answer this question with one word."` prompt suffix.
4. `eval/precompute_augmentation.py`: New script for the split pipeline; saves original image as fallback when augmentation fails (tensor size mismatch on unusual aspect ratios).

---

## A.5 Vanilla Baselines

**Codebase used:** CAAC repository for all three backbones, with model-loading patches for LLaVA-NeXT support.

### A.5.1 Decoding Settings

| Setting | Value |
|---------|-------|
| `do_sample` | `false` (greedy) |
| `max_new_tokens` | 512 |
| `num_beams` | 1 |
| `temperature` | N/A |
| `top_p` | N/A |
| `repetition_penalty` | 1.0 |
| `seed` | 42 |

No mitigation method applied. Same model checkpoints, quantization, and benchmark prompts as the method runs.

---

## A.6 Evaluation Protocols

| Benchmark | Metric / Judge | Script | Notes |
|-----------|----------------|--------|-------|
| CHAIR | CHAIRs, CHAIRi, Recall, Len | `evals/chair.py` | COCO `instances_val2014.json` annotations; input keyed as `{image_id, caption}` |
| AMBER | CHAIR, Cover, Hal, Cog, Len | `evals/inference.py` (`--evaluation_type g`) | Generative split only (ids 1–1,004); `response_length` field required for Len metric |
| MMHal-Bench | Avg. Score (0–6), Hallucination Rate | `eval_mmhal_openrouter.py` | **GPT-4o** via OpenRouter, temperature 0, 96 items. Total API cost ~$0.45 per model |
| MMStar | Per-category accuracy, Overall accuracy | `mmstar_eval.py` (`--scoring exact`) | Heuristic letter extraction; 6 categories × 250 questions. No API needed |

### Output Format Conversions

AGLA outputs require format conversion before scoring:
- **CHAIR:** AGLA JSONL (`{question_id, text, image}`) → CHAIR JSONL (`{image_id, caption}`). Image ID extracted from filename: `COCO_val2014_000000XXXXXX.jpg` → `XXXXXX`.
- **AMBER:** AGLA JSONL → AMBER JSON array (`[{id, response, response_length}]`). `response_length = len(text.split())`.
- **MMStar:** No conversion needed (same schema).
- **MMHal-Bench:** No conversion needed (uses `response_template.json` format with `model_answer` field).

---

## A.7 Cross-Method Comparison (Quick Reference)

| | **VCD** | **M3ID** | **AGLA** |
|---|---------|----------|----------|
| Training at benchmark time | No | No | No |
| Intervention type | Contrastive decoding (noisy image) | Contrastive decoding (no-image) | Contrastive decoding (augmented image) |
| Extra model at inference | None | None | BLIP ITM ("large") |
| Key scalars | $\alpha{=}1$, $\beta{=}0.1$, $t{=}500$ | $\lambda{=}0.2$, $\beta{=}0.1$ | $\alpha{=}2$, $\beta{=}0.5$ |
| Contrastive direction | Subtractive ($-$) | Subtractive ($-$) | Additive ($+$) |
| Decoding | Greedy | Greedy | Sampling (original) / Greedy (LLaVA-NeXT split) |
| GPU memory overhead | ~0 (noise is cheap) | ~0 (text-only pass) | ~1.7GB (BLIP ITM) |

---

## A.8 Implementation Notes

**Software.** Python 3.10; two environments used:
- `lvlm-env`: PyTorch 2.5.1, transformers 4.47.0 (Vanilla, VCD, M3ID via CAAC; LLaVA-NeXT AGLA contrastive decoding; all scoring)
- `agla-env`: PyTorch 2.0.1, transformers 4.34.0, salesforce-lavis 1.0.2 (AGLA augmentation and LLaVA-1.5/InstructBLIP AGLA generation)

**Paths.** Model weights cached under `/path/to/.cache/huggingface/`. COCO val2014 images at `/path/to/LVLM/datasets/coco2014/val2014/` (40,504 images). AMBER images at `/path/to/LVLM/datasets/AMBER/image/` (1,004 images). MMHal-Bench images at `/path/to/LVLM/datasets/mmhal-bench/images/` (96 images). MMStar images at `/path/to/LVLM/datasets/MMStar/images/` (1,500 images). Pre-computed AGLA augmented images stored per-benchmark under `/path/to/LVLM/lvlm-logs/AGLA/augmented_{chair,amber,mmhal,mmstar}/`.

**CAAC codebase patches (for VCD/M3ID/Vanilla on LLaVA-NeXT):**
1. `src/model_utils.py`: Added `LlavaNextForConditionalGeneration` class support and `process_inputs` function for the `llava-next` model type.
2. `src/sampling_utils.py`: `_model_call()` wrapper dynamically filters unsupported keyword arguments (e.g., `image_sizes` for LLaVA-1.5, `past_key_values` for InstructBLIP) to enable a unified generation loop across architectures.
3. `experiments/run_chair_baselines.py` and `run_amber_baselines.py`: Added `"llava"` and `"instructblip"` to model choices; added resume support (skips completed image IDs).

---

*End of draft. Replace notation symbols if they differ from main-text conventions.*
