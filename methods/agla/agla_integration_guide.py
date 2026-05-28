"""
AGLA Integration Guide
===================================
Drop-in AGLA contrastive decoding for InstructBLIP on MMStar.

Prerequisites:
- Pre-computed augmented images at: /scratch/mfazli/augmented_mmstar/{index}.png
- Filenames match MMStar image indices (0.png, 1.png, ..., 1499.png)
- 788 properly augmented, 712 fallback originals (tensor size mismatch on unusual aspect ratios)

Hyperparameters (from AGLA paper):
- alpha = 2.0
- beta = 0.5

IMPORTANT: InstructBLIP (HF) does NOT support past_key_values in forward().
Must do full forward pass each step. Since MMStar answers are 1-5 tokens, this is fine.
"""

import torch
from PIL import Image

# ============================================================
# AGLA contrastive decoding function
# Drop this into your existing InstructBLIP inference loop
# ============================================================

def generate_with_agla(model, processor, raw_image, augmented_image, prompt,
                       alpha=2.0, beta=0.5, max_new_tokens=20, device='cuda'):
    """
    AGLA contrastive decoding for InstructBLIP.
    
    Args:
        model: HuggingFace InstructBlipForConditionalGeneration
        processor: HuggingFace InstructBlipProcessor
        raw_image: PIL Image (original)
        augmented_image: PIL Image (AGLA augmented, loaded from disk)
        prompt: str (the MMStar question)
        alpha: float (AGLA weight, default 2.0)
        beta: float (AGLA cutoff, default 0.5)
    
    Returns:
        response: str
    """
    # Process both images with the same prompt
    inputs = processor(images=raw_image, text=prompt, return_tensors="pt").to(device)
    inputs_cd = processor(images=augmented_image, text=prompt, return_tensors="pt").to(device)

    input_ids = inputs["input_ids"]
    pixel_values = inputs["pixel_values"]
    pixel_values_cd = inputs_cd["pixel_values"]
    qformer_input_ids = inputs.get("qformer_input_ids", None)
    attention_mask = inputs.get("attention_mask", None)

    generated_tokens = []

    for step in range(max_new_tokens):
        # Build full sequence: input + generated tokens so far
        if generated_tokens:
            current_ids = torch.cat([input_ids] + generated_tokens, dim=-1)
            if attention_mask is not None:
                current_mask = torch.cat([
                    attention_mask,
                    torch.ones((1, len(generated_tokens)), dtype=attention_mask.dtype, device=device)
                ], dim=-1)
            else:
                current_mask = None
        else:
            current_ids = input_ids
            current_mask = attention_mask

        # Forward with original image
        with torch.no_grad():
            fwd_kwargs = {"input_ids": current_ids, "pixel_values": pixel_values, "attention_mask": current_mask}
            if qformer_input_ids is not None:
                fwd_kwargs["qformer_input_ids"] = qformer_input_ids
            out = model(**fwd_kwargs)
            logits = out.logits[:, -1, :]

        # Forward with augmented image
        with torch.no_grad():
            fwd_kwargs_cd = {"input_ids": current_ids, "pixel_values": pixel_values_cd, "attention_mask": current_mask}
            if qformer_input_ids is not None:
                fwd_kwargs_cd["qformer_input_ids"] = qformer_input_ids
            out_cd = model(**fwd_kwargs_cd)
            logits_cd = out_cd.logits[:, -1, :]

        # AGLA contrastive formula: logits + alpha * logits_cd
        cutoff = torch.log(torch.tensor(beta, device=device)) + logits.max(dim=-1, keepdim=True).values
        adjusted = logits + alpha * logits_cd
        adjusted = adjusted.masked_fill(logits < cutoff, -float("inf"))
        next_token_id = torch.argmax(adjusted, dim=-1, keepdim=True)

        # Stop at EOS
        if next_token_id.item() == processor.tokenizer.eos_token_id:
            break

        generated_tokens.append(next_token_id)

    # Decode
    if generated_tokens:
        gen_seq = torch.cat(generated_tokens, dim=-1)
        response = processor.tokenizer.decode(gen_seq.squeeze(), skip_special_tokens=True).strip()
    else:
        response = ""

    return response


# ============================================================
# Example usage in your MMStar loop
# ============================================================
"""
import json, os

AUG_DIR = "/scratch/mfazli/augmented_mmstar"

for item in mmstar_data:
    img_path = item["img_path"]
    aug_path = os.path.join(AUG_DIR, img_path.split('/')[-1])
    
    raw_image = Image.open(img_path).convert("RGB")
    
    if os.path.exists(aug_path):
        augmented_image = Image.open(aug_path).convert("RGB")
    else:
        augmented_image = raw_image  # fallback: no augmentation
    
    response = generate_with_agla(
        model, processor, raw_image, augmented_image,
        prompt=item["prompt"],
        alpha=2.0, beta=0.5, max_new_tokens=20
    )
    
    item["model_answer"] = response
"""
