"""
CAAC (Confidence Aware Attention Calibration): inference-time attention upscaling
and calibration for LLaVA-style and InstructBLIP models.
"""
import logging
import math
from functools import partial
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from PIL import Image

logger = logging.getLogger(__name__)

CAAC_SUPPORTED_MODEL_TYPES = frozenset({"llava", "llava-next", "instructblip"})


def assert_caac_supported(model_type: str) -> None:
    if model_type not in CAAC_SUPPORTED_MODEL_TYPES:
        raise ValueError(
            f"CAAC is only supported for {sorted(CAAC_SUPPORTED_MODEL_TYPES)}; got {model_type!r}."
        )


def compute_attention_factor(exp_config: Dict[str, Any], p: float = 1.0) -> float:
    """Dynamic attention scaling factor lambda from confidence mass p in [0,1]."""
    lamb = p * exp_config["min_lamb"] + (1 - p) * exp_config["max_lamb"]
    return max(lamb, 1.0)


def get_calibration_vector(data: torch.Tensor, beta: float = 0.5) -> torch.Tensor:
    """Blend rows toward uniformity and return per-column calibration ratios."""
    if not 0 <= beta <= 1:
        raise ValueError("Beta must be between 0 and 1.")
    row_sums = torch.sum(data, dim=1, keepdim=True)
    row_averages = row_sums / data.shape[1]
    uniform_distribution = torch.ones_like(data) * row_averages
    transformed_data = (1 - beta) * data + beta * uniform_distribution
    calibration_vectors = transformed_data / data
    return calibration_vectors.mean(dim=0)


def attn_calib_compute(
    attn_maps: Dict[str, torch.Tensor],
    dim1_range: torch.Tensor,
    dim2_range: torch.Tensor,
    beta: float = 0.7,
) -> Dict[int, List[torch.Tensor]]:
    calibration_matrices: Dict[int, List[torch.Tensor]] = {}
    with torch.no_grad():
        for layer_idx in range(len(attn_maps.keys())):
            attn_heads = attn_maps[f"language_model.model.layers.{layer_idx}.self_attn"][0]
            calibration_matrices[layer_idx] = []
            for _head_idx, attn_map in enumerate(attn_heads):
                attn_img = attn_map[dim1_range][:, dim2_range]
                calib_vector = get_calibration_vector(attn_img, beta)
                calibration_matrices[layer_idx].append(calib_vector)
    return calibration_matrices


def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(
        batch, num_key_value_heads, n_rep, slen, head_dim
    )
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def build_ref_image(raw_image: Image.Image, ref_type: str) -> Image.Image:
    """Reference image for calibration forward (same spatial size as query image)."""
    w, h = raw_image.size
    if ref_type == "self":
        return raw_image
    if ref_type == "white":
        return Image.new("RGB", (w, h), (255, 255, 255))
    if ref_type == "black":
        return Image.new("RGB", (w, h), (0, 0, 0))
    if ref_type == "noise":
        arr = np.random.randint(0, 256, (h, w, 3), dtype=np.uint8)
        return Image.fromarray(arr, mode="RGB")
    raise ValueError(f"Unknown ref_image type: {ref_type!r}")


class SelfAttentionModifier:
    """Hooks self-attention: image-token logit upscaling + optional calibration."""

    def __init__(self, model, exp_config, img_token_idxs):
        self.model = model
        self.exp_config = exp_config
        self.img_token_idxs = img_token_idxs
        self.original_forwards = {}
        self.calibration_matrices = None
        self.dynamic_factor = None

    def update_calibration_matrix(
        self,
        attention_maps: Dict[str, torch.Tensor],
        ref_image: Image.Image,
        processor,
        model_type: str,
        query: str,
    ):
        """Run one forward with ref_image and the task query; build per-head calibration vectors."""
        from model_utils import process_inputs

        try:
            attention_maps.clear()
            img_token_id = self.model.config.image_token_index
            inputs = process_inputs(ref_image, query, processor, model_type)
            self.img_token_idxs = torch.nonzero(
                inputs["input_ids"][0] == img_token_id, as_tuple=False
            ).flatten().cpu()

            # InstructBLIP scatters Q-Former LM hidden states into text embeddings; with fp16
            # autocast those tensors disagree (Half vs Float) and PyTorch raises on index_put.
            amp_ctx = (
                torch.cuda.amp.autocast(enabled=False)
                if model_type == "instructblip"
                else torch.cuda.amp.autocast(dtype=torch.float16)
            )
            with torch.no_grad(), amp_ctx:
                outputs_ = self.model(**inputs, output_attentions=True)
                _ = outputs_.logits[0, -1].argmax().item()

            if (
                hasattr(outputs_, "language_model_outputs")
                and hasattr(outputs_.language_model_outputs, "attentions")
                and outputs_.language_model_outputs.attentions
            ):
                attn_tuple = outputs_.language_model_outputs.attentions
            elif hasattr(outputs_, "attentions") and outputs_.attentions:
                attn_tuple = outputs_.attentions
            else:
                raise ValueError("Attention maps are empty.")

            for i, attn in enumerate(attn_tuple):
                attention_maps[f"language_model.model.layers.{i}.self_attn"] = attn

            row_range = torch.tensor(self.exp_config["input_token_idx_calibration"])
            column_range = self.img_token_idxs
            self.calibration_matrices = attn_calib_compute(
                attention_maps, row_range, column_range, self.exp_config["beta"]
            )
            return self.calibration_matrices
        except Exception as e:
            logger.error("Error updating calibration matrix: %s", e)
            raise

    def modify_attention_forward(
        self,
        self_attn,
        hidden_states,
        attention_mask=None,
        position_ids=None,
        past_key_value=None,
        output_attentions=False,
        use_cache=False,
        cache_position=None,
        position_embeddings=None,
        **kwargs,
    ):
        rotary_emb = self_attn.rotary_emb
        bsz, q_len, _ = hidden_states.size()
        query_states = self_attn.q_proj(hidden_states)
        key_states = self_attn.k_proj(hidden_states)
        value_states = self_attn.v_proj(hidden_states)
        query_states = query_states.view(bsz, q_len, -1, self_attn.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, -1, self_attn.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, -1, self_attn.head_dim).transpose(1, 2)

        if position_embeddings is None:
            cos, sin = rotary_emb(value_states, position_ids)
        else:
            cos, sin = position_embeddings

        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_value is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_value.update(
                key_states, value_states, self_attn.layer_idx, cache_kwargs
            )

        key_states = repeat_kv(key_states, self_attn.num_key_value_groups)
        value_states = repeat_kv(value_states, self_attn.num_key_value_groups)
        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(
            self_attn.head_dim
        )

        if attention_mask is not None:
            causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
            attn_weights = attn_weights + causal_mask

        if self_attn.layer_idx in self.exp_config.get("img_txt_cal_layers", []):
            factor = self.dynamic_factor
            if factor is None:
                factor = self.exp_config["compute_attention_factor"](self.exp_config, 1)
            attn_weights[..., self.img_token_idxs] *= factor

        if self_attn.layer_idx in self.exp_config.get("img_cal_layers", []) and self.calibration_matrices is not None:
            col_idx = self.img_token_idxs.tolist()
            if attn_weights.shape[-2] == 1:
                row_idx = [0]
            else:
                row_idx = list(range(self.img_token_idxs[-1].item() + 1, attn_weights.shape[-2]))
            cal_vecs = self.calibration_matrices.get(self_attn.layer_idx, None)
            if cal_vecs is not None:
                cal_vecs_tensor = torch.stack(cal_vecs)
                cal_filter = torch.ones_like(attn_weights, dtype=torch.float16, device=attn_weights.device)
                cal_vecs_expanded = cal_vecs_tensor[None, :, None, :].expand(
                    bsz, -1, len(row_idx), len(col_idx)
                )
                row_idx_tensor = torch.tensor(row_idx, device=attn_weights.device)
                col_idx_tensor = torch.tensor(col_idx, device=attn_weights.device)
                row_idx_grid, col_idx_grid = torch.meshgrid(row_idx_tensor, col_idx_tensor, indexing="ij")
                cal_filter[:, :, row_idx_grid, col_idx_grid] = cal_vecs_expanded
                attn_weights = attn_weights * cal_filter

        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_weights = nn.functional.dropout(
            attn_weights, p=self_attn.attention_dropout, training=self_attn.training
        )
        attn_output = torch.matmul(attn_weights, value_states)
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, -1)
        attn_output = self_attn.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value

    def register_hooks(self):
        layers_to_hook = set(self.exp_config.get("img_txt_cal_layers", [])) | set(
            self.exp_config.get("img_cal_layers", [])
        )
        for l in layers_to_hook:
            layer = self.model.language_model.model.layers[l].self_attn
            self.original_forwards[l] = layer.forward
            layer.forward = partial(self.modify_attention_forward, layer)

    def remove_hooks(self):
        for l, original_fn in self.original_forwards.items():
            self.model.language_model.model.layers[l].self_attn.forward = original_fn
        self.original_forwards.clear()


def dynamic_generate_caac(
    raw_image: Image.Image,
    query: str,
    *,
    model,
    processor,
    model_type: str,
    exp_config: Dict[str, Any],
    do_sample: bool = False,
    num_beams: int = 1,
    logger_: Optional[logging.Logger] = None,
) -> Tuple[torch.Tensor, float]:
    """
    KV-cached token loop with CAAC attention hooks. Returns (new_token_ids, n_repeats_ratio).
    """
    log = logger_ or logger
    assert_caac_supported(model_type)

    from model_utils import process_inputs

    if do_sample:
        raise ValueError("do_sample=True is not supported for CAAC generation (greedy only).")

    inputs = process_inputs(raw_image, query, processor, model_type)
    generated = inputs["input_ids"]
    input_length = generated.shape[-1]
    current_input_ids = generated
    attention_mask = inputs["attention_mask"]
    pixel_values = inputs.get("pixel_values")
    image_sizes = inputs.get("image_sizes")
    qformer_input_ids = inputs.get("qformer_input_ids")
    qformer_attention_mask = inputs.get("qformer_attention_mask")

    img_token_idxs = torch.nonzero(
        inputs["input_ids"][0] == model.config.image_token_index,
        as_tuple=False,
    ).flatten().cpu()

    modifier = SelfAttentionModifier(model, exp_config, img_token_idxs)
    attention_maps: Dict[str, torch.Tensor] = {}
    ref_image = build_ref_image(raw_image, exp_config["ref_image"])

    with torch.no_grad():
        modifier.update_calibration_matrix(attention_maps, ref_image, processor, model_type, query)
    attention_maps.clear()

    modifier.register_hooks()
    n_repeats_forward = 0
    max_new_tokens = int(exp_config["max_new_tokens"])

    def _instructblip_forward_kwargs(seq_ids: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Full-sequence forward; InstructBLIP wrapper does not accept past_key_values."""
        kw: Dict[str, torch.Tensor] = {
            "input_ids": seq_ids,
            "attention_mask": torch.ones(seq_ids.shape, dtype=torch.long, device=seq_ids.device),
        }
        if pixel_values is not None:
            kw["pixel_values"] = pixel_values
        if qformer_input_ids is not None:
            kw["qformer_input_ids"] = qformer_input_ids
        if qformer_attention_mask is not None:
            kw["qformer_attention_mask"] = qformer_attention_mask
        return kw

    try:
        with torch.no_grad():
            if model_type == "instructblip":
                for _ in range(max_new_tokens):
                    inputs_for_forward = _instructblip_forward_kwargs(generated)
                    outputs = model(**inputs_for_forward)
                    last_logits = outputs.logits[:, -1, :]
                    confidence = torch.max(
                        torch.nn.functional.softmax(last_logits, dim=-1)
                    ).item()

                    if confidence > exp_config["confidence_threshold"]:
                        modifier.dynamic_factor = exp_config["compute_attention_factor"](
                            exp_config, 1
                        )
                        next_token = torch.argmax(last_logits, dim=-1, keepdim=True)
                    else:
                        n_repeats_forward += 1
                        modifier.dynamic_factor = exp_config["compute_attention_factor"](
                            exp_config, confidence
                        )
                        outputs = model(**inputs_for_forward)
                        last_logits = outputs.logits[:, -1, :]
                        next_token = torch.argmax(last_logits, dim=-1, keepdim=True)

                    generated = torch.cat([generated, next_token], dim=-1)
                    modifier.dynamic_factor = exp_config["compute_attention_factor"](exp_config, 1)

                    if next_token.item() == processor.tokenizer.eos_token_id:
                        break

                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
            else:
                past_key_values = None
                for _ in range(max_new_tokens):
                    inputs_for_forward = {
                        "input_ids": current_input_ids,
                        "attention_mask": attention_mask,
                        "past_key_values": past_key_values,
                        "use_cache": True,
                    }
                    if past_key_values is None:
                        if pixel_values is not None:
                            inputs_for_forward["pixel_values"] = pixel_values
                        if image_sizes is not None:
                            inputs_for_forward["image_sizes"] = image_sizes
                    else:
                        if pixel_values is not None:
                            inputs_for_forward["pixel_values"] = pixel_values
                        if image_sizes is not None:
                            inputs_for_forward["image_sizes"] = image_sizes

                    outputs = model(**inputs_for_forward)
                    last_logits = outputs.logits[:, -1, :]
                    confidence = torch.max(
                        torch.nn.functional.softmax(last_logits, dim=-1)
                    ).item()

                    if confidence > exp_config["confidence_threshold"]:
                        next_token = torch.argmax(last_logits, dim=-1, keepdim=True)
                        past_key_values = outputs.past_key_values
                    else:
                        n_repeats_forward += 1
                        modifier.dynamic_factor = exp_config["compute_attention_factor"](
                            exp_config, confidence
                        )
                        outputs = model(**inputs_for_forward)
                        last_logits = outputs.logits[:, -1, :]
                        next_token = torch.argmax(last_logits, dim=-1, keepdim=True)
                        past_key_values = outputs.past_key_values

                    generated = torch.cat([generated, next_token], dim=-1)
                    current_input_ids = next_token
                    attention_mask = torch.cat(
                        [
                            attention_mask,
                            torch.ones(
                                (1, 1),
                                dtype=attention_mask.dtype,
                                device=attention_mask.device,
                            ),
                        ],
                        dim=-1,
                    )
                    modifier.dynamic_factor = exp_config["compute_attention_factor"](exp_config, 1)

                    if next_token.item() == processor.tokenizer.eos_token_id:
                        break

                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
    finally:
        modifier.remove_hooks()

    new_tokens = generated[:, input_length:]
    denom = max(1, new_tokens.shape[-1])
    return new_tokens, n_repeats_forward / denom
