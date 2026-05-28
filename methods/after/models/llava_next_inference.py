import copy
import os
from functools import partial

import numpy as np
import torch
from PIL import Image
from transformers import LlavaNextForConditionalGeneration, LlavaNextProcessor

from baukit import TraceDict
from models.base import Mllm
from models.llama.modeling_llama import replace_llama_modality_adaptive


class Llava_next(Mllm):
    def __init__(self, model_name_or_path, **kwargs):
        # Critical: ensure `head_out` exists on every LlamaAttention before loading.
        replace_llama_modality_adaptive()

        cache_dir = kwargs.get("cache_dir", None)
        # Some cluster environments are offline but have a shared HF cache populated.
        # Allow callers / env to force offline mode, otherwise prefer normal HF resolution.
        local_files_only = kwargs.get("local_files_only", False)
        if os.environ.get("HF_HUB_OFFLINE") == "1":
            local_files_only = True

        self.processor = LlavaNextProcessor.from_pretrained(
            model_name_or_path,
            cache_dir=cache_dir,
            local_files_only=local_files_only,
        )

        use_cuda = torch.cuda.is_available()
        torch_dtype = kwargs.get("torch_dtype", torch.float16 if use_cuda else torch.float32)
        device_map = kwargs.get("device_map", "auto" if use_cuda else "cpu")
        self.model = LlavaNextForConditionalGeneration.from_pretrained(
            model_name_or_path,
            low_cpu_mem_usage=True,
            torch_dtype=torch_dtype,
            device_map=device_map,
            cache_dir=cache_dir,
            local_files_only=local_files_only,
        )
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.path = model_name_or_path

    def _postprocess_caption(self, text: str) -> str:
        """Extract just the assistant caption (avoid echoing prompt/chat template)."""
        if not isinstance(text, str):
            return str(text)
        # Common chat-template markers.
        for marker in ("ASSISTANT:", "Assistant:", "assistant:"):
            if marker in text:
                text = text.split(marker)[-1]
        return text.strip()

    def _decode_generated_only(self, output_ids: torch.Tensor, input_len: int) -> str:
        """Decode only newly generated tokens (preferred) with a safe fallback."""
        try:
            gen_ids = output_ids[:, input_len:]
            text = self.processor.batch_decode(gen_ids, skip_special_tokens=True)[0]
        except Exception:
            text = self.processor.batch_decode(output_ids, skip_special_tokens=True)[0]
        return self._postprocess_caption(text)

    def _build_inputs(self, prompt: str, filepath: str):
        image = Image.open(filepath).convert("RGB")

        # If prompt is already in AFTER's LLaVA-ish format (contains <image> and roles),
        # avoid re-wrapping with a chat template.
        if "<image>" in prompt and "ASSISTANT" in prompt:
            text = prompt
        else:
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": prompt},
                    ],
                }
            ]
            text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )

        inputs = self.processor(images=image, text=text, return_tensors="pt").to(self.device)
        return inputs

    def evaluate(self, prompt, filepath):
        inputs = self._build_inputs(prompt, filepath)
        with torch.inference_mode():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=getattr(self, "max_new_tokens", 128),
                do_sample=False,
                use_cache=False,
            )
        input_len = int(inputs["input_ids"].shape[-1])
        return self._decode_generated_only(output_ids, input_len=input_len)

    def get_activations(self, prompt, filepath):
        num_layers = self.model.config.text_config.num_hidden_layers
        heads = [
            f"language_model.model.layers.{i}.self_attn.head_out" for i in range(num_layers)
        ]

        inputs = self._build_inputs(prompt, filepath)

        with torch.inference_mode():
            with TraceDict(self.model, heads) as ret:
                output = self.model(**inputs, output_hidden_states=True)

            if hasattr(output, "language_model_outputs") and hasattr(
                output.language_model_outputs, "hidden_states"
            ):
                hidden_states = output.language_model_outputs.hidden_states
            else:
                # Some transformers versions return LlavaNextCausalLMOutputWithPast directly.
                hidden_states = output.hidden_states
            hidden_states = torch.stack(hidden_states, dim=0).squeeze()
            hidden_states = hidden_states.detach().cpu().numpy()

            head_wise_hidden_states = [
                ret[h].output.squeeze().detach().cpu() for h in heads
            ]
            head_wise_hidden_states = torch.stack(head_wise_hidden_states, dim=0).squeeze()
            head_wise_hidden_states = head_wise_hidden_states.numpy()

        return hidden_states, head_wise_hidden_states, None, None

    def get_activations_only_text(self, prompt):
        # Text-only path: run through `language_model` directly (no images).
        num_layers = self.model.config.text_config.num_hidden_layers
        heads = [f"model.layers.{i}.self_attn.head_out" for i in range(num_layers)]

        inputs = self.processor(text=prompt, return_tensors="pt").to(self.device)
        input_ids = inputs["input_ids"]
        inputs_embeds = self.model.language_model.get_input_embeddings()(input_ids)

        with torch.inference_mode():
            with TraceDict(self.model.language_model, heads) as ret:
                output = self.model.language_model(
                    inputs_embeds=inputs_embeds, output_hidden_states=True
                )

            hidden_states = output.hidden_states
            hidden_states = torch.stack(hidden_states, dim=0).squeeze()
            hidden_states = hidden_states.detach().cpu().numpy()

            head_wise_hidden_states = [
                ret[h].output.squeeze().detach().cpu() for h in heads
            ]
            head_wise_hidden_states = torch.stack(head_wise_hidden_states, dim=0).squeeze()
            head_wise_hidden_states = head_wise_hidden_states.numpy()

        return hidden_states, head_wise_hidden_states, None

    def evaluate_with_intervention_youare_offset(self, prompt, filepath, interventions, intervention_fn):
        def _id(head_output, layer_name):
            return head_output

        inputs = self._build_inputs(prompt, filepath)

        num_layers = self.model.config.text_config.num_hidden_layers
        num_heads = self.model.config.text_config.num_attention_heads
        head_dim = self.model.config.text_config.hidden_size // num_heads
        heads = [
            f"language_model.model.layers.{i}.self_attn.head_out" for i in range(num_layers)
        ]

        # 1) Get query head_out activations (for the last token) to compute QAO offsets.
        with torch.inference_mode():
            with TraceDict(self.model, heads) as ret:
                _ = self.model(**inputs, output_hidden_states=True)
        query_hidden_states = [ret[h].output.squeeze() for h in heads]  # per-layer (seq, hidden)

        # 2) Create a per-sample interventions dict with offset-adjusted directions.
        interventions_iter = copy.deepcopy(interventions)
        for name, interv in interventions_iter.items():
            layer = int(name.split(".")[-3])
            q_last = query_hidden_states[layer][-1].reshape(num_heads, head_dim)
            for i, (head, direction, proj_val_std, generator) in enumerate(interv):
                offset = generator(q_last[head].float())
                offset = offset.half()
                direction = direction + offset.detach().cpu().numpy()
                direction = direction / (np.linalg.norm(direction) + 1e-12)
                interv[i] = (head, direction, proj_val_std)

        if not interventions_iter:
            intervene = _id
            layers_to_intervene = []
        else:
            intervene = partial(
                intervention_fn, start_edit_location="lt", interventions=interventions_iter
            )
            layers_to_intervene = list(interventions_iter.keys())

        with TraceDict(self.model, layers_to_intervene, edit_output=intervene):
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=getattr(self, "max_new_tokens", 128),
                do_sample=False,
                use_cache=False,
            )
        input_len = int(inputs["input_ids"].shape[-1])
        return self._decode_generated_only(output_ids, input_len=input_len)

