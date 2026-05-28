import argparse
import json
import os
import sys
import torch
from tqdm import tqdm
from PIL import Image
from transformers import set_seed
from torchvision import transforms

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria
from lavis.models import load_model_and_preprocess
from lavis.common.registry import registry
from sample import evolve_agla_sampling
from augmentation import augmentation

evolve_agla_sampling()


def run_llava(args, data, model_itm, image_processors, text_processors):
    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, None, model_name)
    device = 'cuda'
    loader = transforms.Compose([transforms.ToTensor()])

    for item in tqdm(data, desc="AGLA LLaVA-1.5 on MMHal"):
        image_filename = item["image_src"].split("/")[-1]
        image_path = os.path.join(args.image_folder, image_filename)

        if not os.path.exists(image_path):
            item["model_answer"] = ""
            continue

        raw_image = Image.open(image_path).convert("RGB")
        question = item["question"]

        if model.config.mm_use_im_start_end:
            qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + question
        else:
            qs = DEFAULT_IMAGE_TOKEN + '\n' + question

        conv = conv_templates[args.conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()
        raw_image_tensor = image_processor.preprocess(raw_image, return_tensors='pt')['pixel_values'][0]

        if args.use_agla:
            tensor_image = loader(raw_image.resize((384, 384)))
            image = image_processors["eval"](raw_image).unsqueeze(0).to(device)
            q_text = text_processors["eval"](question)
            tokenized_text = model_itm.tokenizer(q_text, padding='longest', truncation=True, return_tensors="pt").to(device)
            augmented_image = augmentation(image, q_text, tensor_image, model_itm, tokenized_text, raw_image)
            image_tensor = image_processor.preprocess(augmented_image, return_tensors='pt')['pixel_values'][0]
        else:
            image_tensor = None

        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        stopping_criteria = KeywordsStoppingCriteria([stop_str], tokenizer, input_ids)

        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                images=raw_image_tensor.unsqueeze(0).half().cuda(),
                images_cd=(image_tensor.unsqueeze(0).half().cuda() if image_tensor is not None else None),
                cd_alpha=args.alpha,
                cd_beta=args.beta,
                do_sample=True,
                temperature=args.temperature,
                top_p=args.top_p,
                max_new_tokens=512,
                use_cache=True
            )

        input_token_len = input_ids.shape[1]
        outputs = tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)[0].strip()
        if outputs.endswith(stop_str):
            outputs = outputs[:-len(stop_str)].strip()

        item["model_answer"] = outputs
        torch.cuda.empty_cache()

    return data


def run_instructblip(args, data, model_itm, image_processors, text_processors):
    disable_torch_init()
    device = 'cuda'
    model, vis_processors, _ = load_model_and_preprocess(
        name="blip2_vicuna_instruct", model_type="vicuna7b", is_eval=True, device=device
    )
    loader = transforms.Compose([transforms.ToTensor()])

    for item in tqdm(data, desc="AGLA InstructBLIP on MMHal"):
        image_filename = item["image_src"].split("/")[-1]
        image_path = os.path.join(args.image_folder, image_filename)

        if not os.path.exists(image_path):
            item["model_answer"] = ""
            continue

        raw_image = Image.open(image_path).convert("RGB")
        question = item["question"]
        image_tensor = vis_processors["eval"](raw_image).unsqueeze(0).to(device)

        if args.use_agla:
            tensor_image = loader(raw_image.resize((384, 384)))
            image = image_processors["eval"](raw_image).unsqueeze(0).to(device)
            q_text = text_processors["eval"](question)
            tokenized_text = model_itm.tokenizer(q_text, padding='longest', truncation=True, return_tensors="pt").to(device)
            augmented_image = augmentation(image, q_text, tensor_image, model_itm, tokenized_text, raw_image)
            image_tensor_cd = vis_processors["eval"](augmented_image).unsqueeze(0).to(device)
        else:
            image_tensor_cd = None

        with torch.inference_mode():
            outputs = model.generate(
                {"image": image_tensor, "prompt": question},
                use_nucleus_sampling=True,
                num_beams=1,
                top_p=args.top_p,
                repetition_penalty=1,
                images_cd=image_tensor_cd,
                cd_alpha=args.alpha,
                cd_beta=args.beta,
                temperature=args.temperature,
                max_length=512
            )

        item["model_answer"] = outputs[0]
        torch.cuda.empty_cache()

    return data


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type", type=str, choices=["llava", "instructblip"], required=True)
    parser.add_argument("--model-path", type=str, default="liuhaotian/llava-v1.5-7b")
    parser.add_argument("--mmhal_path", type=str, default="/path/to/LVLM/datasets/mmhal-bench")
    parser.add_argument("--image-folder", type=str, default="/path/to/LVLM/datasets/mmhal-bench/images")
    parser.add_argument("--output_file", type=str, required=True)
    parser.add_argument("--conv-mode", type=str, default="llava_v1")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_p", type=float, default=1)
    parser.add_argument("--use_agla", action='store_true', default=True)
    parser.add_argument("--alpha", type=float, default=2.0)
    parser.add_argument("--beta", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    set_seed(args.seed)

    # Load MMHal data
    with open(os.path.join(args.mmhal_path, "response_template.json")) as f:
        data = json.load(f)
    print(f"Loaded {len(data)} questions")

    # Load BLIP ITM model (shared by both model types)
    device = 'cuda'
    model_itm, image_processors, text_processors = load_model_and_preprocess(
        "blip_image_text_matching", "large", device=device, is_eval=True
    )

    # Run the appropriate model
    if args.model_type == "llava":
        data = run_llava(args, data, model_itm, image_processors, text_processors)
    elif args.model_type == "instructblip":
        data = run_instructblip(args, data, model_itm, image_processors, text_processors)

    # Save
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    with open(args.output_file, "w") as f:
        json.dump(data, f, indent=2)
    print(f"Saved to {args.output_file}")
