import argparse
import torch
import os
import json
from tqdm import tqdm
import shortuuid
import sys
import os
import random
import numpy as np
import torch.backends.cudnn as cudnn
import torch.distributed as dist

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + '/experiments')
# print(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria

from utils import dist_util
from utils.logger import create_logger
from glob import glob

from PIL import Image
import math

# MMHal-Bench uses direct JSONL load (no AMBERDataSet-equivalent class needed).

# import kornia
from transformers import set_seed
from avisc_utils.vcd_add_noise import add_diffusion_noise
from avisc_utils.avisc_sample import evolve_avisc_sampling
evolve_avisc_sampling()

torch.multiprocessing.set_sharing_strategy('file_system')


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


# (No recorder() function — that was AMBER-specific for discriminative yes/no
#  conversion. MMHal-Bench is open-ended QA; the judge consumes the raw model_answer.)


def parse_args():
    parser = argparse.ArgumentParser(description="MMHal-Bench open-ended QA on LLaVA with hallucination mitigation")
    parser.add_argument("--model-path", type=str, default="path/checkpoints/llava-v1.5-7b")
    parser.add_argument("--model-base", type=str, default=None)

    parser.add_argument("--conv-mode", type=str, default="llava_v1")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_p", type=float, default=1)
    parser.add_argument("--top_k", type=int, default=None)

    parser.add_argument("--jsonl_path", type=str, default="/path/to/LVLM/datasets/MMHal-Bench/mmhal_inputs.jsonl")
    parser.add_argument("--data_path", type=str, default="/path/to/LVLM/datasets/MMHal-Bench/images")
    parser.add_argument("--log_path", type=str, default="/path/to/LVLM/lvlm-logs/MMHal-Bench/llava")

    parser.add_argument("--noise_step", type=int, default=500)
    parser.add_argument("--use_cd", type=str2bool, default=False)
    parser.add_argument("--cd_alpha", type=float, default=1.0)
    parser.add_argument("--cd_beta", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gpu-id", type=int, default=7, help="specify the gpu to load the model.")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=1)

    parser.add_argument("--use_avisc", type=str2bool, default=True)
    parser.add_argument("--layer_gamma", type=float, default=0.5)
    parser.add_argument("--masking_scheme", type=str, default="zeros")
    parser.add_argument("--lamb", type=float, default=0.99)
    parser.add_argument("--exp_description", type=str, default="..")
    parser.add_argument("--max_token", type=int, default=512)
    parser.add_argument("--use_m3id", type=str2bool, default=False)



    args = parser.parse_args()
    return args


def setup_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    cudnn.benchmark = False
    cudnn.deterministic = True


def main():
    args = parse_args()

    # Setup DDP:
    dist_util.setup_dist(args)
    device = dist_util.device()

    # Setup an experiment folder:
    if dist.get_rank() == 0:
        os.makedirs(
            args.log_path, exist_ok=True
        )  # Make results folder (holds all experiment subfolders)
        model_string_name = args.model_path.split("/")[-1]
        experiment_dir = f"{args.log_path}"  # Create an experiment folder
        os.makedirs(experiment_dir, exist_ok=True)
        logger = create_logger(experiment_dir)
        logger.info(f"Experiment directory created at {experiment_dir}")
        logger.info(f"exp_description: {args.exp_description}")
    else:
        logger = create_logger(None)

    # ========================================
    #      Saving Command-line arguments
    # ========================================
    args_dict = vars(args)
    args_path = os.path.join(experiment_dir, "command_line_args.json")
    with open(args_path, "w") as f:
        json.dump(args_dict, f, indent=4)
    logger.info(f"Command-line arguments saved to: {args_path}")

    # ========================================
    #             Model Initialization
    # ========================================
    print('Initializing Model')
    logger.info(f"use_cd: {args.use_cd}, use_avisc: {args.use_avisc}, use_M3ID: {args.use_m3id}, layer_gamma: {args.layer_gamma}, cd_alpha: {args.cd_alpha}, masking_scheme: {args.masking_scheme}, lamb: {args.lamb}")



    #### for avisc
    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, args.model_base, model_name)

    tokenizer.padding_side = "left"

    # ==============================================================
    # MMHal-Bench data loading: direct JSONL read (one record per line).
    # Input schema per line: {index, question_type, question_topic,
    #   image_id, image_src, image_content, question, gt_answer, img_path}.
    # `img_path` is already absolute (set by mmhal data prep).
    # Open-ended QA — no MCQ suffix; we feed `question` verbatim to the model.
    # `gt_answer` + `image_content` are kept in the output for the GPT-4o judge.
    # ==============================================================
    with open(args.jsonl_path, 'r', encoding='utf-8') as f:
        data_list = [json.loads(line) for line in f if line.strip()]
    print(f"Loaded {len(data_list)} items from {args.jsonl_path}")

    predictions_path = os.path.join(experiment_dir, "predictions.jsonl")

    # Resume: read existing predictions.jsonl if present (skip already-processed indices)
    predictions = []
    processed_indices = set()
    if os.path.exists(predictions_path):
        try:
            with open(predictions_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    predictions.append(json.loads(line))
            processed_indices = {r['index'] for r in predictions}
            print(f"Resuming from {len(processed_indices)} already-processed items")
        except (json.JSONDecodeError, KeyError, IOError) as e:
            print(f"Warning: existing {predictions_path} unreadable ({e}); starting fresh")
            predictions = []
            processed_indices = set()

    print("Start eval...")

    for item in tqdm(data_list, desc="Processing MMHal-Bench"):
        if item['index'] in processed_indices:
            continue

        # Inline AMBERDataSet.__getitem__ (model='llava' branch) — verbatim:
        # NO .convert("RGB") for LLaVA; image_processor handles internally.
        # img_path is already absolute (set by mmhal data prep).
        image_path_str = item['img_path']
        raw_image = Image.open(image_path_str)
        image_one = image_processor.preprocess(raw_image, return_tensor='pt')['pixel_values'][0]
        # image_processor returns numpy.ndarray here (return_tensor='pt' is a typo;
        # correct kwarg is return_tensors='pt' with 's'). Original code worked because
        # DataLoader.default_collate_fn auto-converted numpy→tensor; bypassing DataLoader
        # means we replicate that conversion explicitly. (Bug 8 carryover from AMBER patch.)
        if not isinstance(image_one, torch.Tensor):
            image_one = torch.from_numpy(image_one)

        # Construct batch-style dict the existing loop body expects.
        # (--batch-size 1 always; this is just a 1-element wrapping.)
        image = image_one.unsqueeze(0)
        qs = [item['question']]
        ids = [item['index']]
        image_path = [image_path_str]

        # ==============================================
        #             Text prompt setting
        # ==============================================

        if model.config.mm_use_im_start_end:
            qu = [DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + _ for _ in qs]
        else:
            qu = [DEFAULT_IMAGE_TOKEN + '\n' + _ for _ in qs]

        input_ids = []

        for i in range(args.batch_size):
            conv = conv_templates[args.conv_mode].copy()
            conv.append_message(conv.roles[0], qu[i])
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()

        # ==============================================
        #             Image tensor setting
        # ==============================================

            input_id = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()

            input_ids.append(
                input_id
            )

        def make_batch(input_ids):
            input_ids = [_.squeeze(0) for _ in input_ids]
            max_len = max([_.shape[0] for _ in input_ids])
            input_ids = [torch.cat([torch.zeros(max_len - _.shape[0], dtype=torch.long).cuda(), _], dim=0) for _ in input_ids]
            return torch.stack(input_ids, dim=0)

        input_ids = make_batch(input_ids)
        image_tensor = image


        # ==============================================
        #             avisc method setting
        # ==============================================
        if args.use_cd:
            image_tensor_cd = add_diffusion_noise(image_tensor, noise_step=500)
        else:
            image_tensor_cd = None

        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        keywords = [stop_str]
        stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)

        with torch.inference_mode():
            with torch.no_grad():
                output_ids = model.generate(
                    input_ids,
                    images=image_tensor.half().cuda(),
                    images_cd=(image_tensor_cd.half().cuda() if image_tensor_cd is not None else None),
                    cd_alpha=args.cd_alpha,
                    cd_beta=args.cd_beta,
                    do_sample=True,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    top_k=args.top_k,
                    max_new_tokens=args.max_token,
                    use_cache=True,
                    use_avisc=args.use_avisc,
                    layer_gamma=args.layer_gamma,
                    masking_scheme=args.masking_scheme,
                    lamb=args.lamb,
                    use_m3id=args.use_m3id,
                )

                input_token_len = input_ids.shape[1]
                n_diff_input_output = (input_ids != output_ids[:, :input_token_len]).sum().item()
                if n_diff_input_output > 0:
                    print(f'[Warning] {n_diff_input_output} output_ids are not the same as the input_ids')
                outputs = tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)
                outputs = [_.strip() for _ in outputs]
                outputs = [_[:-len(stop_str)] if _.endswith(stop_str) else _ for _ in outputs]


                for ip, q, a in zip(image_path, qs, outputs):
                    logger.info(f"[{ip}]")
                    logger.info(f"Q: {q}")
                    logger.info(f"A: {a}")

                # MMHal-Bench output: preserve all input fields + add `model_answer`.
                # Field name is `model_answer` per MMHal-Bench's schema (matches the
                # judge script's expected field; original `response_template.json`
                # ships an empty `model_answer` placeholder per record).
                # batch_size=1 always → outputs is a single-element list.
                a = outputs[0]
                record = {**item, "model_answer": a}
                predictions.append(record)

                # Per-item save for resume support (write JSONL after each item)
                with open(predictions_path, 'w', encoding='utf-8') as f:
                    for r in predictions:
                        f.write(json.dumps(r, ensure_ascii=False) + '\n')

    # Final save fallback (redundant if loop completed)
    with open(predictions_path, 'w', encoding='utf-8') as f:
        for r in predictions:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')

if __name__ == "__main__":
    main()
