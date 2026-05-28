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

from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria
from transformers import set_seed, AutoTokenizer, AutoModelForCausalLM

from utils import dist_util
from utils.logger import create_logger
from glob import glob

from PIL import Image
import math

# MMStar uses direct JSONL load (no AMBERDataSet-equivalent class needed).
from lavis.models import load_model_and_preprocess
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
#  conversion. MMStar is MCQ; scoring extracts option letters from response text.)


def parse_args():
    parser = argparse.ArgumentParser(description="MMStar MCQ evaluation on InstructBLIP with hallucination mitigation")
    parser.add_argument("--model-path", type=str, default="path/checkpoints/instruct_blip")
    parser.add_argument("--model-base", type=str, default=None)

    parser.add_argument("--conv-mode", type=str, default="llava_v1")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--top_k", type=int, default=None)

    parser.add_argument("--jsonl_path", type=str, default="path/to/MMStar/mmstar_inputs.jsonl")
    parser.add_argument("--data_path", type=str, default="path/to/MMStar/images")
    parser.add_argument("--log_path", type=str, default="path/logs/mmstar")

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
    parser.add_argument("--max_token", type=int, default=64)
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


    disable_torch_init()
    model, vis_processors, _ = load_model_and_preprocess(name="blip2_vicuna_instruct", model_type="vicuna7b", is_eval=True, device=device)

    # ——— use model.llm_tokenizer (LAVIS) — no separate tokenizer introduced ———
    tokenizer = model.llm_tokenizer


    # ==============================================================
    # MMStar data loading: direct JSONL read (one record per line).
    # Input schema per line: {index, question, prompt, answer, img_path,
    #   category, l2_category}. `img_path` is already absolute (set by
    #   prepare_mmstar_data.py). We feed `prompt` (vlmeval-suffixed) to
    #   the model; `question` (raw) is kept in the output for scoring.
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

    for item in tqdm(data_list, desc="Processing MMStar"):
        if item['index'] in processed_indices:
            continue

        # Inline AMBERDataSet.__getitem__ (model='instructblip' branch) — verbatim:
        # YES .convert("RGB") for InstructBLIP (unlike LLaVA path).
        # img_path is already absolute (set by prepare_mmstar_data.py).
        image_path_str = item['img_path']
        raw_image = Image.open(image_path_str).convert("RGB")
        image_one = vis_processors['eval'](raw_image)

        # Construct batch-style dict the existing loop body expects.
        # (--batch-size 1 always; this is just a 1-element wrapping.)
        image = image_one.unsqueeze(0)
        qs = [item['prompt']]
        ids = [item['index']]
        image_path = [image_path_str]

        # ==============================================
        #             Text prompt setting
        # ==============================================

        if args.use_cd:
            image_tensor_cd = add_diffusion_noise(image, args.noise_step)
        else:
            image_tensor_cd = None

        input_ids = []


        # ==============================================
        #             Image tensor setting
        # ==============================================


        with torch.inference_mode():
            outputs = model.generate(
                {"image": image.to(device), "prompt": qs[0]},
                use_nucleus_sampling=True,
                num_beams=1,
                top_p=args.top_p,
                repetition_penalty=1,
                images_cd=image_tensor_cd.half().to(device) if image_tensor_cd is not None else None,
                cd_beta = args.cd_beta,
                use_avisc=args.use_avisc,
                layer_gamma=args.layer_gamma,
                masking_scheme=args.masking_scheme,
                lamb=args.lamb,
                max_length=args.max_token,
                cd_alpha=args.cd_alpha,
                use_m3id=args.use_m3id,
                )
            outputs = outputs

            for ip, q, a in zip(image_path, qs, outputs):
                    logger.info(f"[{ip}]")
                    logger.info(f"Q: {q}")
                    logger.info(f"A: {a}")

            # MMStar output: preserve all input fields + add `response`.
            # (No recorder() applied — MMStar is uniformly MCQ, unlike AMBER's
            #  generative/discriminative split.)
            # batch_size=1 always → outputs is a single-element list.
            a = outputs[0]
            record = {**item, "response": a}
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
