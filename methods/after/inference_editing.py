"""
This script is used to evaluate MLLMs.
"""
import argparse
import logging
import random
import time
import os
from einops import rearrange
import numpy as np

import torch
import json
from utils import *
from train_estimator import OffsetGenerator


PATH = {
    'llava_v1.5_7B_lht': '/path/to/your/workdir/huggingface/llava-v1.5-7b-liuhaotian', 
    'instructblip_7B': '/path/to/your/workdir/huggingface/instructblip-vicuna-7b-old',
    'llava_next_7B': 'llava-hf/llava-v1.6-vicuna-7b-hf',
    "qwen2_5_vl_instruct": "/path/to/your/workdir/huggingface/qwen2.5-vl-7b-instruct",
    'shikra_7B': '/path/to/your/workdir/AFTER/models/shikra_model/shikra_config.py',
}

def seed_all(seed = 8888):
    torch.manual_seed(seed)
    random.seed(seed)
    
def evaluate(func):
    def wrapper(*args, **kwargs):
        result = func(*args, **kwargs)
        return result
    return wrapper

@evaluate
def evaluate_model(model, args, data):
    model.batch_evaluate(args, data)  

@evaluate
def evaluate_model_with_intervention(model, args, data, interventions={}, intervention_fn=None):
    model.batch_evaluate_with_intervention_youare_offset(args, data, interventions, intervention_fn)  

def get_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--model', type=str, default='llava_v1.5_7B', help="specifies the model to be evaluated.")
    
    parser.add_argument('--probe_dataset', type=str, default='spa_vl', help='feature bank for training probes')
    parser.add_argument('--validate_dataset', type=str, default='MLLMGuard', help="specifies the path to the data")
    parser.add_argument('--save_path', type=str, default='', help='specifies the path to save the results.')

    parser.add_argument('--neg_mode', type=str, default=None, help='specifies the project name in wandb.')
    parser.add_argument('--pos_mode', type=str, default=None, help='specifies the project name in wandb.')
    
    parser.add_argument('--num_heads', type=int, default=48, help='K, number of top heads to intervene on')
    parser.add_argument('--alpha', type=int, default=15, help='alpha, intervention strength')
    parser.add_argument('--use_random_dir', action='store_true', help='use random direction', default=False)
    parser.add_argument('--seed', type=int, default=42, help='seed')
    parser.add_argument('--verbose', type=bool, default=True, help='specifies whether to display verbose outputs.')
    
    parser.add_argument('--device', type=str, default='0')
    parser.add_argument('--categories', type=str, default='all')
    parser.add_argument('--subfix', type=str, default='all')
    parser.add_argument('--model_dir', type=str, default=None, help='local model directory for the selected LVLM')
    parser.add_argument('--data_root', type=str, default=None, help='root folder for local dataset files')
    parser.add_argument('--features_root', type=str, default='features', help='folder containing activation features')
    parser.add_argument('--probes_root', type=str, default='probes', help='folder containing trained offset checkpoints')
    parser.add_argument('--results_root', type=str, default='results', help='folder to save model outputs')
    parser.add_argument('--cache_dir', type=str, default=None, help='Hugging Face cache directory for downloads')
    
    parser.add_argument('--start_layer', type=int, default=0, help='K, number of top heads to intervene on')
    parser.add_argument('--end_layer', type=int, default=31, help='K, number of top heads to intervene on')
    
    parser.add_argument('--data_ratio', type=float, default=1.0, help='K, number of top heads to intervene on')
    parser.add_argument('--offset_name', type=str, default='offset_generator', help='specifies the path to save the results.')
    parser.add_argument('--no_intervention', action='store_true', help='Vanilla LVLM inference (no AFTER editing); skips probes/features.')
    parser.add_argument('--max_new_tokens', type=int, default=128,
                        help='Maximum new tokens for generation (use 512 for free-text captioning benchmarks like CHAIR)')
    parser.add_argument('--num_images', type=int, default=500,
                        help='Number of images to evaluate for CHAIR (default: 500)')
    parser.add_argument('--mmstar_hf_id', type=str, default='Lin-Chen/MMStar',
                        help='Hugging Face dataset id for MMStar')
    parser.add_argument('--mmstar_config', type=str, default='val',
                        help='HF config name for MMStar (default: val)')
    parser.add_argument('--mmstar_split', type=str, default='val',
                        help='HF split for MMStar (default: val)')
    parser.add_argument('--mmstar_max_samples', type=int, default=None,
                        help='If set, only evaluate the first N MMStar samples (debug/smoke)')
    parser.add_argument('--mmhal_data_root', type=str, default=None,
                        help='Root of MMHal-Bench (contains response_template.json and images/)')
    
    args = parser.parse_args()

    return args

def main(args):
    
    import random
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    
    if args.cache_dir:
        os.environ['TRANSFORMERS_CACHE'] = args.cache_dir
        os.environ['HF_HOME'] = args.cache_dir

    model_name = args.model.lower()
    model_path = args.model_dir if args.model_dir else PATH.get(args.model)
    if model_path is None:
        raise ValueError(f"No local model path found for {args.model}. Set --model_dir or update PATH.")

    if 'llava' in model_name and 'lht' in model_name:
        from models.llava_inference_lht import Llava_lht
        model = Llava_lht(model_path)
    elif 'shikra' in model_name:
        from models.shikra_inference import Shikra
        model = Shikra(model_path)
    elif 'qwen2_5_vl_instruct' in model_name:
        from models.qwen2_5_vl_inference import Qwen2_5_VL
        model = Qwen2_5_VL(model_path)
    elif 'instructblip' in model_name:
        from models.instructblip_inference import InstructBlip
        model = InstructBlip(model_path)
    elif 'llava_next' in model_name:
        from models.llava_next_inference import Llava_next
        model = Llava_next(model_path)
    else:
        raise NotImplementedError(
            f'Model {model_name} has not been implemented.'
        )
    
    model.max_new_tokens = args.max_new_tokens

    if 'qwen2_5_vl' in model_name:
        num_layers = 28
        num_heads = 28
    else:
        num_layers = 32
        num_heads = 32
    
    interventions = {}
    if not args.no_intervention:
        # load activations (positive/negative samples together)
        pos_path = os.path.join(args.features_root, f'{args.model}_{args.probe_dataset}_{args.pos_mode}_head_wise.npy')
        neg_path = os.path.join(args.features_root, f'{args.model}_{args.probe_dataset}_{args.neg_mode}_head_wise.npy')

        head_wise_activations, labels = load_and_conbine_activations(
            pos_path=pos_path,
            neg_path=neg_path
            )
        head_wise_activations = rearrange(head_wise_activations, 'b l (h d) -> b l h d', h = num_heads)
        
        args.probe_dataset = f'{args.probe_dataset}_{args.neg_mode};{args.pos_mode}'
        if 'POPE' in args.probe_dataset:
            split_range = 12
        elif 'AMBER' in args.probe_dataset:
            split_range = 2
        # elif args.probe_dataset == 'POPE_test':
        head_wise_activations = head_wise_activations[:int(args.data_ratio * labels.shape[0])]
        labels = labels[:int(args.data_ratio * labels.shape[0])]
        separated_head_wise_activations, separated_labels, idxs_to_split_at = get_separated_activations(labels, head_wise_activations, split_range)
        
        # get directions
        com_directions = get_com_directions(num_layers, num_heads, separated_head_wise_activations, separated_labels)
        top_heads = sort_direction_len(com_directions, num_layers, num_heads, args.num_heads, args.start_layer, args.end_layer)
        print("Heads intervened: ", sorted(top_heads))
        
        checkpoint_path = os.path.join(args.probes_root, f'{args.offset_name}.pth')
        checkpoint = torch.load(checkpoint_path, map_location='cuda')
        offsetgenerators = OffsetGenerator(**checkpoint['model_config']).to('cuda')
        offsetgenerators.load_state_dict(checkpoint['model_state_dict'])
        
        interventions = get_interventions_dict_withoffset(model_name, top_heads, num_heads, com_directions, offsetgenerators)

    def lt_modulated_vector_add(head_output, layer_name, start_edit_location='lt', interventions=None): 
        reshape_signal = False
        if len(head_output.shape) == 3:
            reshape_signal = True
            head_output = rearrange(head_output, 'b s (h d) -> b s h d', h=num_heads)
        for head, direction, proj_val_std in interventions[layer_name]:
            direction_to_add = torch.tensor(direction).to(head_output.device.index)
            if start_edit_location == 'lt': 
                head_output[:, -1, head, :] += args.alpha * proj_val_std * direction_to_add
            else: 
                head_output[:, start_edit_location:, head, :] += args.alpha * proj_val_std * direction_to_add
        if reshape_signal:
            head_output = rearrange(head_output, 'b s h d -> b s (h d)')
        return head_output
    
    
    if 'POPE' in args.validate_dataset:
        ## POPE_coco\ POPE_aokvqa POPE_gqa
        dataset = args.validate_dataset.split('_')[-1] 
        dataset_path = args.data_root if args.data_root else "/path/to/your/workdir/AFTER/data/POPE"
        if args.categories == 'all':
            args.categories = ['adversarial', 'popular', 'random']
        else:
            args.categories = args.categories.split(' ')
        
        for c in args.categories:
            save_dir = os.path.join(args.results_root, 'POPE', dataset)
            os.makedirs(save_dir, exist_ok=True)
            save_name = os.path.join(save_dir, f'{args.model}_{c}_{args.num_heads}_{args.alpha}{args.subfix}.jsonl')
            print(save_name)
            args.save_path = save_name
            validate_data = process_data_pope(dataset_path, dataset, c)

            if args.no_intervention:
                evaluate_model(model, args, validate_data)
            else:
                evaluate_model_with_intervention(model, args, validate_data,
                    interventions=interventions,
                    intervention_fn=lt_modulated_vector_add)
        
    elif args.validate_dataset == 'MME':
        dataset_path = args.data_root if args.data_root else "/path/to/your/workdir/AFTER/data/MME"
        if args.categories == 'all':
            args.categories = ['color', 'count', 'existence', 'position']
        else:
            args.categories = args.categories.split(' ')
        
        for c in args.categories:
            save_dir = os.path.join(args.results_root, args.validate_dataset)
            os.makedirs(save_dir, exist_ok=True)
            save_name = os.path.join(save_dir, f'{args.model}_{c}_{args.num_heads}_{args.alpha}{args.subfix}.jsonl')
            args.save_path = save_name
            validate_data = process_data_mme(dataset_path, c)
            if args.no_intervention:
                evaluate_model(model, args, validate_data)
            else:
                evaluate_model_with_intervention(model, args, validate_data,
                    interventions=interventions,
                    intervention_fn=lt_modulated_vector_add)
    
    elif args.validate_dataset == 'MME_general':
        dataset_path = args.data_root if args.data_root else "/path/to/your/workdir/AFTER/data/MME"
        if args.categories == 'all':
            args.categories = ['artwork', 'celebrity', 'code_reasoning', 'commonsense_reasoning', 
            'landmark', 'numerical_calculation', 'OCR', 'posters', 'scene', 'text_translation']
        else:
            args.categories = args.categories.split(' ')
        
        for c in args.categories:
            save_dir = os.path.join(args.results_root, args.validate_dataset)
            os.makedirs(save_dir, exist_ok=True)
            save_name = os.path.join(save_dir, f'{args.model}_{c}_{args.num_heads}_{args.alpha}{args.subfix}.jsonl')
            args.save_path = save_name
            validate_data = process_data_mme(dataset_path, c)
            if args.no_intervention:
                evaluate_model(model, args, validate_data)
            else:
                evaluate_model_with_intervention(model, args, validate_data,
                    interventions=interventions,
                    intervention_fn=lt_modulated_vector_add)
    
    elif args.validate_dataset == 'AMBER':
        dataset_path = args.data_root if args.data_root else "/path/to/your/workdir/AFTER/data/AMBER"
        if args.categories == 'all':
            args.categories = ['gen', 'dis-attribute_sub', 'dis-existence_sub', 'dis-relation_sub']
        else:
            args.categories = args.categories.split(' ')
        
        for c in args.categories:
            save_dir = os.path.join(args.results_root, args.validate_dataset)
            os.makedirs(save_dir, exist_ok=True)
            save_name = os.path.join(save_dir, f'{args.model}_{c}_{args.num_heads}_{args.alpha}{args.subfix}.jsonl')
            args.save_path = save_name
            validate_data = process_data_amber(dataset_path, c)
            if args.no_intervention:
                evaluate_model(model, args, validate_data)
            else:
                evaluate_model_with_intervention(model, args, validate_data,
                    interventions=interventions,
                    intervention_fn=lt_modulated_vector_add)

    elif args.validate_dataset == 'CHAIR':
        if not args.data_root:
            raise ValueError(
                "CHAIR requires --data_root pointing to COCO val2014 images "
                "(directory containing COCO_val2014_*.jpg)."
            )
        coco_path = args.data_root
        validate_data = process_data_chair(
            coco_path,
            opera_results_path=None,
            seed=args.seed,
            num_images=args.num_images,
        )

        save_dir = os.path.join(args.results_root, 'CHAIR')
        os.makedirs(save_dir, exist_ok=True)
        save_name = os.path.join(save_dir,
                        f'{args.model}_{args.num_heads}_{args.alpha}{args.subfix}.jsonl')
        args.save_path = save_name

        if args.no_intervention:
            evaluate_model(model, args, validate_data)
        else:
            evaluate_model_with_intervention(model, args, validate_data,
                interventions=interventions,
                intervention_fn=lt_modulated_vector_add)

    elif args.validate_dataset == 'MMStar':
        dataset_path = args.data_root if args.data_root else os.path.join(os.getcwd(), 'data')
        validate_data = process_data_mmstar(
            dataset_path,
            hf_dataset_id=args.mmstar_hf_id,
            hf_config=args.mmstar_config,
            split=args.mmstar_split,
            cache_dir=args.cache_dir,
            max_samples=args.mmstar_max_samples,
        )

        save_dir = os.path.join(args.results_root, 'MMStar')
        os.makedirs(save_dir, exist_ok=True)
        save_name = os.path.join(
            save_dir,
            f'{args.model}_{args.num_heads}_{args.alpha}{args.subfix}.jsonl',
        )
        args.save_path = save_name

        if args.no_intervention:
            evaluate_model(model, args, validate_data)
        else:
            evaluate_model_with_intervention(
                model,
                args,
                validate_data,
                interventions=interventions,
                intervention_fn=lt_modulated_vector_add,
            )

    elif args.validate_dataset == 'MMHal':
        if not args.mmhal_data_root:
            raise ValueError(
                "MMHal requires --mmhal_data_root pointing to MMHal-Bench "
                "(contains response_template.json and images/)."
            )
        validate_data = process_data_mmhal(args.mmhal_data_root)

        save_dir = os.path.join(args.results_root, 'MMHal-Bench')
        os.makedirs(save_dir, exist_ok=True)
        args.save_path = os.path.join(
            save_dir,
            f'{args.model}_{args.num_heads}_{args.alpha}{args.subfix}.jsonl',
        )

        if args.no_intervention:
            evaluate_model(model, args, validate_data)
        else:
            evaluate_model_with_intervention(
                model,
                args,
                validate_data,
                interventions=interventions,
                intervention_fn=lt_modulated_vector_add,
            )

if __name__ == "__main__":
    args = get_args()
    
    os.environ['CUDA_VISIBLE_DEVICES'] = args.device
    main(args)
    # seed_all(5555)