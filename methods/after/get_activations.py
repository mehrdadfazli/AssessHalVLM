import os
from tqdm import tqdm
import numpy as np
from utils import *
import argparse

HF_NAMES = {
    'llava_v1.5_7B_lht': '/path/to/your/workdir/huggingface/llava-v1.5-7b-liuhaotian', 
    'instructblip_7B': '/path/to/your/workdir/huggingface/instructblip-vicuna-7b-old',
    'llava_next_7B': 'llava-hf/llava-v1.6-vicuna-7b-hf',
    "qwen2_5_vl_instruct": "/path/to/your/workdir/huggingface/qwen2.5-vl-7b-instruct",
    'shikra_7B': '/path/to/your/workdir/AFTER/models/shikra_model/shikra_config.py',
}

def main(): 
    """
    Specify dataset name as the first command line argument. Current options are 
    "tqa_mc2", "piqa", "rte", "boolq", "copa". Gets activations for all prompts in the 
    validation set for the specified dataset on the last token for llama-7B. 
    """

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='llama_7B')
    parser.add_argument('--dataset_name', type=str, default='tqa_mc2')
    parser.add_argument('--device', type=str, default="0")
    parser.add_argument("--model_dir", type=str, default=None, help='local directory with model data')
    parser.add_argument('--data_root', type=str, default=None, help='root folder for local dataset files')
    parser.add_argument('--features_root', type=str, default='features', help='folder to save extracted activations')
    parser.add_argument('--cache_dir', type=str, default=None, help='Hugging Face cache directory for downloads')
    parser.add_argument('--mode', type=str, default='answer')
    args = parser.parse_args()
    
    os.environ['CUDA_VISIBLE_DEVICES'] = args.device
    if args.cache_dir:
        os.environ['TRANSFORMERS_CACHE'] = args.cache_dir
        os.environ['HF_HOME'] = args.cache_dir

    MODEL = args.model_dir if args.model_dir else HF_NAMES[args.model_name]

    if 'POPE' in args.dataset_name:
        data_path = args.data_root if args.data_root else '/path/to/your/workdir/AFTER/data/POPE'
        datafile_path = os.path.join(data_path, 'train_3k.json')
        prompts, labels, filepaths = process_data_pope_activation(datafile_path, args.model_name, args.mode)
    elif 'AMBER' in args.dataset_name:
        data_path = args.data_root if args.data_root else '/path/to/your/workdir/AFTER/data/AMBER'
        datafile_path = os.path.join(data_path, 'train_3k.json')
        prompts, labels, filepaths = process_data_amber_activation(datafile_path, args.model_name, args.mode)
    else: 
        raise ValueError("Invalid dataset name")
    
    
    if 'llava' in args.model_name and 'lht' in args.model_name:
        from models.llava_inference_lht import Llava_lht
        model = Llava_lht(MODEL)

    elif 'shikra' in args.model_name:
        from models.shikra_inference import Shikra
        model = Shikra(MODEL)
        
    elif 'qwen2_5_vl' in args.model_name:
        from models.qwen2_5_vl_inference import Qwen2_5_VL
        model = Qwen2_5_VL(MODEL)
    
    elif 'instructblip' in args.model_name:
        from models.instructblip_inference import InstructBlip
        model = InstructBlip(MODEL)

    elif 'llava_next' in args.model_name:
        from models.llava_next_inference import Llava_next
        model = Llava_next(MODEL)

    print("Getting activations")
    os.makedirs(args.features_root, exist_ok=True)
    
    all_layer_wise_activations = []
    all_head_wise_activations = []
    for (prompt, filepath) in tqdm(zip(prompts, filepaths)):
        if not os.path.exists(filepath):
            layer_wise_activations, head_wise_activations, _ = model.get_activations_only_text(prompt)
        else:
            layer_wise_activations, head_wise_activations, _, _ = model.get_activations(prompt, filepath)
        all_layer_wise_activations.append(layer_wise_activations[:,-1,:].copy())
        all_head_wise_activations.append(head_wise_activations[:,-1,:].copy())
        
    print("Saving labels")
    np.save(os.path.join(args.features_root, f'{args.model_name}_{args.dataset_name}_labels.npy'), labels)
    
    print("Saving head wise activations")
    np.save(os.path.join(args.features_root, f'{args.model_name}_{args.dataset_name}_head_wise.npy'), all_head_wise_activations)
        
if __name__ == '__main__':
    main()
