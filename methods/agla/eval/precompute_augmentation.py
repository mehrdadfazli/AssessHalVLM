"""
Pre-compute AGLA augmented images and save to disk.
Uses BLIP ITM model only — no target LLM needed.
"""
import argparse
import json
import os
import sys
import torch
from tqdm import tqdm
from PIL import Image
from torchvision import transforms

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from lavis.models import load_model_and_preprocess
from augmentation import augmentation

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--question_file", type=str, required=True)
    parser.add_argument("--image_folder", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--image_key", type=str, default="image", help="Key for image filename in JSON")
    parser.add_argument("--question_key", type=str, default="text", help="Key for question text in JSON")
    args = parser.parse_args()

    device = 'cuda'
    os.makedirs(args.output_dir, exist_ok=True)

    # Load BLIP ITM model
    print("Loading BLIP ITM model...")
    model_itm, image_processors, text_processors = load_model_and_preprocess(
        "blip_image_text_matching", "large", device=device, is_eval=True
    )
    loader = transforms.Compose([transforms.ToTensor()])

    # Load questions
    if args.question_file.endswith('.jsonl'):
        with open(args.question_file) as f:
            data = [json.loads(l) for l in f]
    else:
        with open(args.question_file) as f:
            data = json.load(f)

    print(f"Processing {len(data)} images...")

    for item in tqdm(data, desc="Augmenting images"):
        image_filename = item[args.image_key]
        # Handle MMHal-style URLs
        if '/' in image_filename:
            image_filename = image_filename.split('/')[-1]

        image_path = os.path.join(args.image_folder, image_filename)
        output_path = os.path.join(args.output_dir, image_filename)

        if os.path.exists(output_path):
            continue

        if not os.path.exists(image_path):
            print(f"Warning: {image_path} not found, skipping")
            continue

        try:
            raw_image = Image.open(image_path).convert("RGB")
            question = item[args.question_key]

            tensor_image = loader(raw_image.resize((384, 384)))
            image = image_processors["eval"](raw_image).unsqueeze(0).to(device)
            q_text = text_processors["eval"](question)
            tokenized_text = model_itm.tokenizer(
                q_text, padding='longest', truncation=True, return_tensors="pt"
            ).to(device)

            augmented_image = augmentation(
                image, q_text, tensor_image, model_itm, tokenized_text, raw_image
            )
            augmented_image.save(output_path)
            torch.cuda.empty_cache()

        except Exception as e:
            # Save original image as fallback so downstream script doesn't skip it
            try:
                raw_image = Image.open(image_path).convert("RGB")
                raw_image.save(output_path)
                print(f"Fallback (saved original): {image_filename}")
            except:
                print(f"Error processing {image_filename}: {e}")
            continue

    print(f"Augmented images saved to {args.output_dir}")

if __name__ == "__main__":
    main()
