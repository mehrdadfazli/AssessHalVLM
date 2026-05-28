import sys
sys.path.append('..')

from tqdm import tqdm
import json
import jsonlines
import os

from utils import RESPONSE_DICT

class Mllm:
    
    def __init__(self, model_name_or_path, *args, **kwargs) -> None:
        pass
    
    def evaluate(self, prompt, filepath):
        pass

    def _load_existing_indices(self, save_path: str):
        """Return set of already-completed sample indices from an existing jsonl."""
        if not save_path or not os.path.isfile(save_path):
            return set()
        done = set()
        try:
            with jsonlines.open(save_path, 'r') as reader:
                for row in reader:
                    if isinstance(row, dict) and 'index' in row:
                        done.add(row['index'])
        except Exception:
            # If file is corrupted/partial, we still prefer not to crash inference.
            return set()
        return done

    def _stream_write(self, save_path: str, rows):
        """Append rows to jsonl, flushing to disk after each row (crash-safe)."""
        parent = os.path.dirname(save_path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        with open(save_path, 'a', encoding='utf-8') as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + '\n')
                f.flush()
                os.fsync(f.fileno())
    
    def batch_evaluate(self, args, data):
        done = self._load_existing_indices(getattr(args, 'save_path', ''))
        for sample in tqdm(data):
            prompt = sample['prompt']
            image = sample['img_url']
            res = sample.copy()

            if 'index' in res and res['index'] in done:
                continue
            response = self.evaluate(prompt, image)
            res['response'] = response
                
            if args.verbose:
                print(res)
            self._stream_write(args.save_path, [res])
            if 'index' in res:
                done.add(res['index'])
    
    def batch_evaluate_of_caption(self, args, data):
        done = self._load_existing_indices(getattr(args, 'save_path', ''))
        for sample in tqdm(data):
            prompt = sample['prompt']
            image = sample['img_url']
            caption = sample['caption']
            res = sample.copy()

            if 'index' in res and res['index'] in done:
                continue
            response = self.evaluate_of_caption(prompt, image, caption)
            res['response'] = response
                
            if args.verbose:
                print(res)
            self._stream_write(args.save_path, [res])
            if 'index' in res:
                done.add(res['index'])
    
    def batch_evaluate_of_caption_img(self, args, data):
        done = self._load_existing_indices(getattr(args, 'save_path', ''))
        for sample in tqdm(data):
            prompt = sample['prompt']
            image = sample['img_url']
            caption = sample['caption']
            res = sample.copy()

            if 'index' in res and res['index'] in done:
                continue
            response = self.evaluate_of_caption_img(prompt, image, caption)
            res['response'] = response
                
            if args.verbose:
                print(res)
            self._stream_write(args.save_path, [res])
            if 'index' in res:
                done.add(res['index'])
    
    def batch_evaluate_with_intervention(self, args, data, interventions={}, intervention_fn=None, multiple=False):
        done = self._load_existing_indices(getattr(args, 'save_path', ''))
        for sample in tqdm(data):
            prompt = sample['prompt']
            image = sample['img_url']
            res = sample.copy()

            if 'index' in res and res['index'] in done:
                continue
            
            if multiple == False:
                response = self.evaluate_with_intervention(prompt, image, interventions, intervention_fn)
            else:
                response = self.evaluate_with_multiple_intervention(prompt, image, interventions, intervention_fn)
            res['response'] = response
                        
            if args.verbose:
                print(res)
            self._stream_write(args.save_path, [res])
            if 'index' in res:
                done.add(res['index'])
            
    def batch_evaluate_with_intervention_youare_offset(self, args, data, interventions={}, intervention_fn=None, multiple=False):
        done = self._load_existing_indices(getattr(args, 'save_path', ''))
        for sample in tqdm(data):
            prompt = sample['prompt']
            image = sample['img_url']
            res = sample.copy()

            if 'index' in res and res['index'] in done:
                continue
            
            if multiple == False:
                response = self.evaluate_with_intervention_youare_offset(prompt, image, interventions, intervention_fn)
            else:
                response = self.evaluate_with_multiple_intervention(prompt, image, interventions, intervention_fn)
            res['response'] = response
                        
            if args.verbose:
                print(res)
            self._stream_write(args.save_path, [res])
            if 'index' in res:
                done.add(res['index'])