"""Evaluate the distinct calibrated-view observation condition."""
from __future__ import annotations
import argparse
import json
from gpu_accounting import ROOT, GpuLease
from qwen_read import BASE, extract

PROMPT = ('These three images show panel 0, panel 1, and panel 2 in that order. '
          'Read the identifier printed above the large numeric value in each image. '
          'Return only JSON with keys "ids" (three strings in the image order) and '
          '"values" (three integer values in the same order). '
          'Use null for anything you cannot read. Do not guess hidden characters.')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--split', default='dev')
    parser.add_argument('--gpu-uuid')
    parser.add_argument('--score', action='store_true')
    parser.add_argument('--view-source',choices=['camera','crop'],default='camera')
    parser.add_argument('--memory-cap-gib',type=float,default=32)
    args = parser.parse_args()
    stem='qwen-initial-views' if args.view_source=='camera' else 'qwen-initial-crops'
    data='panel-initial-views-v1' if args.view_source=='camera' else 'panel-initial-crops-v1'
    output = ROOT/'results'/'perception'/f'{stem}-{args.split}.jsonl'
    folders = sorted((ROOT/'data'/data/args.split).iterdir())
    if args.score:
        rows = [json.loads(line) for line in output.read_text().splitlines()]
        if len(rows) != len(folders):
            raise RuntimeError('Reader output is incomplete')
        correct_ids, correct_values, all_ids = [], [], []
        for row in rows:
            source='panel-control-v1' if args.split.startswith('control_') else 'panel-v3'
            lab = json.loads((ROOT/'data'/source/args.split/row['root_id']/'labels.json').read_text())['context_states'][0]
            ids = [a == b for a,b in zip(row['ids'],lab['ids'])]
            correct_ids.extend(ids)
            all_ids.append(all(ids))
            correct_values.extend([a == b for a,b in zip(row['values'],lab['values'])])
        result = {'roots': len(rows), 'identifier_field_recovery': sum(correct_ids)/len(correct_ids),
                  'all_three_identifiers_recovery': sum(all_ids)/len(all_ids),
                  'numeric_field_recovery': sum(correct_values)/len(correct_values),
                  'sensor_condition': ('three calibrated initial 384x384 camera views' if args.view_source=='camera' else 'three 384x384 rectified crops of the original initial wide image')+'; identical three-image prompt; no future information'}
        (output.with_suffix('.metrics.json')).write_text(json.dumps(result, indent=2))
        print(json.dumps(result), flush=True)
        return
    with GpuLease(args.gpu_uuid,stem+'-'+args.split,max_memory_gib=args.memory_cap_gib):
        import torch
        from transformers import Qwen3_5ForConditionalGeneration, AutoProcessor
        from transformers.utils import logging
        logging.disable_progress_bar()
        processor = AutoProcessor.from_pretrained(BASE, local_files_only=True)
        processor.tokenizer.padding_side = 'left'
        model = Qwen3_5ForConditionalGeneration.from_pretrained(BASE, local_files_only=True,
            dtype=torch.bfloat16, device_map={'': 0}, attn_implementation='sdpa').eval()
        output.parent.mkdir(parents=True, exist_ok=True)
        done = {json.loads(line)['root_id'] for line in output.read_text().splitlines()} if output.exists() else set()
        folders = [folder for folder in folders if folder.name not in done]
        with output.open('a', encoding='utf-8') as f, torch.inference_mode():
            for start in range(0, len(folders), 2):
                batch = folders[start:start+2]
                conversations = []
                for folder in batch:
                    meta = json.loads((folder/'complete.json').read_text())
                    content = [{'type': 'image', 'image': str(folder/name)} for name in meta['images']]
                    content.append({'type': 'text', 'text': PROMPT})
                    conversations.append([{'role': 'user', 'content': content}])
                inputs = processor.apply_chat_template(conversations, tokenize=True,
                    add_generation_prompt=True, return_dict=True, return_tensors='pt',
                    processor_kwargs={'padding': True}, enable_thinking=False).to('cuda')
                gen = model.generate(**inputs, max_new_tokens=128, do_sample=False,
                                     pad_token_id=processor.tokenizer.eos_token_id)
                outputs = processor.batch_decode(gen[:,inputs['input_ids'].shape[-1]:], skip_special_tokens=True)
                for folder, raw in zip(batch, outputs):
                    f.write(json.dumps({'root_id': folder.name, 'raw': raw, **extract(raw)})+'\n')
                f.flush()
                if start % 20 == 0:
                    print(json.dumps({'initial_views_read': start+len(batch), 'total': len(folders),
                          'peak_allocated_gib': torch.cuda.max_memory_allocated()/1024**3}), flush=True)


if __name__ == '__main__':
    main()
