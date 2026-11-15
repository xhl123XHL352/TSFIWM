"""Frozen Qwen3.5 visual reader. Never places evaluation transcripts in prompts."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import re
import time
from gpu_accounting import ROOT, GpuLease

BASE = Path("models/qwen3.5-9b")
PROMPT = ('Read the three instrument panels, ordered left to right along the table. '
          'Return only JSON with keys "ids" (three strings) and "values" (three integers). '
          'Read the printed identifier above each large numeric display. '
          'Use null for any identifier or number you cannot read. Do not guess hidden characters.')


def extract(text):
    candidates = re.findall(r'\{[^{}]*\}', text, re.S)
    for candidate in reversed(candidates):
        try:
            data = json.loads(candidate)
            if set(data) >= {'ids', 'values'} and len(data['ids']) == len(data['values']) == 3:
                ids = [str(x) if x is not None else None for x in data['ids']]
                values = []
                for x in data['values']:
                    if isinstance(x, (float, int)) and 0 <= x <= 999:
                        values.append(float(x))
                    elif isinstance(x, str) and x.isdigit() and 0 <= int(x) <= 999:
                        values.append(float(x))
                    else:
                        values.append(None)
                return {'ids': ids, 'values': values, 'parse_valid': True}
        except (ValueError, TypeError):
            continue
    return {'ids': [None]*3, 'values': [None]*3, 'parse_valid': False}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['audit', 'context'], required=True)
    parser.add_argument('--splits', nargs='+', default=['train', 'dev'])
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--gpu-uuid', required=True)
    args = parser.parse_args()
    with GpuLease(args.gpu_uuid, 'qwen-reader-'+args.mode):
        import torch
        from transformers import Qwen3_5ForConditionalGeneration, AutoProcessor
        from transformers.utils import logging
        logging.disable_progress_bar()
        processor = AutoProcessor.from_pretrained(BASE, local_files_only=True)
        model = Qwen3_5ForConditionalGeneration.from_pretrained(BASE, local_files_only=True,
              dtype=torch.bfloat16, device_map={'': 0}, attn_implementation='sdpa').eval()
        processor.tokenizer.padding_side = 'left'
        if args.mode == 'audit':
            images = [ROOT / 'data' / 'panel-audit' / 'after.png']
            destination = ROOT / 'results' / 'engineering'
        else:
            if any(s not in ('train', 'dev') for s in args.splits):
                raise RuntimeError('Independent test requires a frozen evaluation entry point')
            images = []
            spec = json.loads((ROOT/'source'/'panel-config.json').read_text())
            for split in args.splits:
                folders = sorted((ROOT/'data'/'panel-v3'/split).iterdir())
                if len(folders) != spec['splits'][split]['roots'] or any(not (x/'complete.json').exists() for x in folders):
                    raise RuntimeError('Split data generation is not complete')
                for folder in folders:
                    inp = json.loads((folder/'inputs.json').read_text())
                    images.extend(folder / c['image'] for c in inp['context'])
            destination = ROOT / 'results' / 'perception'
        destination.mkdir(parents=True, exist_ok=True)
        output = destination / ('qwen-reader-'+args.mode+'-'+ '-'.join(args.splits)+'.jsonl')
        done = set()
        if output.exists():
            done = {json.loads(line)['image'] for line in output.read_text().splitlines()}
        images = [im for im in images if str(im.relative_to(ROOT)) not in done]
        config = {'model': 'Qwen3.5-9B', 'revision': BASE.name, 'prompt': PROMPT,
                  'batch_size': args.batch_size, 'max_new_tokens': 128, 'do_sample': False,
                  'enable_thinking': False, 'dtype': 'bfloat16', 'attention': 'sdpa',
                  'setting': 'read pixels; no answer or annotation in prompt'}
        (destination/'qwen-reader-config.json').write_text(json.dumps(config, indent=2))
        started = time.monotonic()
        with output.open('a', encoding='utf-8') as f, torch.inference_mode():
            for start in range(0, len(images), args.batch_size):
                batch = images[start:start+args.batch_size]
                conversations = [[{'role': 'user', 'content': [
                    {'type': 'image', 'image': str(im)}, {'type': 'text', 'text': PROMPT}]}] for im in batch]
                inputs = processor.apply_chat_template(conversations, tokenize=True,
                    add_generation_prompt=True, return_dict=True, return_tensors='pt',
                    processor_kwargs={'padding': True}, enable_thinking=False).to('cuda')
                generated = model.generate(**inputs, max_new_tokens=128, do_sample=False)
                outputs = processor.batch_decode(generated[:, inputs['input_ids'].shape[-1]:], skip_special_tokens=True)
                for im, raw in zip(batch, outputs):
                    record = {'image': str(im.relative_to(ROOT)), 'raw': raw, **extract(raw)}
                    f.write(json.dumps(record, ensure_ascii=False)+'\n')
                    if args.mode == 'audit':
                        print(json.dumps(record, ensure_ascii=False), flush=True)
                f.flush()
                if start % (args.batch_size*10) == 0:
                    print(json.dumps({'completed': min(start+len(batch), len(images)),
                         'total': len(images), 'elapsed_s': time.monotonic()-started,
                         'peak_allocated_gib': torch.cuda.max_memory_allocated()/1024**3}), flush=True)
        print(json.dumps({'output': str(output), 'images_this_run': len(images),
                          'elapsed_s': time.monotonic()-started}), flush=True)


if __name__ == '__main__':
    main()
