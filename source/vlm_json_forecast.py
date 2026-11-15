"""Frozen multimodal-model state forecasting, with explicit input conditions."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import re
import time
import hashlib
from datetime import datetime,timezone
from gpu_accounting import GpuLease, ROOT

RULES = '''There are three digital instrument panels, indexed 0,1,2 by their left-to-right physical positions.
Identifiers stay fixed. A press primitive addresses one panel; target -1 means wait. dt is a digital clock interval.
Each panel has an unknown fixed integer gain in {1,2,3}. Infer gains from the calibration observations.
In a counter, a successful press adds that panel's gain; waiting does not change its value.
In a toggle, successful presses alternate adding and subtracting that panel's gain; all toggle modes start off.
In a flow accumulator, a successful press toggles that panel on/off; then every on panel increases by its gain times dt, including during waits. All start off.
A press may physically fail. The future action list gives requested commands, not future contact outcomes.
Use the visible/calibration history to forecast. Do not assume a number is constant just because its panel is not visible.
Return only JSON with keys "ids" (three identifiers) and "values" (four successive rows of three predicted integer values).'''


def parse(text):
    for raw in reversed(re.findall(r'\{[^{}]*\}', text, re.S)):
        try:
            data = json.loads(raw)
            values = data['values']
            if len(values) == 4 and all(len(v) == 3 for v in values):
                values = [[float(x) for x in row] for row in values]
                if all(0 <= x <= 999 for row in values for x in row):
                    return {'values': values, 'ids': data.get('ids'), 'parse_valid': True}
        except (KeyError, TypeError, ValueError):
            continue
    return {'values': [[-1000.0]*3 for _ in range(4)], 'ids': None, 'parse_valid': False}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', choices=['qwen', 'internvl'], required=True)
    parser.add_argument('--condition', choices=['annotation', 'pixels'], required=True)
    parser.add_argument('--split', choices=['dev', 'test', 'test_shift'], default='dev')
    parser.add_argument('--gpu-uuid', required=True)
    parser.add_argument('--memory-cap-gib',type=float,default=32)
    parser.add_argument('--batch-size',type=int,default=1)
    parser.add_argument('--run-tag',default='')
    parser.add_argument('--max-records',type=int)
    args = parser.parse_args()
    assert args.split=='dev' and args.run_tag=='json768', 'Development-only separate formatting protocol'
    if args.split != 'dev' and not (ROOT/'results'/'evaluation-freeze.json').exists():
        raise RuntimeError('Independent evaluation has not been frozen')
    folders = sorted((ROOT/'data'/'panel-v3'/args.split).iterdir())
    spec = json.loads((ROOT/'source'/'panel-config.json').read_text())['splits'][args.split]
    if len(folders) != spec['roots'] or any(not (f/'complete.json').exists() for f in folders):
        raise RuntimeError('Split data generation is not complete')
    work = ROOT/'results'/'vlm'
    work.mkdir(parents=True, exist_ok=True)
    assert not args.run_tag or args.run_tag.replace('-','').isalnum()
    assert 1<=args.batch_size<=16
    stem=f'{args.model}-{args.condition}'+('-'+args.run_tag if args.run_tag else '')+'-'+args.split
    output = work/(stem+'.jsonl')
    done = {(r['root_id'], r['branch']) for line in output.read_text().splitlines()
            if (r := json.loads(line))} if output.exists() else set()
    if len(done)==len(folders)*4:
        print(json.dumps({'preserved_complete':str(output)}));return
    with GpuLease(args.gpu_uuid,'forecast-'+stem,max_memory_gib=args.memory_cap_gib):
        import torch
        from transformers import AutoProcessor, AutoModelForImageTextToText
        from transformers.utils import logging
        logging.disable_progress_bar()
        logging.set_verbosity_error()
        base = (Path('models/qwen3.5-9b')
                if args.model == 'qwen' else ROOT/'models'/'internvl3.5-8b')
        processor = AutoProcessor.from_pretrained(base, local_files_only=True, trust_remote_code=False)
        processor.tokenizer.padding_side='left'
        model = AutoModelForImageTextToText.from_pretrained(base, local_files_only=True,
                   dtype=torch.bfloat16, device_map={'': 0}, attn_implementation='sdpa', trust_remote_code=False).eval()
        config = {**{k:v for k,v in vars(args).items() if k!='max_records'}, 'system_format_instruction': 'You are an exact JSON forecasting API. Return one JSON object with keys ids and values. Your first character must be {. Give no explanation, reasoning, markdown or commentary. ids is a list of three strings. values is a list of four lists, each containing three integer readings.', 'rules_in_prompt': RULES, 'max_new_tokens': 768,
                  'do_sample': False, 'enable_thinking': False, 'dtype': 'bfloat16',
                  'attention': 'sdpa', 'training': 'frozen; privileged qualitative device rules provided',
                  'context': 'four observations and three calibration commands; future commands; no future labels'}
        config_path=work/(stem+'-config.json')
        if config_path.exists() and args.run_tag:
            assert json.loads(config_path.read_text())==config,'Resume configuration differs'
        config_path.write_text(json.dumps(config, indent=2))
        with (work/(stem+'-receipts.jsonl')).open('a') as log:
            log.write(json.dumps({'utc':datetime.now(timezone.utc).isoformat(),'argv':vars(args),
                'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'existing_records':len(done)})+'\n')
        started = time.monotonic()
        count = 0
        pending=[]
        with output.open('a', encoding='utf-8') as f, torch.inference_mode():
            def emit_batch(batch):
                nonlocal count
                tick=time.monotonic()
                conversations=[[{'role':'system','content':[{'type':'text','text':'You are an exact JSON forecasting API. Return one JSON object with keys ids and values. Your first character must be {. Give no explanation, reasoning, markdown or commentary. ids is a list of three strings. values is a list of four lists, each containing three integer readings.'}]},{'role':'user','content':item[3]}] for item in batch]
                inputs=processor.apply_chat_template(conversations,tokenize=True,add_generation_prompt=True,
                    return_dict=True,return_tensors='pt',processor_kwargs={'padding':True},enable_thinking=False).to('cuda')
                generated=model.generate(**inputs,max_new_tokens=768,do_sample=False,
                    pad_token_id=processor.tokenizer.eos_token_id)
                decoded=processor.batch_decode(generated[:,inputs['input_ids'].shape[-1]:],skip_special_tokens=True)
                pixel_shape=list(inputs['pixel_values'].shape) if 'pixel_values' in inputs else None
                for index,(item,raw) in enumerate(zip(batch,decoded)):
                    root_id,branch,family,_=item
                    record={'root_id':root_id,'branch':branch,'family':family,'raw':raw,**parse(raw),
                        'input_tokens':int(inputs['attention_mask'][index].sum()),
                        'pixel_values_batch_shape':pixel_shape}
                    f.write(json.dumps(record,ensure_ascii=False)+'\n')
                f.flush();count+=len(batch)
                print(json.dumps({'model':args.model,'run_tag':args.run_tag,'completed_this_run':count,
                    'batch':len(batch),'batch_elapsed_s':time.monotonic()-tick,'elapsed_s':time.monotonic()-started,
                    'peak_allocated_gib':torch.cuda.max_memory_allocated()/1024**3}),flush=True)
            for folder in folders:
                inp = json.loads((folder/'inputs.json').read_text())
                if args.condition == 'annotation':
                    # Strictly select context annotations; never serialize labels' future fields.
                    lab = json.loads((folder/'labels.json').read_text())
                    history = [{'ids': s['ids'], 'values': s['values']} for s in lab['context_states']]
                else:
                    history = None
                for branch in inp['branches']:
                    if (inp['root_id'], branch['branch']) in done:
                        continue
                    description = {'family': inp['family'],
                        'calibration_commands': [c['preceding_action'] for c in inp['context'][1:]],
                        'future_commands': branch['actions']}
                    content = []
                    if history is not None:
                        description['observed_states'] = history
                    else:
                        for t, obs in enumerate(inp['context']):
                            content.extend([{'type': 'text', 'text': f'Observed calibration image {t}:'},
                                            {'type': 'image', 'image': str(folder/obs['image'])}])
                    content.append({'type': 'text', 'text': RULES+'\n'+json.dumps(description)})
                    if args.max_records is not None and count+len(pending)>=args.max_records:
                        break
                    pending.append((inp['root_id'],branch['branch'],inp['family'],content))
                    if len(pending)==args.batch_size:
                        emit_batch(pending);pending=[]
                if args.max_records is not None and count>=args.max_records:
                    break
            if pending:
                emit_batch(pending)
        print(json.dumps({'output': str(output), 'completed_this_run': count, 'elapsed_s': time.monotonic()-started}), flush=True)


if __name__ == '__main__':
    main()
