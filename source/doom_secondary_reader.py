"""Reader sensitivity control; released PARSeq, RGB HUD word crops, no tuning.

The primary native-font target transcription is retained. Raw-target PARSeq
agreement is reported, rather than assuming a generic reader is valid on HUDs.
"""
import argparse,json,os,sys
from pathlib import Path
os.environ['CUDA_VISIBLE_DEVICES']=''
import numpy as np
from PIL import Image
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'cache/repos/parseq'))
from strhub.models.utils import create_model
p=argparse.ArgumentParser();p.add_argument('--subset',choices=['development','test-seen','test-unseen'],required=True);a=p.parse_args()
torch.set_num_threads(4)
model=create_model('parseq',pretrained=False).eval()
weights=torch.load(ROOT/'models/parseq/parseq-bb5792a6.pt',map_location='cpu',weights_only=True)
model.model.load_state_dict(weights,strict=True)
for folder in sorted((ROOT/'results/doomshift').glob(a.subset+'-*')):
    if not (folder/'complete.json').exists():continue
    output=folder/'parseq-readout.json'
    if output.exists():continue
    records=json.loads((folder/'template-readout.json').read_text())
    tasks=[]
    for index,r in enumerate(records):
        for condition in r['readout']:
            for field,x in [('ammo',5),('health',51)]:tasks.append((index,condition,field,x))
        r['parseq']={}
    with torch.inference_mode():
        for offset in range(0,len(tasks),128):
            batch=tasks[offset:offset+128];images=[]
            for index,condition,field,x in batch:
                image=Image.open(ROOT/records[index]['images']/(condition+'.png')).convert('RGB')
                word=image.crop((x,210,x+39,228)).resize((128,32),Image.Resampling.BICUBIC)
                images.append(np.asarray(word,dtype=np.float32).transpose(2,0,1)/127.5-1)
            probs=model(torch.as_tensor(np.stack(images))).softmax(-1);labels,confidence=model.tokenizer.decode(probs)
            for (index,condition,field,x),text,conf in zip(batch,labels,confidence):
                records[index]['parseq'].setdefault(condition,{})[field]={'text':text,'value':int(text) if text.isdigit() else None,
                    'token_probability_product':float(conf.prod())}
    output.write_text(json.dumps(records))
    print(json.dumps({'completed_secondary_reader':folder.name,'words':len(tasks)}),flush=True)
