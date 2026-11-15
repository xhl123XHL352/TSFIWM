"""PARSeq control with calibrated text regions in the original wide image.

The quadrilateral comes from the shared fixed camera/panel geometry, not from
the label string. This is a localized-reading condition, not text detection.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
import numpy as np
from PIL import Image
from gpu_accounting import ROOT, GpuLease


def projection_matrix():
    eye = np.array([.70, -.65, 1.45])
    forward = np.array([0, .15, .65])-eye
    forward /= np.linalg.norm(forward)
    side = np.cross(forward, [0,0,1])
    side /= np.linalg.norm(side)
    up = np.cross(side, forward)
    view = np.eye(4)
    view[:3,:3] = [side, up, -forward]
    view[:3,3] = [-side@eye, -up@eye, forward@eye]
    f, near, far = 1/np.tan(np.deg2rad(48)/2), .05, 4
    proj = np.array([[f/(768/640),0,0,0],[0,f,0,0],
              [0,0,(far+near)/(near-far),2*far*near/(near-far)],[0,0,-1,0]])
    return proj@view


def crop(image, panel, field):
    u0, v0, u1, v1 = (75,40,437,145) if field == 'id' else (70,180,442,355)
    corners = []
    for u,v in [(u0,v0),(u1,v0),(u1,v1),(u0,v1)]:
        xyz = np.array([[-.26,0,.26][panel]+(u/512-.5)*.21, .13+(.5-v/512)*.21, .672, 1])
        clip = projection_matrix()@xyz
        ndc = clip[:2]/clip[3]
        corners.append([(ndc[0]+1)*image.width/2, (1-ndc[1])*image.height/2])
    destination = [(0,0),(255,0),(255,63),(0,63)]
    a,b = [],[]
    for (x,y),(u,v) in zip(destination,corners):
        a.extend([[x,y,1,0,0,0,-u*x,-u*y],[0,0,0,x,y,1,-v*x,-v*y]])
        b.extend([u,v])
    coefficients = np.linalg.solve(np.array(a),np.array(b))
    return image.transform((256,64),Image.Transform.PERSPECTIVE,coefficients,
                            resample=Image.Resampling.BICUBIC)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--split',default='dev')
    parser.add_argument('--audit',action='store_true')
    parser.add_argument('--score',action='store_true')
    parser.add_argument('--gpu-uuid')
    args = parser.parse_args()
    folders = sorted((ROOT/'data'/'panel-v3'/args.split).iterdir())
    output = ROOT/'results'/'perception'/f'parseq-calibrated-{args.split}.jsonl'
    if args.audit:
        image = Image.open(folders[0]/'context-0.png').convert('RGB')
        canvas = Image.new('RGB',(256*3,64*2),'white')
        for panel in range(3):
            for row,field in enumerate(['id','value']):
                canvas.paste(crop(image,panel,field),(panel*256,row*64))
        canvas.save(ROOT/'data'/'calibrated-word-audit.png')
        print(str(ROOT/'data'/'calibrated-word-audit.png'))
        return
    if args.score:
        records = [json.loads(s) for s in output.read_text().splitlines()]
        expected = len(folders)*4*3*2
        if len(records)!=expected:
            raise RuntimeError('Reading comparison incomplete')
        correct = {'id':[],'value':[]}
        casefold=[]
        for r in records:
            state = json.loads((ROOT/'data'/'panel-v3'/args.split/r['root_id']/'labels.json').read_text())['context_states'][r['time']]
            target = state['ids'][r['panel']] if r['field']=='id' else f"{state['values'][r['panel']]:03d}"
            correct[r['field']].append(r['text']==target)
            if r['field']=='id':
                casefold.append(r['text'].upper()==target)
        result={'word_regions':len(records),'identifier_exact':float(np.mean(correct['id'])),
                'identifier_casefold':float(np.mean(casefold)), 'numeric_transcript_exact':float(np.mean(correct['value'])),
                'condition':'original wide pixels, calibrated word-region rectification, PARSeq base'}
        output.with_suffix('.metrics.json').write_text(json.dumps(result,indent=2))
        print(json.dumps(result))
        return
    with GpuLease(args.gpu_uuid,'parseq-calibrated-'+args.split,max_memory_gib=4):
        import torch
        sys.path.insert(0,str(ROOT/'cache'/'repos'/'parseq'))
        from strhub.models.utils import create_model
        model=create_model('parseq',pretrained=False).eval()
        state=torch.load(ROOT/'models'/'parseq'/'parseq-bb5792a6.pt',map_location='cpu',weights_only=True)
        model.model.load_state_dict(state,strict=True)
        model=model.cuda()
        tasks=[]
        for folder in folders:
            for t in range(4):
                for panel in range(3):
                    for field in ['id','value']:
                        tasks.append((folder,t,panel,field))
        output.parent.mkdir(parents=True,exist_ok=True)
        if output.exists():
            raise RuntimeError('Preserve existing OCR evidence; inspect before rerunning')
        with output.open('w') as f,torch.inference_mode():
            for start in range(0,len(tasks),128):
                batch=tasks[start:start+128]
                tensors=[]
                for folder,t,panel,field in batch:
                    image=Image.open(folder/f'context-{t}.png').convert('RGB')
                    word=crop(image,panel,field).resize((128,32),Image.Resampling.BICUBIC)
                    tensors.append(np.asarray(word,dtype=np.float32).transpose(2,0,1)/127.5-1)
                probabilities=model(torch.as_tensor(np.stack(tensors),device='cuda')).softmax(-1)
                labels,confidences=model.tokenizer.decode(probabilities)
                for task,text,confidence in zip(batch,labels,confidences):
                    folder,t,panel,field=task
                    f.write(json.dumps({'root_id':folder.name,'time':t,'panel':panel,'field':field,
                        'text':text,'token_probability_product':float(confidence.prod())})+'\n')
                f.flush()
                if start%512==0:
                    print(json.dumps({'words_read':start+len(batch),'total':len(tasks),
                          'peak_allocated_gib':torch.cuda.max_memory_allocated()/1024**3}),flush=True)


if __name__=='__main__':
    main()
