"""Matched Qwen presentation control: rectify whole panels from the wide view.

This adds no camera observation. Geometry, output size, panel order and prompt
are fixed before reading the result. No label strings are used to crop.
"""
import argparse
import hashlib
import json
import numpy as np
from PIL import Image
from calibration_geometry import projection_matrix
from panel_env import ROOT


def rectify(image,panel):
    corners=[]
    for u,v in [(0,0),(512,0),(512,512),(0,512)]:
        xyz=np.array([[-.26,0,.26][panel]+(u/512-.5)*.21,.13+(.5-v/512)*.21,.672,1])
        clip=projection_matrix()@xyz;ndc=clip[:2]/clip[3]
        corners.append([(ndc[0]+1)*image.width/2,(1-ndc[1])*image.height/2])
    a=[];b=[]
    for (x,y),(u,v) in zip([(0,0),(383,0),(383,383),(0,383)],corners):
        a.extend([[x,y,1,0,0,0,-u*x,-u*y],[0,0,0,x,y,1,-v*x,-v*y]]);b.extend([u,v])
    coef=np.linalg.solve(np.array(a),np.array(b))
    return image.transform((384,384),Image.Transform.PERSPECTIVE,coef,resample=Image.Resampling.BICUBIC)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--split',default='dev');a=p.parse_args()
    source='panel-control-v1' if a.split.startswith('control_') else 'panel-v3'
    folders=sorted((ROOT/'data'/source/a.split).iterdir())
    for folder in folders:
        out=ROOT/'data'/'panel-initial-crops-v1'/a.split/folder.name
        if (out/'complete.json').exists():
            continue
        out.mkdir(parents=True,exist_ok=True)
        image=Image.open(folder/'context-0.png').convert('RGB')
        images=[]
        for panel in range(3):
            name=f'panel-{panel}.png';rectify(image,panel).save(out/name);images.append(name)
        (out/'complete.json').write_text(json.dumps({'images':images,'source_image':str((folder/'context-0.png').relative_to(ROOT)),
            'source_sha256':hashlib.sha256((folder/'context-0.png').read_bytes()).hexdigest(),
            'script_sha256':hashlib.sha256(__import__('pathlib').Path(__file__).read_bytes()).hexdigest(),
            'geometry':'full 512x512 panel texture quadrilateral, fixed calibrated projection; no label strings'},indent=2))
    print(json.dumps({'split':a.split,'roots':len(folders),'condition':'rectified original wide view'}),flush=True)
