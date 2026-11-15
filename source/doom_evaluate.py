"""Strictly load the full public DoomShift UNet and retain native-DDIM outputs.

This is an evaluation adapter, not a smaller reproduction of the public model.
All data/code/checkpoint paths are relative to the private research project.
"""
from __future__ import annotations
import argparse
import hashlib
import io
import json
from pathlib import Path
import sys
import time
from unittest.mock import patch
from functools import lru_cache
import numpy as np
from gpu_accounting import ROOT, GpuLease

REPO = ROOT/'cache'/'repos'/'doomshift'
sys.path.insert(0, str(REPO))


@lru_cache(maxsize=1)
def load_full_model(backbone='unet'):
    import torch
    from diffusers import UNet2DConditionModel, PixArtTransformer2DModel
    from backbones import build_model
    from diffusion_v import checkpoint_objective
    run = '040-unet-nexttic' if backbone=='unet' else '041-pixart-nexttic'
    ck = torch.load(ROOT/'models'/'doomshift'/run/'snap_0200000.pt', map_location='cpu', weights_only=True, mmap=True)
    a = ck['args']
    # The full trained state below overwrites EVERY tensor. Loading warm-start
    # SD weights before overwriting them would only repeat a 3.4GB download.
    config = json.loads((ROOT/'models'/'doom-base'/('sd14/unet' if backbone=='unet' else 'pixart/transformer')/'config.json').read_text())
    model_class = UNet2DConditionModel if backbone=='unet' else PixArtTransformer2DModel
    def architecture_only(*unused, **kw):
        return model_class.from_config(config, **({'num_class_embeds':kw['num_class_embeds']}
            if backbone=='unet' else {'use_additional_conditions':False}))
    bits = int(a.get('resolved_control_bits') or a.get('control_bits') or 0)
    assert a['tic_stride'] == 1 and a['action_history'] == 32 and bits == 19, a
    assert a['context_frames'] == 32 and a.get('objective', 'v') == 'v', a
    with patch.object(model_class, 'from_pretrained', side_effect=architecture_only):
        model = build_model(backbone, a['num_actions'], a['context_frames'],
            a['noise_buckets'], grad_ckpt=False, action_dropout=a['action_dropout'],
            latent_channels=4, phase_buckets=0, action_history=32, control_bits=bits)
    report = model.load_state_dict(ck['model'], strict=True)
    count = sum(p.numel() for p in model.parameters())
    assert count == (860532932 if backbone=='unet' else 628481184) and not report.missing_keys and not report.unexpected_keys
    info = {'backbone':backbone, 'parameters': count, 'missing_keys': report.missing_keys,
        'unexpected_keys': report.unexpected_keys, 'checkpoint_step': ck.get('step'),
        'weight_source': 'model', 'checkpoint_dtype': str(next(iter(ck['model'].values())).dtype),
        'objective': checkpoint_objective(ck), 'action_history': 32, 'control_bits': bits,
        'context_frames': 32, 'tic_stride': 1, 'torch': torch.__version__,
        'construction': 'Exact public architecture from pinned base config, then strict full released state load; no warm-start tensors left unloaded',
        'source_revision': (REPO/'SOURCE_REVISION').read_text().strip()}
    return model.eval().requires_grad_(False), info


@lru_cache(maxsize=2)
def cached_vae(path):
    from doomdit_utils import build_vae
    return build_vae(path,device='cuda',latent_channels=4,scaling_factor=.18215)


def key_noise(shape, episode, start, seed, offset, device):
    import torch
    digest = hashlib.sha256(f'{seed}:{episode}:{start}:{offset}'.encode()).digest()
    g = torch.Generator(device='cpu').manual_seed(int.from_bytes(digest[:8], 'little') % (2**63-1))
    return torch.randn(shape, generator=g).to(device)


def evaluate(args):
    import torch
    import pyarrow.parquet as pq
    from PIL import Image
    from doom_data import TicWindowDataset
    from doomdit_utils import build_vae
    from diffusion_v import VDiffusion
    torch.set_num_threads(4)
    model, identity = load_full_model(args.backbone)
    out = ROOT/'results'/'doomshift'/args.name
    if (out/'complete.json').exists():
        raise RuntimeError('Preserve existing complete experiment: '+str(out))
    out.mkdir(parents=True, exist_ok=True)
    (out/'model-identity.json').write_text(json.dumps(identity, indent=2))
    print(json.dumps(identity), flush=True)
    if args.audit_only:
        return
    ds = TicWindowDataset(ROOT/'data'/'doomshift'/'latents'/'sd15'/args.latent_segment,
        episode_ids=[args.episode], context_frames=32, horizon=args.horizon,
        with_horizon=True, action_history=32)
    raw = pq.read_table(ROOT/'data'/'doomshift'/'raw'/args.raw_segment/f'ep_{args.episode:05}.parquet').to_pydict()
    meta = np.load(ROOT/'data'/'doomshift'/'latents'/'sd15'/args.latent_segment/f'ep_{args.episode:05}_meta.npz')
    for field in ['health', 'ammo', 'tic', 'episode_id', 'map_id']:
        assert np.array_equal(meta[field], np.array(raw[field])), 'Raw/latent row mismatch: '+field
    # Development-only strata. Model inputs never include these target values.
    rng = np.random.default_rng(args.seed)
    strata = {False: [], True: []}
    for i in range(len(ds)):
        slot, s = ds.locate(i)
        t = s+32
        changed = any(meta[f][t+args.horizon-1] != meta[f][t-1] for f in ['health','ammo'])
        strata[bool(changed)].append(i)
    selected_protocol = None
    if args.windows_file:
        selected_protocol = json.loads(Path(args.windows_file).read_text())
        assert selected_protocol['episode']==args.episode and selected_protocol['horizon']==args.horizon
        selected = selected_protocol['indices']
    else:
        selected = sorted([int(i) for c in [False, True] for i in rng.choice(strata[c],
            size=min(args.windows//2, len(strata[c])), replace=False)])
    protocol = {'episode': args.episode, 'indices': selected, 'horizon': args.horizon,
        'development_only': 6000<=args.episode<6100, 'backbone':args.backbone,
        'strata': {str(k):len(v) for k,v in strata.items()},
        'steps': 50, 'eta': 0, 'spacing': 'linear', 'context_noise': 0,
        'sample_seed': args.seed, 'dtype_forward': 'autocast bf16, fp32 model and DDIM state',
        'vae': 'stabilityai/sd-vae-ft-mse@31f26fdeee1355a5c34592e401dd41e45d25a493',
        'decoder_dtype': 'float32', 'scale': .18215, 'decoder_crop': 'first 240 rows',
        'selection': selected_protocol['selection'] if selected_protocol else 'Equal changed/unchanged physical gamevar endpoint strata; engineering diagnostic, not visibility-aligned corpus estimate',
        'windows_source':args.windows_file,
        'tuned_decoder':args.tuned_decoder,
        'source_hash': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'dataset_summary': ds.summary}
    (out/'protocol.json').write_text(json.dumps(protocol, indent=2))
    model = model.cuda()
    vae = cached_vae(str(ROOT/'models'/'doom-base'/'stock-vae'))
    tuned = cached_vae(str(ROOT/'models'/'doomshift'/'decoders'/'vae_decoder_sd1x_mse_lpips'/'vae')) if args.tuned_decoder else None
    diffusion = VDiffusion(device='cuda', objective=identity['objective'])
    def decode(z, decoder=None):
        return ((decoder or vae).decode(z.float()/.18215).sample[:,:,:240]*.5+.5).clamp(0,1)
    def save_img(tensor, path):
        array = (tensor.detach().float().cpu().permute(1,2,0).numpy()*255).round().astype(np.uint8)
        Image.fromarray(array).save(path)
        return array
    predictions = []
    started = time.monotonic()
    with torch.inference_mode(), (out/'windows.jsonl').open('w') as log:
        for idx in selected:
            ctx, tgts, acts, phases = ds[idx]
            slot, start = ds.locate(idx)
            ctx, tgts, acts = ctx[None].cuda(), tgts.cuda(), acts.cuda()
            run = ctx
            for step in range(args.horizon):
                noise = key_noise((1,4,32,40), args.episode, start, args.seed, step, 'cuda')
                bucket = torch.zeros(1, dtype=torch.long, device='cuda')
                act = acts[step:step+1]
                with torch.autocast('cuda', dtype=torch.bfloat16):
                    pred = diffusion.ddim_sample(lambda xt,t:model(xt,t,act,run,bucket),
                        noise.shape, steps=50, eta=0, noise=noise, device='cuda')
                if step < args.horizon-1:
                    run = torch.cat([run[:,4:], pred.float()], 1)
            target_row = start+32+args.horizon-1
            previous_row = start+31
            images = decode(torch.cat([pred, tgts[-1:], ctx[:,-4:]], 0))
            folder = out/'frames'/f'window-{idx:06}'
            folder.mkdir(parents=True, exist_ok=True)
            raw_target = Image.open(io.BytesIO(raw['frame'][target_row])).convert('RGB')
            raw_previous = Image.open(io.BytesIO(raw['frame'][previous_row])).convert('RGB')
            raw_target.save(folder/'target.png')
            raw_previous.save(folder/'previous.png')
            pix = [save_img(images[k], folder/f'{name}.png') for k,name in enumerate(['predicted','reconstructed','copy-decoded'])]
            if tuned is not None:
                tuned_images = decode(torch.cat([pred, tgts[-1:], ctx[:,-4:]], 0), tuned)
                for k,name in enumerate(['predicted','reconstructed','copy-decoded']):
                    save_img(tuned_images[k], folder/f'{name}-tuned.png')
            reference = np.asarray(raw_target).astype(float)/255
            hud_psnr = []
            for array in pix:
                mse = np.mean((array[208:].astype(float)/255-reference[208:])**2)
                hud_psnr.append(float(-10*np.log10(max(mse,1e-10))))
            record = {'index':idx, 'episode':args.episode, 'map':int(meta['map_id'][target_row]),
                'start':start, 'previous_row':previous_row, 'target_row':target_row,
                'previous':{f:int(meta[f][previous_row]) for f in ['health','ammo']},
                'target':{f:int(meta[f][target_row]) for f in ['health','ammo']},
                'changed':{f:bool(meta[f][previous_row]!=meta[f][target_row]) for f in ['health','ammo']},
                'latent_mse':float((pred-tgts[-1:]).square().mean()),
                'hud_psnr_pred_raw':hud_psnr[0], 'hud_psnr_vae_raw':hud_psnr[1],
                'hud_psnr_copy_dec_raw':hud_psnr[2], 'images':str(folder.relative_to(ROOT)),
                'elapsed_s':time.monotonic()-started}
            log.write(json.dumps(record)+'\n'); log.flush()
            predictions.append(pred.detach().cpu().half().numpy()[0])
            print(json.dumps(record), flush=True)
    np.save(out/'predicted-latents.npy', np.stack(predictions))
    complete = {'windows':len(selected), 'elapsed_s':time.monotonic()-started,
        'gpu_peak_gib':torch.cuda.max_memory_allocated()/1024**3}
    (out/'complete.json').write_text(json.dumps(complete, indent=2))
    print(json.dumps(complete), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--gpu-uuid', required=True)
    p.add_argument('--episode', type=int, default=6000)
    p.add_argument('--backbone', choices=['unet','pixart'], default='unet')
    p.add_argument('--latent-segment', default='val')
    p.add_argument('--raw-segment', default='arenas')
    p.add_argument('--tuned-decoder', action='store_true')
    p.add_argument('--windows', type=int, default=16)
    p.add_argument('--horizon', type=int, default=1)
    p.add_argument('--seed', type=int, default=1911)
    p.add_argument('--name', required=True)
    p.add_argument('--windows-file')
    p.add_argument('--audit-only', action='store_true')
    args = p.parse_args()
    assert args.name.replace('-','').isalnum()
    if not 6000<=args.episode<6100 and not args.audit_only and not (ROOT/'results'/'doom-evaluation-freeze.json').exists():
        raise RuntimeError('Unseen-map independent evaluation requires protocol freeze')
    if args.audit_only:
        evaluate(args)
    else:
        with GpuLease(args.gpu_uuid, 'doomshift-'+args.name, max_memory_gib=8):
            evaluate(args)
