"""Generate manuscript tables directly from complete independent audit JSON."""
import hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];CACHE=ROOT/'cache/results';OUT=ROOT/'manuscript/tables';OUT.mkdir(parents=True,exist_ok=True)
used=[]
def load(name):
 path=CACHE/name;used.append(path);return json.loads(path.read_text(encoding='utf-8'))
def interval(point,ci,digits=3):return f'{point:.{digits}f} [{ci[0]:.{digits}f},{ci[1]:.{digits}f}]'
def write(name,text):(OUT/name).write_text(text+'\n',encoding='utf-8')

def panel():
 splits=['test','test_shift'];ds=[load('round-3-final-'+s+'.json') for s in splits]
 rows=[('Global absolute','state-budget1000--global_abs'),('Global residual','state-budget1000--global_delta'),
  ('Local residual','state-budget1000--local_delta'),('Global centered residual','state-budget1000--global_relative'),
  ('Local centered residual','state-budget1000--local_relative'),('Local gated centered residual','state-budget1000--local_gated_relative'),
  ('Local centered, complete actions','state-controls1000--local_fullaction_relative'),('Global centered, targets hidden','state-controls1000--global_relative_no_targets')]
 lines=[r'\begin{table*}[t]',r'\caption{Independent tabletop CGE. Learned rows average three 1,000-epoch seeds; brackets give paired-root bootstrap intervals. Annotated and PARSeq numerical inputs coincide. $^{\dagger}$Exploratory forecast application of known equations and the training-fitted execution table; observed CGE is one in each test.}\label{tab:panel}',r'\centering\small\setlength{\tabcolsep}{4pt}',r'\begin{tabular}{lcc}',r'\toprule',r'Reference & Ordinary test & Range/background shift\\',r'\midrule']
 for label,key in rows:
  vals=[d['seed_summary']['panel-annotation/'+key] for d in ds]
  lines.append(' & '.join([label]+[interval(v['mean_cge'],v['mean_seed_root_bootstrap_ci95'],4) for v in vals])+r'\\')
 lines.append(r'\midrule')
 for label,key in [('Known rules, estimated gains','known-rules'),('Copy last reading','copy')]:
  vals=[d['results']['panel-annotation/'+key+'-'+s+'.jsonl'] for d,s in zip(ds,splits)]
  lines.append(' & '.join([label]+[interval(v['balanced_accuracy'],v['balanced_ci95'],4) for v in vals])+r'\\')
 if all((CACHE/('round-posthoc-structured-'+s+'.json')).exists() for s in splits):
  vals=[load('round-posthoc-structured-'+s+'.json')['results']['posthoc-rule-table/rules-table-'+s+'.jsonl'] for s in splits]
  lines.append(' & '.join([r'Known rules + execution table$^{\dagger}$']+[f"{v['balanced_accuracy']:.4f}" for v in vals])+r'\\')
  write('rule-table-saturation.json',json.dumps({s:{'observed_cge':v['balanced_accuracy'],'strict_numerical_root_success':v['root_all_steps_all_branches_success'],
    'strict_root_extreme_event_ci95':[.025**(1/v['roots']),1.] if v['root_all_steps_all_branches_success']==1 else v['root_success_ci95'],
    'interpretation':'Exploratory numerical forecast reference using known equations and the training-fitted execution table. Interval applies to strict root success.'} for s,v in zip(splits,vals)},indent=2))
 lines.extend([r'\bottomrule',r'\end{tabular}',r'\end{table*}']);write('panel.tex','\n'.join(lines))
def control():
 splits=['control_test','control_shift'];ds=[load('control-'+s+'.json') for s in splits]
 rows=[('Base centered local','Two-step','Wide','local_fullaction_relative-seedmean-original-full-two_step-wide-'),
 ('Window-augmented local','Two-step','Wide','local_fullaction_relative-seedmean-window_aug-window4-two_step-wide-'),
 ('History-augmented local','Two-step','Wide','local_fullaction_relative-seedmean-history_aug-full-two_step-wide-'),
 ('History-augmented local','Open loop','Wide','local_fullaction_relative-seedmean-history_aug-full-open_loop-wide-'),
 ('History-augmented local','Two-step','Near','local_fullaction_relative-seedmean-history_aug-full-two_step-near-'),
 ('Known rules','Two-step','Wide','rules-seed0-original-full-two_step-wide-'),
 ('Known rules','Open loop','Wide','rules-seed0-original-full-open_loop-wide-'),
 ('Rules + execution table','Two-step','Wide','rules_table-seed0-original-full-two_step-wide-'),
 ('Rules + execution table','Open loop','Wide','rules_table-seed0-original-full-open_loop-wide-'),
 ('Rules + execution table','Two-step','Near','rules_table-seed0-original-full-two_step-near-'),
 ('Copy','Two-step','Wide','copy-seed0-original-full-two_step-wide-'),
 ('Base centered local$^{*}$','Open loop','Wide','local_fullaction_relative-seedmean-original-full-open_loop-wide-')]
 lines=[r'\begin{table*}[t]',r'\caption{Independent goal success on 30 roots per split. Neural rows average three seeds; brackets give root intervals, using conservative extreme-event bounds for all-success outcomes. Near identities use three additional initial views; numerical feedback uses PARSeq. $^{*}$Exploratory open-loop condition.}\label{tab:control}',r'\centering\small\setlength{\tabcolsep}{4pt}',r'\begin{tabular}{lllcc}',r'\toprule',r'Predictor & Execution & Identities & Ordinary goals & Shifted goals\\',r'\midrule']
 for name,feedback,ids,key in rows:
  values=[d['seed_means'][key+s] for d,s in zip(ds,splits)]
  lines.append(' & '.join([name,feedback,ids]+[interval(v['success'],v['success_ci95']) for v in values])+r'\\')
 lines.extend([r'\bottomrule',r'\end{tabular}',r'\end{table*}']);write('control.tex','\n'.join(lines))
 ranges=[]
 for d,s in zip(ds,splits):
  for key,v in d['seed_means'].items():
   if v['seeds']==3:ranges.append({'split':s,'condition':key,'mean':v['success'],'seed_range':v['seed_range']})
 write('control-seed-ranges.json',json.dumps(ranges,indent=2))
def doom():
 if not all((CACHE/f'doom-{s}.json').exists() for s in ['test_seen','test_unseen']):return
 ds=[load('doom-'+s+'.json') for s in ['test_seen','test_unseen']]
 lines=[r'\begin{table*}[t]',r'\caption{Visible HUD CGE for public DoomShift checkpoints. Stock and tuned decoders share each latent prediction. Brackets give paired episode-bootstrap intervals: eight seen-map and thirteen unseen-map episodes. Ground-truth (GT) reconstructions are shared across backbones.}\label{tab:doom}',r'\centering\small\setlength{\tabcolsep}{5pt}',r'\begin{tabular}{llcccc}',r'\toprule',r'Latent source & Tics & Seen: stock & Seen: tuned & Unseen: stock & Unseen: tuned\\',r'\midrule']
 for model,title in [('unet','U-Net prediction'),('pixart','PixArt prediction'),('reconstructed','GT reconstruction')]:
  for h in ['1','16']:
   vals=[]
   for d in ds:
    conditions=d['horizons'][h]['backbones']['unet' if model=='reconstructed' else model]['conditions']
    for condition in (['reconstructed','reconstructed-tuned'] if model=='reconstructed' else ['predicted','predicted-tuned']):
     v=conditions[condition];vals.append(interval(v['cge'],v['cge_ci95']))
   lines.append(' & '.join([title,h]+vals)+r'\\')
 lines.extend([r'\bottomrule',r'\end{tabular}',r'\end{table*}']);write('doom.tex','\n'.join(lines))
def vlm():
 d=load('round-vlm-development-final-dev.json')
 rows=[]
 for name,v in d['results'].items():
  rows.append({'condition':name,'roots':v['roots'],'cge':v['balanced_accuracy'],'ci95':v['balanced_ci95'],'format_failure':v['parse_failure_rate'],'interpretation':'Development only; no cross-supervision architecture ranking. Zero under all-invalid JSON is protocol failure, not evidence of zero dynamics ability.'})
 write('vlm-development.json',json.dumps(rows,indent=2))

def supplement():
 ds=[load('round-3-final-'+s+'.json') for s in ['test','test_shift']]
 names={'global_abs':'Global absolute','global_delta':'Global residual','local_delta':'Local residual',
  'global_relative':'Global centered','local_relative':'Local centered','local_gated_relative':'Local gated centered',
  'local_fullaction_relative':'Local complete actions','global_relative_no_targets':'Global targets hidden'}
 lines=[r'\begin{table*}[t]',r'\caption{Numerical CGE by training budget. Values are three-seed means, with seed minima and maxima in parentheses. Each budget uses development checkpoint selection. Annotated and PARSeq numerical inputs coincide.}\label{tab:sbudget}',r'\centering\small\setlength{\tabcolsep}{4pt}',r'\begin{tabular}{llcc}',r'\toprule',r'Reference & Epochs & Ordinary CGE (seed range) & Shift CGE (seed range)\\',r'\midrule']
 for epochs in [100,1000]:
  for variant,label in names.items():
   if epochs==100 and variant in ['local_fullaction_relative','global_relative_no_targets']:continue
   group='state-controls1000' if variant in ['local_fullaction_relative','global_relative_no_targets'] else 'state-budget'+str(epochs)
   vals=[d['seed_summary']['panel-annotation/'+group+'--'+variant] for d in ds]
   cells=[f"{v['mean_cge']:.4f} ({v['seed_min_max'][0]:.4f},{v['seed_min_max'][1]:.4f})" for v in vals]
   lines.append(' & '.join([label,str(epochs)]+cells)+r'\\')
 lines.extend([r'\bottomrule',r'\end{tabular}',r'\end{table*}']);write('supp-budget.tex','\n'.join(lines))
 lines=[r'\begin{table*}[t]',r'\caption{Strict numerical success across all 48 readings per root. Values average three 1,000-epoch seeds; brackets give paired family-stratified root intervals conditional on these checkpoints. Identifiers are scored separately.}\label{tab:sstrict}',r'\centering\small\setlength{\tabcolsep}{4pt}',r'\begin{tabular}{lcc}',r'\toprule',r'Reference & Ordinary strict root success & Shift strict root success\\',r'\midrule']
 for variant,label in names.items():
  group='state-controls1000' if variant in ['local_fullaction_relative','global_relative_no_targets'] else 'state-budget1000'
  vals=[d['seed_summary']['panel-annotation/'+group+'--'+variant] for d in ds]
  lines.append(' & '.join([label]+[interval(v['mean_numeric_root_success'],v['numeric_root_success_ci95']) for v in vals])+r'\\')
 lines.extend([r'\bottomrule',r'\end{tabular}',r'\end{table*}']);write('supp-strict.tex','\n'.join(lines))
 p=load('perception-controls.json')
 lines=[r'\begin{table*}[t]',r'\caption{Development observation control on 120 roots. Both conditions share the frozen Qwen model, three-image prompt, panel order, and image dimensions. Brackets give paired root intervals. Native views add observations; rectified crops use the initial wide image.}\label{tab:sviews}',r'\centering\small\setlength{\tabcolsep}{4pt}',r'\begin{tabular}{lccc}',r'\toprule',r'Outcome & Rectified wide crops & Native initial views & Native minus crops\\',r'\midrule']
 for key,label in [('identifier_fields','Identifier word exactness'),('all_identifiers','All three identifiers correct'),('numeric_fields','Numerical word exactness')]:
  crop=p['conditions']['rectified_wide'][key];native=p['conditions']['native_camera'][key];diff=p['paired_differences'][key]
  lines.append(' & '.join([label,interval(crop['exact'],crop['ci95']),interval(native['exact'],native['ci95']),interval(diff['native_minus_crops'],diff['ci95'])])+r'\\')
 lines.extend([r'\bottomrule',r'\end{tabular}',r'\end{table*}']);write('supp-views.tex','\n'.join(lines))
 v=load('round-vlm-development-final-dev.json')['results']
 labels={'vlm/internvl-pixels-dev.jsonl':'InternVL, pixels, 384 tokens',
  'vlm/qwen-annotation-dev.jsonl':'Qwen, transcripts, 384 tokens',
  'vlm/qwen-pixels-batched8-dev.jsonl':'Qwen, pixels, 384 tokens',
  'vlm/qwen-pixels-json768-dev.jsonl':'Qwen, pixels, JSON control, 768 tokens',
  'vlm/qwen-annotation-json768-dev.jsonl':'Qwen, transcripts, JSON control, 768 tokens'}
 lines=[r'\begin{table*}[t]',r'\caption{Direct development forecasts on 120 roots and 480 four-step branches. Brackets give family-stratified root intervals. Invalid output is incorrect under the forecast API. The multimodal checkpoints are frozen and receive the public rules.}\label{tab:svlm}',r'\centering\small\setlength{\tabcolsep}{4pt}',r'\begin{tabular}{lcc}',r'\toprule',r'Forecast condition & CGE & Invalid format fraction\\',r'\midrule']
 for key,label in labels.items():
  if key not in v:continue
  r=v[key];lines.append(' & '.join([label,interval(r['balanced_accuracy'],r['balanced_ci95']),f"{r['parse_failure_rate']:.3f}"])+r'\\')
 lines.extend([r'\bottomrule',r'\end{tabular}',r'\end{table*}']);write('supp-vlm.tex','\n'.join(lines))
 if (CACHE/'doom-raw-label-audit.json').exists():
  raw=load('doom-raw-label-audit.json')
  lines=[r'\begin{table*}[t]',r'\caption{Seven visible health endpoints matching the current tic in episode 37, map 16, at horizon one. Each has zero glyph-template distance. The visible readings are the evaluation labels.}\label{tab:stiming}',r'\centering\small\setlength{\tabcolsep}{4pt}',r'\begin{tabular}{lcccc}',r'\toprule',r'Window index / role & Raw row & Visible health & Previous-tic health & Current-tic health\\',r'\midrule']
  for r in raw['exceptions']:
   i=r['nearby_rows'].index(r['row'])
   lines.append(' & '.join([str(r['index'])+' / '+r['role'],str(r['row']),str(r['visible']['health']),str(r['aligned_gamevars']['health']),str(r['nearby_gamevars']['health'][i])])+r'\\')
  lines.extend([r'\bottomrule',r'\end{tabular}',r'\end{table*}']);write('supp-doom-timing.tex','\n'.join(lines))
 if not all((CACHE/f'doom-{s}.json').exists() for s in ['test_seen','test_unseen']):return
 doom_data=[load('doom-'+s+'.json') for s in ['test_seen','test_unseen']]
 if any('secondary_parseq' not in entry for d in doom_data for h in d['horizons'].values() for entry in h['backbones'].values()):return
 lines=[r'\begin{table*}[t]',r'\caption{Changed and kept HUD exactness entering CGE. Both architectures and decoders use the same event-stratified windows and visible labels.}\label{tab:sfields}',r'\centering\small\setlength{\tabcolsep}{4pt}',r'\begin{tabular}{lllcccc}',r'\toprule',r'Maps / tics & Backbone & Decoder & Ammo changed & Ammo kept & Health changed & Health kept\\',r'\midrule']
 for d,title in zip(doom_data,['Seen','Unseen']):
  for h in ['1','16']:
   for model in ['unet','pixart']:
    for condition,label in [('predicted','Stock'),('predicted-tuned','Tuned')]:
     f=d['horizons'][h]['backbones'][model]['conditions'][condition]['fields']
     vals=[f[field][metric] for field in ['ammo','health'] for metric in ['changed_exact','unchanged_exact']]
     lines.append(' & '.join([title+' / '+h,{'unet':'U-Net','pixart':'PixArt'}[model],label]+[f'{x:.3f}' for x in vals])+r'\\')
 lines.extend([r'\bottomrule',r'\end{tabular}',r'\end{table*}']);write('supp-doom-fields.tex','\n'.join(lines))
 agreement=[]
 for d,title in zip(doom_data,['Seen','Unseen']):
  lines=[r'\begin{table}[t]',r'\caption{HUD reader sensitivity on '+title.lower()+r' maps. Fixed and PARSeq columns give CGE; PSNR is in dB. GT denotes ground-truth latent reconstruction.}\label{tab:sreader'+title.lower()+r'}',r'\centering\small\setlength{\tabcolsep}{4pt}',r'\begin{tabular}{lrrr}',r'\toprule',r'Image source & Fixed & PARSeq & PSNR\\',r'\midrule']
  for h in ['1','16']:
   lines.append(r'\multicolumn{4}{l}{\textit{Horizon: '+h+r' tic'+('s' if h!='1' else '')+r'}}\\')
   models=d['horizons'][h]['backbones']
   for model,condition,label in [('unet','previous','Raw last image'),('unet','copy-decoded','Stock decoded last'),('unet','copy-decoded-tuned','Tuned decoded last'),('unet','reconstructed','Stock GT reconstruction'),('unet','reconstructed-tuned','Tuned GT reconstruction'),('unet','predicted','U-Net stock prediction'),('unet','predicted-tuned','U-Net tuned prediction'),('pixart','predicted','PixArt stock prediction'),('pixart','predicted-tuned','PixArt tuned prediction')]:
    entry=models[model];fixed=entry['conditions'][condition];secondary=entry['secondary_parseq']['conditions'][condition]
    short={'Raw last image':'Raw previous','Stock decoded last':'Stock last','Tuned decoded last':'Tuned last','Stock GT reconstruction':'Stock GT','Tuned GT reconstruction':'Tuned GT','U-Net stock prediction':'U-Net stock','U-Net tuned prediction':'U-Net tuned','PixArt stock prediction':'PixArt stock','PixArt tuned prediction':'PixArt tuned'}[label]
    lines.append(' & '.join([short,f"{fixed['cge']:.3f}",f"{secondary['cge']:.3f}",f"{fixed['hud_psnr']:.2f}"])+r'\\')
   if h=='1':lines.append(r'\midrule')
   agreement.append({'maps':title,'horizon':h,**models['unet']['secondary_parseq']['raw_target_agreement']})
  lines.extend([r'\bottomrule',r'\end{tabular}',r'\end{table}']);write('supp-doom-reader-'+title.lower()+'.tex','\n'.join(lines))
 paired=[]
 for title in ['seen','unseen']:
  body=(OUT/('supp-doom-reader-'+title+'.tex')).read_text(encoding='utf-8')
  body=body.replace(r'\begin{table}[t]','').replace(r'\end{table}','').strip()
  paired.append(r'\begin{minipage}[t]{0.48\textwidth}'+'\n'+body+'\n'+r'\end{minipage}')
 write('supp-doom-readers.tex',r'\begin{table*}[t]'+'\n'+paired[0]+'\n'+r'\hfill'+'\n'+paired[1]+'\n'+r'\end{table*}')
 write('supp-doom-reader-agreement.json',json.dumps(agreement,indent=2))
 lines=[r'\begin{table}[t]',r'\caption{PARSeq agreement with raw visible targets. These are sample transcription rates, not generation scores.}\label{tab:srawreader}',r'\centering\small\setlength{\tabcolsep}{4pt}',r'\begin{tabular}{lcc}',r'\toprule',r'Maps / tics & Ammo & Health\\',r'\midrule']
 for r in agreement:lines.append(' & '.join([r['maps']+' / '+r['horizon'],f"{r['ammo']:.4f}",f"{r['health']:.4f}"])+r'\\')
 lines.extend([r'\bottomrule',r'\end{tabular}',r'\end{table}']);write('supp-doom-agreement.tex','\n'.join(lines))
def character_readout():
 d=load('hud-character-metrics.json')['results']
 rows=[('previous','Raw previous'),('gt-stock','GT stock'),('gt-tuned','GT tuned'),
  ('unet-stock','U-Net stock'),('unet-tuned','U-Net tuned'),('pixart-stock','PixArt stock'),('pixart-tuned','PixArt tuned')]
 lines=[r'\begin{table*}[t]',r'\caption{Secondary CER on the independent HUD outputs, using identical raw visible target strings for both readers. Fixed denotes the calibrated font reader. Values are aggregate edit counts divided by target characters; paired decoder-difference intervals are reported in the text. Lower is better.}\label{tab:scharacters}',r'\centering\small\setlength{\tabcolsep}{4pt}',r'\begin{tabular}{lrrrrrrrr}',r'\toprule',r' & \multicolumn{2}{c}{Seen: 1 tic} & \multicolumn{2}{c}{Seen: 16 tics} & \multicolumn{2}{c}{Unseen: 1 tic} & \multicolumn{2}{c}{Unseen: 16 tics}\\',r'\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-7}\cmidrule(lr){8-9}',r'Image source & Fixed & PARSeq & Fixed & PARSeq & Fixed & PARSeq & Fixed & PARSeq\\',r'\midrule']
 for source,label in rows:
  vals=[d[f'{split}/{h}/{source}/{reader}'] for split in ['seen','unseen'] for h in [1,16] for reader in ['fixed','parseq']]
  lines.append(' & '.join([label]+[f"{v['cer']:.3f}" for v in vals])+r'\\')
 lines.extend([r'\bottomrule',r'\end{tabular}',r'\end{table*}']);write('supp-hud-characters.tex','\n'.join(lines))

panel();control();doom();vlm();supplement();character_readout()
(OUT/'provenance.json').write_text(json.dumps({'generator_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'inputs':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in used}},indent=2),encoding='utf-8')
print(json.dumps({'generated':[p.name for p in OUT.iterdir() if p.is_file()]}))
