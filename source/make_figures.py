"""Evidence-bound paper figures. All quantitative inputs are canonical audits.

Destination: IJDAR two-column paper, 6.9in width, Times New Roman >=9pt.
Figures: observed actuator witness, matched training budgets, and independent
control / decoder comparisons when their complete audits are available.
PDF and SVG retain live vector text; recorded simulator images stay raster.
"""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from PIL import Image
ROOT=Path(__file__).resolve().parents[1]
CACHE=ROOT/'cache/results';OUT=ROOT/'manuscript/figures';OUT.mkdir(parents=True,exist_ok=True)
try:
    font_manager.findfont('Times New Roman',fallback_to_default=False)
    figure_font='Times New Roman'
except ValueError:
    figure_font='DejaVu Serif'
plt.rcParams.update({'font.family':figure_font,'font.size':10,'axes.titlesize':11,
    'axes.labelsize':10,'xtick.labelsize':9,'ytick.labelsize':9,'legend.fontsize':9,
    'pdf.fonttype':42,'ps.fonttype':42,'svg.fonttype':'none','axes.spines.top':False,'axes.spines.right':False})
used=[]
def load(name):
    p=CACHE/name;used.append(p);return json.loads(p.read_text(encoding='utf-8'))
def save(fig,name):
    fig.savefig(OUT/(name+'.pdf'),bbox_inches='tight',pad_inches=.025)
    fig.savefig(OUT/(name+'.png'),dpi=220,bbox_inches='tight',pad_inches=.025)
    fig.savefig(OUT/(name+'.svg'),bbox_inches='tight',pad_inches=.025)
    plt.close(fig)


def witness():
    fig=plt.figure(figsize=(6.9,4.55))
    panel=fig.add_axes([0,0,1,1]);panel.set_axis_off();panel.set_xlim(0,1);panel.set_ylim(0,1)
    panel.text(.01,.985,'(a) Diagnostic controls for visual text',va='top',fontsize=10)
    columns=[(.17,'Observation',['Pixels / transcripts','Identifiers and numbers','Identifier / value exactness']),
        (.51,'State update',['Commands and observed history','Paired action branches','CGE / strict root success']),
        (.84,'Rendering',['Fixed predicted latent','Stock / tuned decoder','CGE / character error'])]
    for x,title,lines in columns:
        panel.text(x,.925,title,ha='center',fontweight='bold',fontsize=10)
        panel.plot([x-.13,x+.13],[.905,.905],color='#60707a',lw=.65)
        for y,line in zip([.865,.82,.775],lines):panel.text(x,y,line,ha='center',fontsize=9)
    for x0,x1 in [(.315,.35),(.655,.69)]:
        panel.annotate('',xy=(x1,.925),xytext=(x0,.925),arrowprops={'arrowstyle':'->','lw':.8,'color':'#48545c'})
    panel.text(.5,.705,'Identifiers persist; numerical text follows the executed transition.',ha='center',fontsize=10,fontstyle='italic')
    axes=[fig.add_axes([.005,.115,.475,.54]),fig.add_axes([.52,.115,.475,.54])]
    for ax,name,label,value in zip(axes,['alias-left-pose.png','alias-right-pose.png'],['(b) After requesting slot 1','(c) After requesting slot 2'],[56,53]):
        p=ROOT/'cache/figure-assets'/name;used.append(p)
        ax.imshow(Image.open(p));ax.set_axis_off();ax.set_title(label,pad=5)
        ax.text(.5,-.035,f'Next true slot 0 value: {value}',transform=ax.transAxes,ha='center',va='top',fontsize=10)
    fig.text(.5,.018,'Shared slot 0 history: 50, 53, 53, 53 → 53; next request: slot 0, Δt = 2',ha='center',fontsize=10)
    save(fig,'actuator-witness')


def budget():
    data=load('round-2-dev.json')['results']
    variants=['global_abs','global_delta','local_delta','global_relative','local_relative','local_gated_relative']
    labels=['Global absolute','Global residual','Local residual','Global centered','Local centered','Local gated centered']
    matrix=np.array([[np.mean([data[f'state-budget{epochs}/{v}-seed-{seed}/dev-predictions.jsonl']['balanced_accuracy'] for seed in [17,29,43]]) for epochs in [100,1000]] for v in variants])
    assert np.isfinite(matrix).all() and (matrix>=0).all() and (matrix<=1).all()
    fig,ax=plt.subplots(figsize=(6.9,2.65))
    for i,(early,late) in enumerate(matrix):
        ax.plot([early,late],[i,i],color='#b7c4c9',lw=1.8,zorder=1)
    ax.scatter(matrix[:,0],range(6),facecolors='white',edgecolors='#525a60',s=30,zorder=3,label='100 epochs')
    ax.scatter(matrix[:,1],range(6),color='#216778',marker='D',s=26,zorder=3,label='1,000 epochs')
    ax.set_yticks(range(6),labels=labels);ax.set_ylim(5.55,-.55)
    ax.set_xlim(0,1);ax.set_xticks([0,.25,.5,.75,1]);ax.set_xlabel('Development CGE')
    ax.tick_params(axis='y',length=0,pad=8);ax.spines['left'].set_visible(False)
    ax.grid(axis='x',color='#e7eaec',lw=.6);ax.set_axisbelow(True)
    ax.legend(frameon=False,ncol=2,loc='lower center',bbox_to_anchor=(.5,1.015),borderaxespad=0)
    fig.subplots_adjust(left=.26,right=.985,top=.87,bottom=.18)
    save(fig,'training-budget')


def control():
    if not all((CACHE/f'control-{s}.json').exists() for s in ['control_test','control_shift']):return
    rows=[('Base + feedback','local_fullaction_relative-seedmean-original-full-two_step-wide-'),
        ('Window aug. + feedback','local_fullaction_relative-seedmean-window_aug-window4-two_step-wide-'),
        ('History aug. + feedback','local_fullaction_relative-seedmean-history_aug-full-two_step-wide-'),
        ('History aug. + open loop','local_fullaction_relative-seedmean-history_aug-full-open_loop-wide-'),
        ('Rules + table + feedback','rules_table-seed0-original-full-two_step-wide-')]
    fig,axes=plt.subplots(1,2,figsize=(6.9,2.8),sharey=True)
    for ax,split,title in zip(axes,['control_test','control_shift'],['(a) Ordinary goals','(b) Shifted goals']):
        data=load(f'control-{split}.json')['seed_means']
        for i,(label,key) in enumerate(rows):
            r=data[key+split];v=r['success'];lo,hi=r['success_ci95']
            ax.errorbar(v,i,xerr=[[v-lo],[hi-v]],fmt='o',capsize=2,color='#156b89' if i<4 else '#98572d',ms=4)
        ax.set_xlim(0,1.03);ax.set_xticks([0,.25,.5,.75,1]);ax.set_xlabel('Goal success');ax.set_title(title,loc='left');ax.set_yticks(range(len(rows)),labels=[r[0] for r in rows]);ax.grid(axis='x',color='#e7eaec',lw=.6);ax.tick_params(axis='y',length=0);ax.spines['left'].set_visible(False)
    axes[0].invert_yaxis();fig.subplots_adjust(left=.265,right=.985,top=.86,bottom=.18,wspace=.16);save(fig,'goal-control')


def doom():
    if not all((CACHE/f'doom-{s}.json').exists() for s in ['test_seen','test_unseen']):return
    fig,axes=plt.subplots(1,2,figsize=(6.9,2.75),sharey=True)
    colors={'unet':'#156b89','pixart':'#98572d'}
    for ax,subset,title in zip(axes,['test_seen','test_unseen'],['(a) Seen maps (8 episodes)','(b) Unseen maps (13 episodes)']):
        data=load(f'doom-{subset}.json')
        for b,offset in [('unet',-.04),('pixart',.04)]:
            for condition,style,marker in [('predicted','--','o'),('predicted-tuned','-','s')]:
                x=[];y=[];low=[];high=[]
                for h in ['1','16']:
                    r=data['horizons'][h]['backbones'][b]['conditions'][condition];v=r['cge'];lo,hi=r['cge_ci95']
                    x.append(int(h)+(offset*10 if h=='16' else offset));y.append(v);low.append(v-lo);high.append(hi-v)
                ax.errorbar(x,y,yerr=[low,high],color=colors[b],linestyle=style,marker=marker,ms=3,capsize=2,label={'unet':'U-Net','pixart':'PixArt'}[b]+(' stock' if condition=='predicted' else ' tuned'))
        ax.set_xticks([1,16]);ax.set_xlim(-.5,17.5);ax.set_xlabel('Rollout horizon (tics)');ax.set_ylim(0,1.02);ax.set_title(title,loc='left');ax.grid(axis='y',color='#e7eaec',lw=.6)
    axes[0].set_ylabel('Mean field CGE')
    handles,labels=axes[0].get_legend_handles_labels()
    fig.legend(handles,labels,ncol=4,loc='lower center',bbox_to_anchor=(.53,-.005),frameon=False,columnspacing=1.4)
    fig.subplots_adjust(left=.08,right=.99,top=.88,bottom=.29,wspace=.18);save(fig,'doom-decoder')


def hud_example():
    path=CACHE/'doom-dev6000-unet-h16.json';used.append(path)
    records=json.loads(path.read_text(encoding='utf-8'))
    r=min(records,key=lambda row:row['index'])
    assert r['index']==66 and r['episode']==6000
    conditions=[('previous','(a) Raw previous'),('predicted','(b) Stock prediction'),('reconstructed','(c) Stock GT reconstruction'),
        ('target','(d) Raw target'),('predicted-tuned','(e) Tuned prediction'),('reconstructed-tuned','(f) Tuned GT reconstruction')]
    fig,axes=plt.subplots(2,3,figsize=(6.9,2.15))
    for ax,(condition,title) in zip(axes.flat,conditions):
        p=ROOT/'cache/figure-assets'/('doom-first-'+condition+'.png');used.append(p)
        # Exact pixel crop of the native image; no retouching or neural editing.
        ax.imshow(Image.open(p).crop((0,206,106,239)),interpolation='nearest')
        ax.set_axis_off();ax.set_title(title,pad=3)
        values=r['readout'][condition]['values']
        ax.text(.5,-.08,f"Readout: ammo {values['ammo']}, health {values['health']}",transform=ax.transAxes,ha='center',va='top',fontsize=9)
    fig.subplots_adjust(left=.005,right=.995,bottom=.1,top=.88,hspace=.65,wspace=.12)
    save(fig,'doom-hud-example')


if __name__=='__main__':
    witness();budget();control();doom();hud_example()
    aliases={'actuator-witness':'Fig1','training-budget':'Fig2','goal-control':'FigS1','doom-decoder':'FigS2','doom-hud-example':'Fig3'}
    for name,alias in aliases.items():
        for ext in ['.pdf','.png','.svg']:
            source=OUT/(name+ext)
            if source.exists():(OUT/(alias+ext)).write_bytes(source.read_bytes())
    (OUT/'provenance.json').write_text(json.dumps({'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'inputs':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in used},
        'aliases':aliases,'size':'6.9in intended figure width; >=9pt native labels','scope':'Canonical numerical audit records and unedited simulator screenshots'},indent=2),encoding='utf-8')
    print(json.dumps({'figures':str(OUT),'inputs':len(used)}))
