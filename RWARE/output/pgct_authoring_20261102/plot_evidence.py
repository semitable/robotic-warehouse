from pathlib import Path
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[2]
D=json.loads((Path(__file__).parent/'evidence.json').read_text(encoding='utf-8'))
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.spines.top':False,'axes.spines.right':False,
                     'pdf.fonttype':42,'svg.fonttype':'none','axes.grid':True,'grid.alpha':.2})
blue='#3634B8'; orange='#B65B20'
fig,axs=plt.subplots(1,2,figsize=(10,3.25),constrained_layout=True)
b=D['discovery']['bins']; x=[r['step']/1e6 for r in b]
for k,c,label in [('same',blue,'Same team'),('cross',orange,'Different team')]:
    axs[0].plot(x,[r[k] for r in b],color=c,lw=2,label=label)
axs[0].axhline(.5,color='#606060',ls='--',lw=1,label='Approval threshold')
axs[0].set(xlabel='Training steps (million)',ylabel='Mean response distance (EMA)',ylim=(0,1.05))
axs[0].legend(fontsize=8,loc='upper right')
axs[1].plot(x,[r['mature'] for r in b],color=blue,lw=2)
axs[1].set(xlabel='Training steps (million)',ylabel='Reward predictors passing maturity',ylim=(0,6.3),yticks=[0,2,4,6])
for ext in ['pdf','png','svg']: fig.savefig(ROOT/f'paper/figures/discovery.{ext}',dpi=220)
plt.close(fig)
for mode in ['discovery','learning']:
    fig,ax=plt.subplots(figsize=(7.3,4.65))
    fig.subplots_adjust(left=.13,right=.98,top=.98,bottom=.25)
    ax.grid(False)
    if mode=='discovery':
        for k,c,label in [('same',blue,'Same team'),('cross',orange,'Different team')]:
            ax.plot(x,[r[k] for r in b],color=c,lw=2,label=label)
        ax.axhline(.5,color='#656565',lw=1.5,label='Approval threshold')
        ax.set(ylim=(0,1.05),ylabel='Response distance (EMA)')
    else:
        v=D['ippo']['curve']; ax.plot([r['step']/1e6 for r in v],[r['deliveries'] for r in v],color=blue,lw=2,label='Independent PPO (seed 1)')
        ax.set(ylim=(0,30),ylabel='Deliveries / episode')
    ax.set(xlabel='Training steps (million)',xlim=(0,5))
    ax.legend(loc='upper center',bbox_to_anchor=(.5,-.18),ncol=3 if mode=='discovery' else 1,fontsize=10,frameon=False)
    for ext in ['pdf','png']: fig.savefig(ROOT/f'presentation/figures/{mode}_chart.{ext}',dpi=210)
    plt.close(fig)
fig,ax=plt.subplots(figsize=(6.3,3.35),constrained_layout=True)
v=D['ippo']['curve']; ax.plot([r['step']/1e6 for r in v],[r['deliveries'] for r in v],color=blue,lw=2,label='Independent PPO (seed 1)')
ax.set(xlabel='Training steps (million)',ylabel='Deliveries per 500-step episode',xlim=(0,5),ylim=(0,30))
ax.legend(fontsize=9,loc='upper left')
for ext in ['pdf','png','svg']: fig.savefig(ROOT/f'paper/figures/ippo_delivery.{ext}',dpi=220)
plt.close(fig)
