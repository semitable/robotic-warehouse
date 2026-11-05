"""Offline diagnosis of saved PGCT curriculum logs; does not train or edit runs."""
from pathlib import Path
import json
import math
import re
import statistics

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT / 'runs/pgct09700_physical_mixing_20260929_155338_079437'
OUT = Path(__file__).resolve().parent
PAIRS = [(0, 2), (1, 3), (0, 1), (0, 3), (1, 2), (2, 3)]
COLORS = ['#1764ab', '#20a4be', '#bbb4a5', '#e58b29', '#a56caa', '#b43c35']


def fields(line):
    return {k: float(v) for k, v in re.findall(r'(?<!\w)([\w]+)=(-?\d+(?:\.\d+)?)', line)}


def mean(values):
    return statistics.mean(values) if values else None


def load_stage(stage):
    comm = [json.loads(s) for s in (stage / 'communication.jsonl').read_text().splitlines()]
    logs = list(stage.glob('wandb/run-*/files/output.log'))
    if len(logs) != 1:
        raise ValueError(f'Expected exactly one output log for {stage.name}')
    train, evaluation = [], []
    for line in logs[0].read_text(encoding='utf-8', errors='replace').splitlines():
        if line.startswith('iter='):
            train.append(fields(line))
        elif line.startswith('eval iter='):
            evaluation.append(fields(line))
    assert len(comm) == len(train), (stage.name, len(comm), len(train))
    assert all(r['env_steps'] == t['steps'] for r, t in zip(comm, train))
    last = comm[-1]
    pair_summary = []
    for i, j in PAIRS:
        active = [k+1 for k, r in enumerate(comm) if r['transfer_weights'][i][j] > 0]
        qualified = [r for k, r in enumerate(comm, 1)
                     if k >= 100 and r['directed_probe_counts'][i][j] >= 8]
        pair_summary.append(dict(agents=[i+1, j+1], same_team=i%2 == j%2,
                                 final_distance_ema=last['policy_distance_ema'][i][j],
                                 active_updates=len(active), first_active=active[0] if active else None,
                                 last_active=active[-1] if active else None,
                                 below_threshold_qualified_updates=sum(r['policy_distance_ema'][i][j]<.12 for r in qualified)))
    summary = dict(stage=stage.name, updates=len(comm), pairs=pair_summary,
                   first_5_eval_deliveries=mean([r['deliveries'] for r in evaluation[:5]]),
                   last_5_eval_deliveries=mean([r['deliveries'] for r in evaluation[-5:]]),
                   first_100_train_entropy=mean([r['ent'] for r in train[:100]]),
                   last_100_train_entropy=mean([r['ent'] for r in train[-100:]]),
                   first_eval=evaluation[0], last_periodic_eval=evaluation[-1],
                   final_eval=json.loads((stage/'final_eval_metrics.json').read_text()),
                   final_probe_entropy_nats=[math.log(5)-v for v in last['policy_information']])
    return dict(comm=comm, train=train, evaluation=evaluation, summary=summary)


def distance_panel(ax, data):
    x = np.arange(1, len(data['comm'])+1)
    for (i,j), color in zip(PAIRS, COLORS):
        ax.plot(x, [r['policy_distance_ema'][i][j] for r in data['comm']],
                color=color, lw=1.5 if i%2==j%2 else .9,
                label=f'{i+1}-{j+1} '+('same team' if i%2==j%2 else 'cross team'))
    ax.axhline(.12, color='black', ls='--', lw=1, label='Gate threshold 0.12')
    ax.set_ylim(0, .7)
    ax.set_ylabel('JS distance EMA (nats)')


def gate_panel(ax, data):
    x = np.arange(1, len(data['comm'])+1)
    for same, color, label in [(True, COLORS[0], 'Same team'), (False, COLORS[-1], 'Cross team')]:
        pairs = [(i,j) for i,j in PAIRS if (i%2==j%2)==same]
        eligibility, active = [], []
        for k,r in enumerate(data['comm'], 1):
            eligible = sum(k>=100 and r['directed_probe_counts'][i][j]>=8
                           and r['policy_distance_ema'][i][j]<.12 for i,j in pairs)/len(pairs)
            eligibility.append(eligible)
            active.append(sum(r['transfer_weights'][i][j]>0 for i,j in pairs)/len(pairs))
        ax.plot(x, eligibility, color=color, alpha=.25, lw=.7)
        smoothed = np.convolve(active, np.ones(100)/100, mode='valid')
        ax.plot(x[99:], smoothed, color=color, lw=1.7, label=label)
    ax.set_ylim(-.03, 1.03)
    ax.set_ylabel('Accepted pair fraction')
    ax.set_title('Gate: faint = eligible; solid = active, 100-update mean', fontsize=10)


def delivery_panel(ax, data):
    train=data['train']; ev=data['evaluation']
    y=np.convolve([r['deliveries'] for r in train], np.ones(100)/100, mode='valid')
    ax.plot([r['iter'] for r in train][99:], y, color='#b2b7bf', lw=1.2, label='Train, 100-update mean')
    ax.plot([r['iter'] for r in ev], [r['deliveries'] for r in ev], color='#267347', marker='.', lw=1.3, label='Evaluation, 20 episodes')
    ax.set_ylabel('Deliveries per 500-step episode')


def entropy_panel(ax, data):
    comm=data['comm']; x=np.arange(1,len(comm)+1)
    for agent in range(4):
        y=[math.log(5)-r['policy_information'][agent]
           if any(v>0 for row in r['directed_probe_counts'] for v in row) else float('nan') for r in comm]
        ax.plot(x,y,lw=1.2,label=f'Agent {agent+1}')
    ax.set_ylabel('Mean probe action entropy (nats)')
    ax.set_title('Shared fixed probes, zero hidden state, L=1',fontsize=10)


def decorate(ax):
    ax.grid(alpha=.18)
    ax.set_xlabel('PPO update within stage')
    ax.spines[['top','right']].set_visible(False)


def main():
    plt.rcParams.update({'font.size':10, 'figure.dpi':140})
    stages=[load_stage(p) for p in sorted(RUN.glob('stage*')) if p.is_dir()]
    assert len(stages)==4
    summary=dict(source_run=str(RUN), threshold=.12, same_team_pairs=[[1,3],[2,4]],
                 caveats=['Distances are stored EMA values, updated only when physically neighboring.',
                          'Probe entropy is derived as log(5)-KL(policy||uniform), not online entropy.',
                          'Evaluation seeds vary by checkpoint; curves are descriptive, not paired causal tests.',
                          'Shelf mixing changes across stages; cross-stage delivery differences are confounded.',
                          'No matched PGCT-off curriculum was found in this run.'],
                 stages=[s['summary'] for s in stages])
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False),encoding='utf-8')
    fig,axes=plt.subplots(3,4,figsize=(16,10),sharex='col')
    for c,d in enumerate(stages):
        distance_panel(axes[0,c],d)
        axes[0,c].set_title(f'Stage {c+1}: home ratio {[1,.875,.75,.5][c]}')
        gate_panel(axes[1,c],d)
        axes[1,c].set_title('Gate eligibility / active transfer',fontsize=10)
        delivery_panel(axes[2,c],d)
        for r in range(3): decorate(axes[r,c])
    axes[0,0].legend(fontsize=7,ncol=2)
    axes[1,0].legend(fontsize=8)
    axes[2,0].legend(fontsize=7)
    fig.suptitle('PGCT curriculum: behavior distance, gates, and learning performance',fontsize=16)
    fig.tight_layout(rect=(0,.04,1,.96))
    fig.text(.5,.018,'Faint gate traces = policy eligibility; solid = physical accepted transfer (100-update mean). Stage environments differ.',ha='center',fontsize=10)
    fig.savefig(OUT/'curriculum.png')
    plt.close(fig)
    fig,axes=plt.subplots(2,2,figsize=(13,8))
    d=stages[-1]
    distance_panel(axes[0,0],d); gate_panel(axes[0,1],d)
    entropy_panel(axes[1,0],d); delivery_panel(axes[1,1],d)
    for ax in axes.flat:
        decorate(ax); ax.legend(fontsize=8,loc='best')
    fig.suptitle('Stage 04: behavior divergence and gates alongside entropy and deliveries',fontsize=15)
    fig.tight_layout(rect=(0,.03,1,.95))
    fig.text(.5,.015,'Gate closure is observed; role specialization and transfer usefulness are separate hypotheses.',ha='center',fontsize=10)
    fig.savefig(OUT/'stage04.png'); plt.close(fig)
    for s in summary['stages']:
        print(json.dumps({k:v for k,v in s.items() if k in ['stage','updates','first_5_eval_deliveries','last_5_eval_deliveries','first_100_train_entropy','last_100_train_entropy','final_probe_entropy_nats']}))
    print('Saved analysis to',OUT)


if __name__=='__main__':
    main()
