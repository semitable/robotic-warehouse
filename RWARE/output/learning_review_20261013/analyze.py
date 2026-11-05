"""Read-only snapshot analysis of saved training logs; writes only this directory."""
import json
from pathlib import Path
from datetime import datetime, timezone, timedelta
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
EXP = ROOT / 'runs/sp_20261008_120638_983201'
PAIRS = [(0,2),(1,3),(0,1),(0,3),(1,2),(2,3)]
LABELS = ['1-3 same','2-4 same','1-2 cross','1-4 cross','2-3 cross','3-4 cross']

def rows(path):
    if not path.exists(): return []
    data = path.read_text(encoding='utf8').splitlines()
    result = []
    for i,line in enumerate(data):
        try: result.append(json.loads(line))
        except json.JSONDecodeError:
            if i != len(data)-1: raise
    return result

def finite(value):
    if isinstance(value, dict): return {k:finite(v) for k,v in value.items()}
    if isinstance(value, list): return [finite(v) for v in value]
    if isinstance(value, (np.integer,np.floating)): value=value.item()
    if isinstance(value,float) and not np.isfinite(value): return None
    return value

manifest = json.loads((EXP/'experiment.json').read_text())
snapshot = dict(as_of=datetime.now(timezone(timedelta(hours=9))).isoformat(),
                experiment=str(EXP), initial_checkpoint=manifest['initial_checkpoint'],
                evaluation='20 fixed reset seeds starting at 5000001, deterministic masked policy, horizon 500',
                auc='Trapezoidal time average over observed evaluations only; no invented step-zero score',
                jobs={}, comparisons={}, curriculum=[], pretraining=[])
curves = {}
gates = {}
for job in manifest['jobs']:
    name=job['name']; d=EXP/name
    result=dict(status=job['status'], shape=job['grid_shape'], seed=job['seed'], target_steps=job['effective_timesteps'])
    snapshot['jobs'][name]=result
    if not d.exists(): continue
    logs=rows(d/'metrics.jsonl'); comm=rows(d/'communication.jsonl'); spatial=rows(d/'spatial.jsonl')
    evals={}
    train=[]
    for r in logs:
        m=r['metrics']; step=r['env_steps']
        if 'train/env_steps' in m: train.append(r)
        if 'eval/deliveries' in m: evals[step]={k.removeprefix('eval/'):v for k,v in m.items() if k.startswith('eval/')}
        if 'eval/final_deliveries' in m: evals[step]={k.removeprefix('eval/final_'):v for k,v in m.items() if k.startswith('eval/final_')}
    if not evals: continue
    steps=np.array(sorted(evals)); metrics=[evals[s] for s in steps]
    curves[name]=(steps,metrics)
    delivery=np.array([m['deliveries'] for m in metrics]); returned=np.array([m['returns_home'] for m in metrics])
    peak=int(np.argmax(delivery))
    result.update(last_training_step=max(r['env_steps'] for r in logs), last_eval_step=int(steps[-1]),
                  eval_count=len(steps), first_eval_step=int(steps[0]), first_eval_deliveries=float(delivery[0]),
                  last_eval=metrics[-1], last10_deliveries=float(delivery[-10:].mean()),
                  last10_temporal_std=float(delivery[-10:].std()),
                  peak_deliveries=float(delivery[peak]), peak_step=int(steps[peak]),
                  observed_time_average_deliveries=float(np.trapezoid(delivery,steps)/(steps[-1]-steps[0])),
                  agent_deliveries_per_episode=[metrics[-1][f'agent_{i+1}_delivery_count']/20 for i in range(4)],
                  agent_returns_per_episode=[metrics[-1][f'agent_{i+1}_return_count']/20 for i in range(4)],
                  training_actor_skipped_counts=[sum(r['metrics'].get(f'update/agent_{i}/actor_update_skipped',0)>0 for r in train) for i in range(4)],
                  train_rows=len(train), critic_accuracy_logged=any('critic_mc_mse' in m for m in metrics))
    result['team_deliveries_per_episode']=[sum(result['agent_deliveries_per_episode'][i] for i in range(4) if i%2==team) for team in range(2)]
    for threshold in (20,30,35,40,45):
        reached=steps[delivery>=threshold]
        result[f'first_observed_delivery_ge_{threshold}']=int(reached[0]) if len(reached) else None
    if train:
        for label,part in [('first100',train[:100]),('last100',train[-100:])]:
            result[label]={key:float(np.mean([r['metrics'][key] for r in part if key in r['metrics']])) for key in
                           ['update/value_loss','update/entropy','update/approx_kl','update/pgct_peer_loss','update/pgct_peer_weight','train/sps']}
        result['weighted_peer_loss_to_value_loss_last100'] = (.01*result['last100']['update/pgct_peer_loss']/(.5*result['last100']['update/value_loss'])) if job['mode']=='pgct' else 0
    sdict={r['env_steps']:r for r in spatial}
    common=[r for r in comm if r['env_steps'] in sdict]
    counts=np.zeros(6,int); candidates=np.zeros(6,int); invalid=0; window=[]
    gate_matrix=[]; gate_steps=[]
    for r in common:
        w=np.array(r['transfer_weights']); a=(w>0)|(w.T>0)
        geom=np.array(sdict[r['env_steps']]['update_end_geometric_candidates'],bool)
        invalid+=int(np.any(a & ~geom))
        active=np.array([a[i,j] for i,j in PAIRS]); cand=np.array([geom[i,j] for i,j in PAIRS])
        counts+=active; candidates+=cand; gate_steps.append(r['env_steps']);gate_matrix.append(active.astype(int))
    gates[name]=(np.array(gate_steps),np.array(gate_matrix))
    result['transfer']=dict(aligned_updates=len(common),outside_range_updates=invalid,
                            accepted_pair_updates=dict(zip(LABELS,counts.tolist())),
                            candidate_pair_updates=dict(zip(LABELS,candidates.tolist())),
                            acceptance_rates=dict(zip(LABELS,np.divide(counts,candidates,out=np.zeros(6,float),where=candidates>0).tolist())))
    for start in range(0,int(result['last_training_step']),1000000):
        subset=[r for r in common if start<r['env_steps']<=start+1000000]
        ac=np.zeros(6,int);ca=np.zeros(6,int)
        for r in subset:
            w=np.array(r['transfer_weights']); g=np.array(sdict[r['env_steps']]['update_end_geometric_candidates'])
            ac+=np.array([w[i,j]>0 or w[j,i]>0 for i,j in PAIRS]);ca+=np.array([g[i,j] for i,j in PAIRS])
        window.append(dict(start=start,end=min(start+1000000,result['last_training_step']),updates=len(subset),accepted=ac.tolist(),candidates=ca.tolist()))
    result['transfer']['windows']=window
    if common:
        last=common[-1]; ema=np.array(last['policy_distance_ema'])
        result['transfer']['final_ema']=dict(zip(LABELS,[float(ema[i,j]) for i,j in PAIRS]))
        result['transfer']['first_active_step']=next((r['env_steps'] for r in common if np.any(np.array(r['transfer_weights'])>0)),None)
    usable=[sdict[r['env_steps']] for r in common]
    total=sum(s['sampled_steps'] for s in usable)
    if total:
        occ=sum(np.array(s['occupancy_counts']) for s in usable)
        result['spatial']=dict(sampled_steps=total,mean_neighbors=float(sum(s['mean_physical_neighbors']*s['sampled_steps'] for s in usable)/total),
            left_occupancy=(sum(np.array(s['left_occupancy_fraction'])*s['sampled_steps'] for s in usable)/total).tolist(),
            phase_fractions=(sum(np.array(s['phase_counts']) for s in usable)/total).tolist())

for mapname in ('e1','e2'):
    p=curves[f'{mapname}_pgct_s1'];q=curves[f'{mapname}_ippo_s1']
    assert np.array_equal(p[0],q[0])
    x=p[0]; yp=np.array([m['deliveries'] for m in p[1]]);yq=np.array([m['deliveries'] for m in q[1]])
    diff=yp-yq
    comparison=dict(final_delivery_difference=float(diff[-1]),mean_time_weighted_delivery_difference=float(np.trapezoid(diff,x)/(x[-1]-x[0])),
        relative_time_average_difference=float(np.trapezoid(diff,x)/np.trapezoid(yq,x)),
        evaluations_pgct_above=int((diff>0).sum()),evaluations_ippo_above=int((diff<0).sum()),
        first_observed_difference_step=int(x[np.flatnonzero(diff)[0]]) if np.any(diff) else None,windows=[])
    for lo,hi in [(x[0],1000000),(1000000,2000000),(2000000,3000000),(3000000,x[-1])]:
        xx=np.r_[lo,x[(x>lo)&(x<hi)],hi]; pp=np.interp(xx,x,yp);qq=np.interp(xx,x,yq)
        comparison['windows'].append(dict(start=int(lo),end=int(hi),pgct=float(np.trapezoid(pp,xx)/(hi-lo)),ippo=float(np.trapezoid(qq,xx)/(hi-lo))))
    snapshot['comparisons'][mapname]=comparison

old=json.loads((ROOT/'output/gating_diagnosis/summary.json').read_text())
for stage in old['stages']:
    e=stage['final_eval']
    snapshot['curriculum'].append(dict(name=stage['stage'],steps=stage['updates']*1024,deliveries=e['eval_deliveries'],returns_home=e['eval_returns_home'],
        accepted_pairs={f"{p['agents'][0]}-{p['agents'][1]}":p['active_updates'] for p in stage['pairs']},
        agent_deliveries=[e[f'eval_agent_{i+1}_delivery_count']/20 for i in range(4)]))
for p in (ROOT/'runs/recurrent_ippo_consensus').glob('*/final_eval_metrics.json'):
    e=json.loads(p.read_text());snapshot['pretraining'].append(dict(name=p.parent.name,deliveries=e['eval_deliveries'],returns_home=e['eval_returns_home'],
        agent_counts=[e[f'eval_agent_{i+1}_delivery_count'] for i in range(4)],warning='Unmatched pretraining schedules/checkpoints; not a causal PGCT comparison'))

fig,axes=plt.subplots(2,3,figsize=(15,7.6),dpi=150,constrained_layout=True)
for col,mapname in enumerate(('e1','e2','e3')):
    for mode,color in [('pgct','#237a45'),('ippo','#5969bb')]:
        name=f'{mapname}_{mode}_s1'
        if name not in curves:continue
        x,ms=curves[name]
        for row,key in enumerate(('deliveries','returns_home')):
            y=[m[key] for m in ms];axes[row,col].plot(x/1e6,y,color=color,lw=1.6,label=mode.upper())
            axes[row,col].scatter([x[-1]/1e6],[y[-1]],color=color,s=15)
    axes[0,col].set_title(f"{mapname.upper()} {manifest['jobs'][col*2]['grid_shape']} / seed 1")
    for row in range(2):
        axes[row,col].set_xlabel('Additional environment steps (M)');axes[row,col].grid(alpha=.2);axes[row,col].legend()
        axes[row,col].set_ylabel(('Deliveries','Shelf returns')[row]+' / 500-step episode')
fig.suptitle('Matched Stage03 branches; 20 fixed evaluation seeds; E3 PGCT incomplete / IPPO not started')
fig.savefig(OUT/'learning.png');plt.close(fig)

fig,axes=plt.subplots(3,4,figsize=(15,9),dpi=140,constrained_layout=True)
for row,mapname in enumerate(('e1','e2','e3')):
    for mode,color in [('pgct','#237a45'),('ippo','#5969bb')]:
        name=f'{mapname}_{mode}_s1'
        if name not in curves:continue
        x,ms=curves[name]
        for i in range(4):
            axes[row,i].plot(x/1e6,[m[f'agent_{i+1}_delivery_count']/20 for m in ms],color=color,label=mode.upper(),lw=1.3)
            axes[row,i].set_title(f'{mapname.upper()} / Agent {i+1} / team {i%2}')
            axes[row,i].set_xlabel('Additional steps (M)');axes[row,i].set_ylabel('Deliveries / episode');axes[row,i].grid(alpha=.2)
axes[0,0].legend();fig.savefig(OUT/'agents.png');plt.close(fig)

fig,axes=plt.subplots(3,1,figsize=(13,8),dpi=150,constrained_layout=True)
for row,mapname in enumerate(('e1','e2','e3')):
    x,a=gates[f'{mapname}_pgct_s1'];ax=axes[row]
    edges=np.r_[x[0]-512,(x[1:]+x[:-1])/2,x[-1]+512]/1e6
    im=ax.pcolormesh(edges,np.arange(7)-.5,a.T,cmap='Greens',vmin=0,vmax=1)
    ax.invert_yaxis();ax.set_yticks(range(6),LABELS);ax.set_xlabel('Additional steps (M)');ax.set_title(f'{mapname.upper()} PGCT / actual transfer presence per pair-update')
fig.colorbar(im,ax=list(axes),label='Transfer present (0 / 1)',shrink=.7)
fig.savefig(OUT/'gates.png');plt.close(fig)

(OUT/'analysis.json').write_text(json.dumps(finite(snapshot),ensure_ascii=False,indent=2,allow_nan=False),encoding='utf8')
for name,r in snapshot['jobs'].items():
    print(name,{k:r.get(k) for k in ['status','last_training_step','last_eval_step','first_eval_deliveries','last10_deliveries','peak_deliveries','peak_step','observed_time_average_deliveries','agent_deliveries_per_episode','training_actor_skipped_counts','weighted_peer_loss_to_value_loss_last100']})
    if 'transfer' in r:print('TRANSFER',r['transfer']);print('SPATIAL',r.get('spatial'));print('LAST100',r.get('last100'))
print('COMPARISONS',snapshot['comparisons'])
print('CURRICULUM',snapshot['curriculum'])
