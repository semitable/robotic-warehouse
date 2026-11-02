"""Read existing PGCT artifacts only; never imports a trainer or launches an environment."""
from pathlib import Path
import json, csv, hashlib, sys
sys.stdout.reconfigure(encoding='utf-8')
ROOT=Path(__file__).resolve().parents[2]; OUT=Path(__file__).parent
def rows(path):
    with path.open(encoding='utf-8') as f:
        for line in f:
            try: yield json.loads(line)
            except json.JSONDecodeError: continue  # active logs may have an incomplete final line
gate=ROOT/'runs/acv2_s1_20261029_084340/b6_corner_indiv_pen005_acv2_pgct_s1'
arcs=list(rows(gate/'transfer.jsonl'))
metrics=list(rows(gate/'metrics.jsonl'))
curve=[]
for lo in range(0,5_000_001,100000):
    block=[r for r in metrics if lo < r['env_steps'] <= lo+100000]
    def avg(key):
        a=[r['metrics'][key] for r in block if key in r['metrics']]
        return sum(a)/len(a) if a else None
    curve.append(dict(step=lo+50000,same=avg('update/ac_v2_audit_distance_same'),
                      cross=avg('update/ac_v2_audit_distance_cross'),
                      mature=avg('update/ac_v2_mature_agents')))
curve=[x for x in curve if x['same'] is not None and x['cross'] is not None]
same=sum(r['same_team_arcs'] for r in arcs); cross=sum(r['cross_team_arcs'] for r in arcs)
cross_first=next((r['env_steps'] for r in arcs if r['cross_team_arcs']),None)
ippo=ROOT/'runs/ac_20261023_185556/b6_corner_indiv_pen005_ac_ippo_s1'
ipos=list(rows(ippo/'metrics.jsonl'))
keys=sorted(set(k for r in ipos for k in r['metrics'] if 'eval' in k and ('deliver' in k or 'agent' in k)))
print('Evaluation keys:',keys[:35])
delivery_key=next((k for k in ['eval/deliveries','eval/delivery_count'] if any(k in r['metrics'] for r in ipos)),None)
delivery=[dict(step=r['env_steps'],deliveries=r['metrics'][delivery_key]) for r in ipos if delivery_key in r['metrics']]
plans=[]
for d in sorted((ROOT/'runs').glob('maxgae_br1M_*')):
    m=json.loads((d/'experiment.json').read_text(encoding='utf-8'))
    jobs=[]
    for j in m['jobs']:
        folder=d/j['name']; status=folder/'job_status.json'
        tail=list(rows(folder/'metrics.jsonl')) if (folder/'metrics.jsonl').exists() else []
        jobs.append(dict(mode=j['mode'],seed=j['seed'],status=json.loads(status.read_text()).get('status') if status.exists() else 'planned',
                         last_logged_steps=tail[-1]['env_steps'] if tail else 0))
    plans.append(dict(path=str(d.relative_to(ROOT)),jobs=jobs))
cfg=json.loads((ROOT/'runs/acv2_s1_20261029_084340/experiment.json').read_text())['jobs'][0]['config']
sources=[gate/'transfer.jsonl',gate/'metrics.jsonl',ippo/'metrics.jsonl',ROOT/'latent_team/max_gae.py',ROOT/'latent_team/reward_probe_ac_v2.py',ROOT/'latent_team/train_from_scratch.py']
data=dict(discovery=dict(same_arc_updates=same,cross_arc_updates=cross,precision=same/(same+cross),first_cross_step=cross_first,
                         last_step=arcs[-1]['env_steps'],bins=curve,source=str(gate.relative_to(ROOT)),
                         caution='Component diagnostic with update-end contacts; not final integrated PGCT performance.'),
          ippo=dict(curve=delivery,source=str(ippo.relative_to(ROOT)),evaluation_key=delivery_key),branch_plans=plans,
          training_config={k:v for k,v in cfg.items() if any(s in k for s in ['gamma','gae_lambda','hidden','sequence','rollout','clip','epoch','minibatch','entropy','learning_rate','eval_'])},
          sources={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources})
(OUT/'evidence.json').write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding='utf-8')
for name,records in [('discovery',curve),('ippo_delivery',delivery)]:
    with (ROOT/'paper/figures'/f'{name}.csv').open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=records[0].keys()); w.writeheader(); w.writerows(records)
print(json.dumps({k:v for k,v in data['discovery'].items() if k!='bins'},ensure_ascii=False))
print('IPPO evaluations',len(delivery),'last',delivery[-1] if delivery else None)
print('Training config',data['training_config'])
print('Branch snapshots',plans)
