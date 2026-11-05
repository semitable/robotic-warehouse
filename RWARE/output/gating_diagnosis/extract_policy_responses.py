"""Forward-only actor comparison on the saved stage04 canonical bank."""
from pathlib import Path
from types import SimpleNamespace
import json
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from examples import train_recurrent_ippo_consensus as T
from rware.utils.semantic_observation import semantic_observation_spec

RUN = ROOT/'runs/pgct09700_physical_mixing_20260929_155338_079437'
OUT = Path(__file__).resolve().parent
ACTIONS = ['NOOP', 'FORWARD', 'LEFT', 'RIGHT', 'TOGGLE_LOAD']
PAIRS = [(0,2),(1,3),(0,1),(0,3),(1,2),(2,3)]


@torch.inference_mode()
def evaluate_checkpoint(path, sequences):
    checkpoint = T._load_checkpoint(str(path), torch.device('cpu'))
    cfg = checkpoint['config']
    spec = semantic_observation_spec(SimpleNamespace(sensor_range=cfg['sensor_range']))
    assert sequences.shape == (48, 1, spec.obs_dim)
    policies, agents = [], []
    for state in checkpoint['agents']:
        net = T.RecurrentActorCritic(
            spec.obs_dim, len(ACTIONS), cfg['mlp_hidden_dim'], cfg['recurrent_hidden_dim'],
            cfg['obs_encoder'], spec.spatial_channels, spec.spatial_size, spec.ego_dim)
        net.load_state_dict(state)
        net.eval()
        h, _ = net.initial_hidden(48, torch.device('cpu'))
        logits, _ = net._actor_step(sequences[:,0],h)
        policies.append(logits.softmax(-1))
        assert all(torch.equal(v, state[k]) for k,v in net.state_dict().items())
        agents.append(net)
    p=torch.stack(policies).clamp_min(1e-8)
    entropy=-(p*p.log()).sum(-1)
    pairs=[]
    for i,j in PAIRS:
        midpoint=(p[i]+p[j])/2
        js=((p[i]*(p[i].log()-midpoint.log())).sum(-1)
            +(p[j]*(p[j].log()-midpoint.log())).sum(-1))/2
        pairs.append(dict(agents=[i+1,j+1],same_team=i%2==j%2,
                          mean_js=float(js.mean()),
                          argmax_disagreement_fraction=float((p[i].argmax(-1)!=p[j].argmax(-1)).float().mean()),
                          js_per_probe=js.tolist(),largest_difference_probe_indices=js.topk(5).indices.tolist()))
    computed,_,_=T.policy_distance_similarity_matrices(agents,sequences,torch.device('cpu'),.25,True)
    assert all(abs(computed[i,j]-pair['mean_js'])<1e-6 for (i,j),pair in zip(PAIRS,pairs))
    return dict(checkpoint=str(path),iteration=checkpoint['iteration'],env_steps=checkpoint['env_steps'],
                probabilities=p.tolist(),entropy_per_probe=entropy.tolist(),
                mean_probe_entropy=entropy.mean(1).tolist(),
                mean_action_probabilities=p.mean(1).tolist(),pairs=pairs)


def main():
    torch.set_num_threads(1)
    torch.manual_seed(0)
    stage04=RUN/'stage04_4ag_physical_home0.500'
    bank=torch.load(stage04/'canonical_probes.pt',map_location='cpu',weights_only=False)
    sequences=bank['sequences']
    unique_inputs, input_ids=torch.unique(sequences.reshape(len(sequences),-1),dim=0,return_inverse=True)
    results={}
    for stage in sorted(RUN.glob('stage*')):
        results[stage.name]=evaluate_checkpoint(stage/'final.pt',sequences)
    final=results[stage04.name]
    result=dict(probe_bank=str(stage04/'canonical_probes.pt'),probe_shape=list(sequences.shape),
                unique_input_count=len(unique_inputs),input_equivalence_ids=input_ids.tolist(),
                bank_metadata=bank['metadata'],actions=ACTIONS,action_mask_applied=False,
                explanation='All checkpoints use exactly the saved stage04 bank and zero actor hidden state. '
                'Unmasked probabilities match the training distance computation. Means across probes are '
                'descriptive only: distance is mean JS per probe, not JS between averaged probabilities.',
                checkpoints=results)
    (OUT/'policy_responses.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    argmax=np.asarray(final['probabilities']).argmax(-1)
    colors=['#aaaaaa','#3c8d59','#3d6bbb','#e59734','#a457a9']
    fig,ax=plt.subplots(figsize=(15,3.7),dpi=150)
    m=ax.imshow(argmax,cmap=ListedColormap(colors),norm=BoundaryNorm(np.arange(-.5,5.5,1),5),aspect='auto')
    ax.set_yticks(range(4),['Agent 1 / team 0','Agent 2 / team 1','Agent 3 / team 0','Agent 4 / team 1'])
    ax.set_xticks(range(0,48,2)); ax.set_xlabel('Saved canonical probe index (0-based)')
    ax.set_title('Stage04 final: preferred action on identical canonical inputs (zero hidden state)')
    fig.colorbar(m,ax=ax,ticks=range(5),pad=.015).ax.set_yticklabels(ACTIONS)
    fig.tight_layout();fig.savefig(OUT/'preferred_actions.png');plt.close(fig)
    fig,axes=plt.subplots(2,2,figsize=(12,7),dpi=150)
    for row,pair in enumerate(final['pairs'][:2]):
        i,j=[v-1 for v in pair['agents']]
        selected,seen=[],set()
        for k in np.argsort(pair['js_per_probe'])[::-1]:
            input_id=int(input_ids[k])
            if input_id not in seen:
                selected.append(int(k));seen.add(input_id)
            if len(selected)==2:
                break
        for col,k in enumerate(selected):
            ax=axes[row,col];x=np.arange(5)
            ax.bar(x-.18,final['probabilities'][i][k],width=.36,label=f'Agent {i+1}')
            ax.bar(x+.18,final['probabilities'][j][k],width=.36,label=f'Agent {j+1}')
            ax.set_xticks(x,ACTIONS,fontsize=8);ax.set_ylim(0,1.05)
            ax.set_title(f'Same team {i+1}-{j+1}, probe {k}, JS={pair["js_per_probe"][k]:.3f}')
            ax.set_ylabel('Action probability');ax.legend();ax.spines[['top','right']].set_visible(False)
    fig.suptitle('Largest same-team disagreements: shared stage04 probes, unmasked policies')
    fig.tight_layout();fig.savefig(OUT/'same_team_probability_examples.png');plt.close(fig)
    for stage,r in results.items():
        print(json.dumps(dict(stage=stage,mean_probe_entropy=r['mean_probe_entropy'],
                             same_team=[{k:v for k,v in p.items() if k not in ['js_per_probe','largest_difference_probe_indices']} for p in r['pairs'][:2]])))
    print('Final action probability means:',json.dumps(final['mean_action_probabilities']))
    print('Unique canonical inputs:',len(unique_inputs),'of',len(sequences))


if __name__=='__main__':
    main()
