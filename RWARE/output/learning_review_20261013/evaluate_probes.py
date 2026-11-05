"""Posthoc canonical probe responses, matched PGCT bank for each map."""
import sys,json
from pathlib import Path
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[2];OUT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'examples'))
from diagnose_pgct_rollout_compatibility import load_model,model_hash
from pgct_probe_diagnostics import common_probe_response,PAIRS,PAIR_LABELS

torch.set_num_threads(1)
results={}
fig,axes=plt.subplots(2,2,figsize=(12,8),dpi=150,constrained_layout=True)
for row,mapname in enumerate(['e1','e2']):
    bank_path=ROOT/'runs/sp_20261008_120638_983201'/f'{mapname}_pgct_s1'/'canonical_probes.pt'
    bank=torch.load(bank_path,map_location='cpu',weights_only=False)['sequences']
    for col,mode in enumerate(['pgct','ippo']):
        name=f'{mapname}_{mode}_s1'
        agents,meta=load_model(bank_path.parent.parent/name/'final.pt');before=model_hash(agents)
        r=common_probe_response(agents,bank,torch.device('cpu'));assert before==model_hash(agents)
        r.update(checkpoint=meta['checkpoint'],bank=str(bank_path),unchanged_model_hash=before)
        results[name]=r
        values=np.array([[r['mean_js'][i][j],r['unique_mean_js'][i][j]] for i,j in PAIRS])
        ax=axes[row,col];im=ax.imshow(values,cmap='magma_r',vmin=0,vmax=np.log(2),aspect='auto')
        ax.set_yticks(range(6),PAIR_LABELS);ax.set_xticks([0,1],[f'{len(bank)} weighted probes',f"{r['unique_probe_count']} unique inputs"])
        ax.set_title(f'{mapname.upper()} {mode.upper()} final / zero hidden / no action mask')
        for i in range(6):
            for j in range(2):ax.text(j,i,f'{values[i,j]:.3f}',ha='center',va='center',color='white' if values[i,j]>.34 else 'black')
        print(name, 'JS48',values[:,0].round(4).tolist(),'JS_unique',values[:,1].round(4).tolist(),flush=True)
fig.colorbar(im,ax=list(axes.flat),label='Natural-log JS; smaller = more similar',shrink=.7)
fig.savefig(OUT/'probes.png');plt.close(fig)
(OUT/'probe_responses.json').write_text(json.dumps(results,indent=2),encoding='utf8')
