"""Frozen on-policy critic evaluation, separate from training and deterministic task metrics."""
import json
import sys
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[2]
OUT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'examples'))
import train_recurrent_ippo_consensus as T
from diagnose_pgct_rollout_compatibility import load_model,model_hash
from pgct_probe_diagnostics import finite_horizon_returns

@torch.inference_mode()
def evaluate(name):
    agents,meta=load_model(ROOT/'runs/sp_20261008_120638_983201'/name/'final.pt')
    before=model_hash(agents);cfg=SimpleNamespace(**meta['config']);device=torch.device('cpu')
    episodes=[]
    for episode in range(10):
        rng=np.random.default_rng(5100001+episode)
        with T.make_env(cfg.env_id,meta['env_kwargs'],cfg.observation_format) as env:
            obs,_=env.reset(seed=5100001+episode)
            ah=torch.zeros((4,cfg.recurrent_hidden_dim));ch=torch.zeros_like(ah)
            vv=[];rr=[];terminated=False;truncated=False
            for step in range(500):
                x=torch.as_tensor(T.flatten_multi_agent_obs(env,obs),dtype=torch.float32)
                mask=torch.as_tensor(T.local_rware_action_masks(env,4,5,enabled=cfg.action_mask))
                actions=[];v=[]
                for i,net in enumerate(agents):
                    p=net.action_probabilities(x[i],ah[i],mask[i]).numpy()[0].astype(float);p/=p.sum()
                    actions.append(int(rng.choice(5,p=p)))
                    _,_,val,next_ah,next_ch=net.act(x[i],ah[i],ch[i],deterministic=True,action_mask=mask[i])
                    ah[i]=next_ah.squeeze(0);ch[i]=next_ch.squeeze(0);v.append(val.item())
                obs,rewards,done,trunc,_=env.step(actions);rr.append(rewards);vv.append(v)
                terminated=bool(np.any(done));truncated=bool(np.any(trunc))
                if terminated or truncated:break
            r=np.array(rr);v=np.array(vv);mc=finite_horizon_returns(r,cfg.gamma)
            x=torch.as_tensor(T.flatten_multi_agent_obs(env,obs),dtype=torch.float32)
            boundary=np.zeros(4) if terminated else np.array([net._critic_step(x[i].unsqueeze(0),ch[i].unsqueeze(0))[0].item() for i,net in enumerate(agents)])
            nv=np.concatenate([v[1:],boundary[None]],axis=0)
            error=(v-mc)**2;td=r+cfg.gamma*nv-v
            prefix=max(1,len(v)-300)
            episodes.append(dict(seed=5100001+episode,length=len(v),terminated=terminated,truncated=truncated,
                mc_mse=error.mean(0).tolist(),mc_mae=np.abs(v-mc).mean(0).tolist(),td_abs=np.abs(td).mean(0).tolist(),
                mc_mse_early_prefix=error[:prefix].mean(0).tolist(),early_prefix_steps=prefix,
                mean_value=v.mean(0).tolist(),mean_finite_return=mc.mean(0).tolist(),total_reward=r.sum(0).tolist()))
    after=model_hash(agents);assert before==after
    keys=['mc_mse','mc_mae','td_abs','mc_mse_early_prefix','mean_value','mean_finite_return','total_reward']
    means={key:np.mean([e[key] for e in episodes],axis=0).tolist() for key in keys}
    result=dict(checkpoint=meta['checkpoint'],env_steps=meta['env_steps'],unchanged_model_hash=before,
                policy='stochastic masked; independent local NumPy RNG per episode',episodes=episodes,agent_means=means,
                aggregate={key:float(np.mean(value)) for key,value in means.items()},
                target='Finite-horizon discounted empirical return, zero tail at step 500',
                early_prefix='First 200 steps of 500-step episodes, >=300 remaining steps; continuing tail bias attenuated, not removed',
                td='One-step absolute TD error; bootstrap on truncation, zero on termination',
                limitation='Each model visits its own states. This does not isolate critic quality on a shared on-policy state distribution; 10 evaluation episodes and one training seed are exploratory.')
    (OUT/f'{name}_critic.json').write_text(json.dumps(result,indent=2),encoding='utf8')
    print(name,json.dumps(result['aggregate']), 'agents',json.dumps(means),flush=True)
    return result

if __name__=='__main__':
    torch.set_num_threads(1)
    results={}
    for name in ['e1_pgct_s1','e1_ippo_s1','e2_pgct_s1','e2_ippo_s1']:
        results[name]=evaluate(name)
    (OUT/'critic_summary.json').write_text(json.dumps(results,indent=2),encoding='utf8')
