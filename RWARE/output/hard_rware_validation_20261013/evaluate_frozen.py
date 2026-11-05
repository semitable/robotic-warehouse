"""Challenge calibration on one fixed E2 model; no updates or transfer."""
import json
import sys
from pathlib import Path
import torch

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from examples.diagnose_pgct_rollout_compatibility import load_model, model_hash
from examples.train_hard_rware_experiments import T, ENV_ID, PROFILES, profile_kwargs, constrained_runtime, preserved_hashes

torch.set_num_threads(1)
checkpoint=ROOT/'runs/sp_20261008_120638_983201/e2_pgct_s1/final.pt'
agents,meta=load_model(checkpoint)
before=model_hash(agents);source_hashes=preserved_hashes()
results={}
with constrained_runtime():
    for profile in PROFILES:
        kwargs={**meta['env_kwargs'],**profile_kwargs(profile), 'render_mode':'rgb_array'}
        result,video=T.evaluate(ENV_ID,kwargs,'semantic',agents,5,500,meta['config']['gamma'],
            1593,meta['config']['recurrent_hidden_dim'],torch.device('cpu'),seed=5000001,use_action_mask=True)
        results[profile]=result
        print(profile,{k:result[k] for k in ['eval_deliveries','eval_returns_home','eval_return','eval_deadlock_agent_rate']},flush=True)
    # Short RGB video integration test, separate from the 500-step task scores.
    _,video=T.evaluate(ENV_ID,kwargs,'semantic',agents,1,16,meta['config']['gamma'],1593,
        meta['config']['recurrent_hidden_dim'],torch.device('cpu'),seed=5000001,
        use_action_mask=True,collect_video=True)
    T.save_local_eval_video(video,Path(__file__).with_name('dock_video_smoke_v2.mp4'),10)
assert before==model_hash(agents)
assert source_hashes==preserved_hashes()
report=dict(checkpoint=str(checkpoint),policy='frozen E2 PGCT final, deterministic masked',episodes=5,horizon=500,
    seed_start=5000001,model_hash_unchanged=before,preserved_source_hashes=source_hashes,results=results,
    limitation='One pretrained policy and five reset seeds. Distribution-shift challenge only; not training difficulty, a performance ceiling, or evidence that PGCT helps.')
Path(__file__).with_name('frozen_difficulty_pilot_v2.json').write_text(json.dumps(report,indent=2),encoding='utf8')
