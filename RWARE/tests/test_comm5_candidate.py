"""T178: b6_corner_indiv_comm5_pen005 differs from b6_corner_indiv_pen005 only in the contact range, and the
range never reaches observations, rewards or dynamics (it only defines who is in contact). No training."""
import json
from pathlib import Path
from types import SimpleNamespace
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from latent_team import common as C

OLD, NEW = "b6_corner_indiv_pen005", "b6_corner_indiv_comm5_pen005"


def jobs(tmp_path, variant, modes):
    from latent_team.train_from_scratch import plan
    if not (C.REFERENCE_EXPERIMENT / "experiment.json").exists():
        pytest.skip("Local reference configuration missing")
    args = SimpleNamespace(reference_experiment=C.REFERENCE_EXPERIMENT, output_dir=tmp_path / variant, maps=["e1"],
                           seeds=[1, 2], modes=modes, timesteps=4096, home_ratio=.5, learning_rate=2e-4,
                           entropy_coef=.05, save_interval=250, track=False, save_video=False, smoke_test=False,
                           eval_seed=5000001, stage_label="test", wandb_project="test", wandb_mode="offline",
                           probe_bank="canonical", team_reward="individual", env_variant=variant,
                           reward_probe_ac=True, ac_gate="v2")
    return {(j["seed"], j["mode"]): j for j in plan(args)[1]["jobs"]}


def test_comm5_changes_only_the_contact_range(tmp_path):
    modes = ["ippo", "pgct_max", "allk_max", "oracle_max"]
    old, new = jobs(tmp_path, OLD, modes), jobs(tmp_path, NEW, modes)
    assert old.keys() == new.keys()
    for key in old:
        a, b = (json.loads(x[key]["config"]["env_kwargs_json"]) for x in (old, new))
        assert a.pop("communication_range") == 3 and b.pop("communication_range") == 5
        assert a == b  # goals (sides swap on even seeds), individual reward, penalties, agents
        assert new[key]["name"].startswith(NEW + "_acv2_")
        rest = lambda job: {k: v.replace(NEW, OLD) if isinstance(v, str) else v  # names, paths, tags
                            for k, v in job["config"].items() if k != "env_kwargs_json"}
        assert rest(old[key]) == rest(new[key])
    goals = {seed: json.loads(new[(seed, "ippo")]["config"]["env_kwargs_json"])["team_goal_positions"]
             for seed in (1, 2)}
    assert goals[1] == goals[2][::-1]


def test_ippo_rollout_is_identical_under_both_ranges(tmp_path):
    from examples import train_recurrent_ippo_consensus as T
    old, new = jobs(tmp_path, OLD, ["ippo"]), jobs(tmp_path, NEW, ["ippo"])
    rng = np.random.default_rng(0)
    actions = rng.integers(0, 5, size=(300, 6))
    traces = []
    for job in (old[(1, "ippo")], new[(1, "ippo")]):
        cfg = job["config"]
        kwargs = json.loads(cfg["env_kwargs_json"])
        kwargs.update(n_agents=cfg["agent_count"], sensor_range=cfg["sensor_range"])
        with T.make_env(cfg["env_id"], kwargs, cfg["observation_format"]) as env:
            obs, _ = env.reset(seed=123)
            trace = [np.concatenate([np.ravel(o) for o in obs])]
            contacts = 0
            for act in actions:
                obs, rew, term, trunc, _ = env.step(act.tolist())
                trace += [np.concatenate([np.ravel(o) for o in obs]), np.asarray(rew, dtype=np.float64)]
                u = env.unwrapped
                contacts += sum(u._agents_within_comm_range(i, j) for i in range(6) for j in range(i + 1, 6))
            traces.append((trace, contacts))
    (t3, c3), (t5, c5) = traces
    assert all(np.array_equal(x, y) for x, y in zip(t3, t5))
    assert c5 > c3  # only who is in contact changes


def test_harder_variants_change_one_thing_each(tmp_path):
    base = jobs(tmp_path, NEW, ["ippo"])[(1, "ippo")]
    wide = jobs(tmp_path, "b6_wide_corner_indiv_comm5_pen005", ["ippo"])[(1, "ippo")]
    req2 = jobs(tmp_path, "b6_corner_indiv_comm5_req2_pen005", ["ippo"])[(1, "ippo")]
    env = lambda job: json.loads(job["config"]["env_kwargs_json"])
    b, w, r = env(base), env(wide), env(req2)
    assert w.pop("shelf_columns") == 5 and b.get("shelf_columns", 3) == 3
    assert w.pop("team_goal_positions") == [[[0, 10]], [[15, 10]]]
    assert {k: v for k, v in b.items() if k not in ("shelf_columns", "team_goal_positions")} == w
    assert (r["request_queue_size"], r["request_queue_size_per_team"]) == (4, [2, 2])
    assert {k: v for k, v in r.items() if not k.startswith("request_queue")} == \
        {k: v for k, v in b.items() if not k.startswith("request_queue")}
    # the trainer would otherwise raise the requests back to the 3 agents per team
    assert req2["config"]["auto_scale_request_queue"] is False and base["config"]["auto_scale_request_queue"]
