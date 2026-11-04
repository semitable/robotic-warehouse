"""D108: gate v2 action_max distance. The pooled distance is diluted by an event kind both teams are paid for
(shelf returns); the maximum over public actions both agents experienced is not. No training runs."""
from pathlib import Path
from types import SimpleNamespace
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from latent_team import reward_probe_ac_v2 as V2

R = 2
SIZE = V2.view_size(R)
FORWARD, TOGGLE = 1, 4


def options(**kwargs):
    return V2.ACV2Config(**dict(dict(buffer_size=4000, event_buffer_size=64, batch_size=64, train_steps=300,
                                      min_events=5, min_rounds=1, interval=1, mirror=False, probes_per_agent=16),
                                 **kwargs))


def views(seed, n=1):
    return np.random.default_rng(seed).integers(0, 2, size=(n, SIZE)).astype(np.uint8)


DELIVERY = {0: (views(1)[0], FORWARD, views(2)[0]), 1: (views(3)[0], FORWARD, views(4)[0])}  # team goals
RETURN = (views(5)[0], TOGGLE, views(6)[0])  # paid to every team (same public scene)


def teach(ctrl, teams):
    """Agent i is paid for delivering at goal teams[i] and for every return; it also sees the other goal unpaid.
    Recent own events alternate delivery / return, so half of every agent's probes are shared returns."""
    background = views(9, 200)
    for rep in range(30):
        for i, team in enumerate(teams):
            p = ctrl.predictors[i]
            for k in range(6):
                p.learn(background[(rep * 6 + k) % 200], k % 5, background[(rep * 6 + k + 1) % 200], 0.)
            for goal, (b, a, f) in DELIVERY.items():
                p.learn(b, a, f, 1. if goal == team else 0.)
            p.learn(*RETURN, 1.)


def test_action_max_is_the_largest_term_of_the_pooled_weighted_mean():
    rng = np.random.default_rng(0)
    p, q = rng.uniform(0, 1, 60), rng.uniform(0, 1, 60)
    a = rng.choice([FORWARD, 2, TOGGLE], 60)
    pooled = V2.response_distance(p, q)
    mass = {g: np.sum(np.abs(p[a == g]) + np.abs(q[a == g])) for g in np.unique(a)}
    terms = {g: V2.response_distance(p[a == g], q[a == g]) for g in np.unique(a)}
    assert pooled == pytest.approx(sum(mass[g] / sum(mass.values()) * terms[g] for g in terms))
    d, g = V2.action_max_distance(p, q, a, a, a, min_count=1)
    assert d == pytest.approx(max(terms.values())) and terms[g] == d and pooled <= d


def test_action_max_compares_only_actions_both_agents_experienced():
    a = np.array([FORWARD] * 8 + [TOGGLE] * 24)
    p = np.r_[np.ones(8), np.full(24, .9)]
    q = np.r_[np.zeros(8), np.full(24, .9)]
    assert V2.response_distance(p, q) < .2  # diluted by the shared kind
    both = np.array([FORWARD] * 8 + [TOGGLE] * 8)
    assert V2.action_max_distance(p, q, a, both, both, 8) == (pytest.approx(1.), FORWARD)
    few_deliveries = np.array([FORWARD] * 2 + [TOGGLE] * 14)
    assert V2.action_max_distance(p, q, a, both, few_deliveries, 8) == (pytest.approx(0.), TOGGLE)
    assert V2.action_max_distance(p, q, a, both, np.array([2] * 16), 8) == (None, None)


@pytest.mark.parametrize("snapshot", [False, True])
def test_shared_returns_let_pooled_accept_other_team_but_not_action_max(snapshot):
    decisions = {}
    for mode in ("pooled", "action_max"):
        ctrl = V2.ACV2Controller(4, options(distance_mode=mode, contact_snapshot=snapshot))
        teach(ctrl, [0, 1, 0, 1])
        if snapshot:  # refit happens after the comparison: fit once, then freeze the rollout-start state
            for p in ctrl.predictors:
                p.train()
            ctrl.snapshot()
        allocation, log = ctrl(1, np.ones((4, 4), bool), active=True)
        assert log["ac_mature"] == [True] * 4
        rows = {(r["receiver"], r["donor"]): r for r in log["ac_pairs"]}
        decisions[mode] = (allocation, rows)
    allocation, rows = decisions["pooled"]
    assert rows[(0, 1)]["distance"] < .5 and allocation[0, 1] > 0  # the late-training failure (D107)
    allocation, rows = decisions["action_max"]
    cross, same = rows[(0, 1)], rows[(0, 2)]
    assert cross["pooled_distance"] < .5 < cross["distance"] and cross["stratum"] == FORWARD
    assert not cross["accepted"] and allocation[0, 1] == 0 and allocation[1, 2] == 0
    assert same["accepted"] and same["distance"] < .5 and allocation[0, 2] > 0 and allocation[1, 3] > 0


def test_pair_without_a_common_action_is_not_compared():
    ctrl = V2.ACV2Controller(2, options(distance_mode="action_max", min_probes=20))  # 16 probes per agent
    teach(ctrl, [0, 0])
    allocation, log = ctrl(1, np.ones((2, 2), bool), active=True)
    (row,) = log["ac_pairs"]
    assert row["reason"] == "no_common_action" and row["distance"] is None and row["pooled_distance"] < .5
    assert not allocation.any() and ctrl.rounds[0, 1] == 0
    assert ctrl.last_metrics["ac_v2_pairs_no_common_action"] == 1.0


def test_default_is_pooled_old_configs_load_and_unknown_mode_is_rejected():
    assert V2.ACV2Config().distance_mode == "pooled"
    old = {k: v for k, v in V2.ACV2Config().__dict__.items() if k != "distance_mode"}
    assert V2.ACV2Config(**old).distance_mode == "pooled"
    with pytest.raises(ValueError, match="distance_mode"):
        V2.ACV2Config(distance_mode="mean")


def test_real_trainer_action_max_on_comm5(tmp_path):
    """Permissive gate values (wiring only, not research values): real trainer, contact range 5, no mirror,
    action_max compare -> approval -> Max-GAE donors."""
    import json
    import torch
    from test_max_gae import max_plan
    from latent_team import train_from_scratch as S
    checkpoint = S.C.ROOT / "runs/ac_20261023_185556/b6_corner_indiv_pen005_ac_ippo_s1/checkpoint_01000.pt"
    if not checkpoint.is_file():
        pytest.skip("Local 1M checkpoint missing")
    options = tmp_path / "ac.json"
    options.write_text(json.dumps(dict(interval=1, batch_size=16, train_steps=2, min_events=1, min_event_response=-1.0,
                                       event_reward=.001, min_rounds=1, distance_threshold=1.0, min_probes=1,
                                       mirror=False)))
    output, manifest = max_plan(tmp_path, modes=["pgct_max"], ac_gate="v2", ac_config=options,
                                ac_distance="action_max", env_variant="b6_corner_indiv_comm5_pen005",
                                init_checkpoint=checkpoint)
    job = manifest["jobs"][0]
    assert job["name"].startswith("b6_corner_indiv_comm5_pen005_acv2_amax_")
    S.run_job(output, manifest, job["name"], manifest["eval_seed"])
    folder = output / job["name"]
    rows = [json.loads(s) for s in (folder / "max_gae.jsonl").read_text().splitlines()]
    compared = [p for r in rows for p in r["ac_pairs"] if p["distance"] is not None]
    assert compared and all(p["stratum"] is not None and p["pooled_distance"] is not None for p in compared)
    for r in rows:
        contact = np.asarray(r["contact"], bool) & ~np.eye(6, dtype=bool)
        assert np.array_equal(np.asarray(r["donors"], bool), contact & np.asarray(r["approved"], bool))
    final = torch.load(folder / "final.pt", weights_only=False)
    config = final["reward_probe_ac_v2"]["config"]
    assert config["distance_mode"] == "action_max" and not config["mirror"] and config["contact_snapshot"]
    assert final["effective_env_kwargs"]["communication_range"] == 5


def plan_args(tmp_path, **kwargs):
    from latent_team.common import REFERENCE_EXPERIMENT
    if not (REFERENCE_EXPERIMENT / "experiment.json").exists():
        pytest.skip("Local reference configuration missing")
    return SimpleNamespace(**dict(dict(
        reference_experiment=REFERENCE_EXPERIMENT, output_dir=tmp_path / "plan", maps=["e1"], seeds=[1],
        modes=["ippo", "pgct"], timesteps=64, home_ratio=.5, learning_rate=2e-4, entropy_coef=.05,
        save_interval=0, track=False, save_video=False, smoke_test=True, stage_label="ac_test",
        wandb_project="test", wandb_mode="offline", env_variant="b_e1f6_comm3_pen005", team_reward="individual",
        probe_bank="canonical", eval_seed=5000001, reward_probe_ac=True, ac_gate="v2"), **kwargs))


def test_plan_ac_distance_flag_names_jobs_and_records_the_mode(tmp_path):
    from latent_team.train_from_scratch import plan
    _, manifest = plan(plan_args(tmp_path, ac_distance="action_max"))
    assert manifest["reward_probe_ac_v2"]["distance_mode"] == "action_max"
    for job in manifest["jobs"]:
        assert "_acv2_amax_" in job["name"] and job["reward_probe_ac_v2"]["distance_mode"] == "action_max"
    _, default = plan(plan_args(tmp_path / "default"))
    assert all("_amax" not in j["name"] and j["reward_probe_ac_v2"]["distance_mode"] == "pooled"
               for j in default["jobs"])
    with pytest.raises(ValueError, match="ac-gate v2"):
        plan(plan_args(tmp_path / "v1", ac_gate="v1", ac_distance="action_max"))
