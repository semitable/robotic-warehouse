"""Gate v2 information flow and decisions. No policy optimisation or long-running training."""
import json
from pathlib import Path
from types import SimpleNamespace
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from latent_team import reward_probe_ac_v2 as V2

R = 2
SIZE = V2.view_size(R)


def options(**kwargs):
    return V2.ACV2Config(**dict(dict(buffer_size=4000, event_buffer_size=64, batch_size=64, train_steps=300,
                                      min_events=5, min_rounds=1, interval=1, mirror=False, probes_per_agent=16),
                                 **kwargs))


def views(seed, n=1):
    return np.random.default_rng(seed).integers(0, 2, size=(n, SIZE)).astype(np.uint8)


EVENT = {0: (views(1)[0], 1, views(2)[0]), 1: (views(3)[0], 1, views(4)[0])}  # two goals' delivery views


def teach(ctrl, teams, deliver=True):
    """teams[i] = goal whose delivery pays agent i, or None (never paid). Others see both goals."""
    background = views(9, 200)
    for rep in range(30):
        for i, team in enumerate(teams):
            p = ctrl.predictors[i]
            for k in range(6):
                b = background[(rep * 6 + k) % 200]
                p.learn(b, k % 5, background[(rep * 6 + k + 1) % 200], 0.)
            for goal, (b, a, f) in EVENT.items():
                paid = deliver and team == goal
                if team is not None or goal == 0:
                    p.learn(b, a, f, 1. if paid else 0.)


def test_mirror_is_an_involution_and_swaps_left_right():
    v = views(5, 3)
    m = V2.mirror_view(v, R)
    assert np.array_equal(V2.mirror_view(m, R), v)
    assert np.array_equal(m[:, [3, 4]], v[:, [4, 3]])
    grid = v[:, V2.EGO:].reshape(3, 11, 5, 5)
    mg = m[:, V2.EGO:].reshape(3, 11, 5, 5)
    assert np.array_equal(mg[:, 0], grid[:, 0, :, ::-1])
    assert np.array_equal(mg[:, V2.DIR_LEFT], grid[:, V2.DIR_RIGHT, :, ::-1])
    assert V2.mirror_action([0, 1, 2, 3, 4]).tolist() == [0, 1, 3, 2, 4]


def test_gate_pairs_teammates_rejects_other_team_and_unpaid_agent_abstains():
    ctrl = V2.ACV2Controller(5, options())
    teach(ctrl, [0, 1, 0, 1, None])
    allocation, log = ctrl(1, np.ones((5, 5), bool), active=True)
    assert log["ac_mature"] == [True, True, True, True, False]
    donors = {i: set(np.flatnonzero(allocation[i]).tolist()) for i in range(5)}
    assert donors == {0: {2}, 1: {3}, 2: {0}, 3: {1}, 4: set()}
    rows = {(r["receiver"], r["donor"]): r for r in log["ac_pairs"]}
    assert rows[(0, 2)]["distance"] < .5 < rows[(0, 1)]["distance"]
    assert rows[(0, 4)]["reason"] == "immature"
    audit = V2.audit(ctrl, [0, 1, 0, 1, 0])
    assert audit["ac_v2_audit_same_accepted"] == 2 and audit["ac_v2_audit_cross_accepted"] == 0


def test_lost_contact_inactive_and_untrained_give_no_donor():
    ctrl = V2.ACV2Controller(4, options())
    teach(ctrl, [0, 1, 0, 1])
    graph = np.ones((4, 4), bool)
    assert ctrl(1, graph, active=True)[0][0, 2] > 0
    graph[0, 2] = graph[2, 0] = False
    assert not ctrl(2, graph, active=True)[0][0].any()
    assert not ctrl(3, np.ones((4, 4)), active=False)[0].any()
    fresh = V2.ACV2Controller(2, options())
    assert not fresh(1, np.ones((2, 2)), active=True)[0].any()


def test_private_random_streams_leave_the_trainer_rng_untouched():
    torch.manual_seed(123)
    np.random.seed(7)
    torch_state, numpy_state = torch.get_rng_state(), np.random.get_state()
    ctrl = V2.ACV2Controller(4, options())
    teach(ctrl, [0, 1, 0, 1])
    ctrl(1, np.ones((4, 4)), active=True)
    assert torch.equal(torch.get_rng_state(), torch_state)
    assert np.random.get_state()[1].tolist() == numpy_state[1].tolist()


def test_install_refuses_team_reward():
    cfg = SimpleNamespace(env_kwargs_json=json.dumps(dict(team_reward_mode="team", n_agents=2)),
                          observation_format="semantic", sensor_range=5, env_id="test", agent_count=2)
    with pytest.raises(ValueError, match="individual"):
        with V2.install(cfg):
            pass


def v2_plan(tmp_path, gate="v2"):
    from latent_team.train_from_scratch import plan
    from latent_team.common import REFERENCE_EXPERIMENT
    if not (REFERENCE_EXPERIMENT / "experiment.json").exists():
        pytest.skip("Local reference configuration missing")
    args = SimpleNamespace(reference_experiment=REFERENCE_EXPERIMENT, output_dir=tmp_path/"plan", maps=["e1"],
                           seeds=[1], modes=["ippo", "pgct"], timesteps=64, home_ratio=.5,
                           learning_rate=2e-4, entropy_coef=.05, save_interval=0, track=False, save_video=False,
                           smoke_test=True, stage_label="ac_test", wandb_project="test", wandb_mode="offline",
                           env_variant="b_e1f6_comm3_pen005", team_reward="individual", probe_bank="canonical",
                           eval_seed=5000001, reward_probe_ac=True, ac_gate=gate)
    _, manifest = plan(args)
    return tmp_path/"plan", manifest


def test_plan_v2_is_opt_in_and_keeps_v1_names(tmp_path):
    _, manifest = v2_plan(tmp_path)
    assert manifest["reward_probe_ac"] is None and manifest["reward_probe_ac_v2"]["distance_threshold"] == .5
    for job in manifest["jobs"]:
        assert "_acv2_" in job["name"] and "reward_probe_ac" not in job and job["reward_probe_ac_v2"]
        assert json.loads(job["config"]["env_kwargs_json"])["team_reward_mode"] == "individual"
    _, v1 = v2_plan(tmp_path/"v1", gate="v1")
    assert all("_ac_" in j["name"] and "reward_probe_ac_v2" not in j for j in v1["jobs"])


def test_runner_uses_v2_allocation_logs_metrics_and_saves_state(monkeypatch, tmp_path):
    from latent_team import train_from_scratch as S
    from examples import train_recurrent_ippo_consensus as T
    output, manifest = v2_plan(tmp_path)
    job = next(j for j in manifest["jobs"] if j["mode"] == "pgct")
    returned = []

    def collect(env, *args, **kwargs):
        return env.step([0]*env.unwrapped.n_agents)

    def update(*args, **kwargs):
        return {}

    def no_optimisation_train(cfg):
        kwargs = json.loads(cfg.env_kwargs_json)
        env = T.make_env(cfg.env_id, kwargs, cfg.observation_format)
        try:
            obs, _ = env.reset(seed=cfg.seed)
            assert not np.asarray(obs)[:, :2].any()
            n = len(obs)
            agents = [torch.nn.Linear(1, 1) for _ in range(n)]
            T.collect_rollout(env)
            graph = SimpleNamespace(state_dict=lambda: {})
            for _ in range(5):
                returned.append(T.update_ippo(agents, [], None, cfg, graph, torch.device("cpu"),
                                              np.random.default_rng(1), peer_allocation=np.ones((n, n), np.float32)))
            T.save_checkpoint(output/job["name"]/"final.pt", cfg, agents, [], graph, 1, 1, env_kwargs=kwargs)
        finally:
            env.close()

    monkeypatch.setattr(T, "collect_rollout", collect)
    monkeypatch.setattr(T, "update_ippo", update)
    monkeypatch.setattr(T, "train", no_optimisation_train)
    S.run_job(output, manifest, job["name"], manifest["eval_seed"])
    rows = [json.loads(line) for line in (output/job["name"]/"transfer.jsonl").read_text().splitlines()]
    assert rows[-1]["ac_model"] == "mlp_reward_event_probe_v2"
    assert not any(np.asarray(r["actual_allocation"]).any() for r in rows)  # no own reward events yet
    assert "ac_v2_gate_arcs" in returned[-1] and "ac_v2_mature_agents" in returned[-1]
    saved = torch.load(output/job["name"]/"final.pt", weights_only=False)
    assert "reward_probe_ac_v2" in saved and saved["effective_env_kwargs"]["team_reward_mode"] == "individual"
    assert T.update_ippo is update and T.collect_rollout is collect
