"""No policy optimisation or long-running training: information flow and gate tests."""
import json
from pathlib import Path
from types import SimpleNamespace
import sys

import gymnasium as gym
import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from latent_team import reward_probe_ac as AC


def scene(kind=0, side=11):
    obs = np.zeros(20+13*side*side, dtype=np.float32)
    obs[2] = obs[3] = obs[7] = 1  # carrying, up, highway
    g = obs[20:].reshape(13, side, side)
    c = side//2
    g[1] = 1
    g[3, c, c] = g[5, c, c] = g[6, c, c] = g[10, c, c] = 1
    g[11, c-1, c+kind] = 1
    return obs


def options(**kwargs):
    return AC.ACConfig(**dict(dict(min_samples=2, min_episodes=2, min_queries=2,
                                  min_rounds=1, interval=1), **kwargs))


def queries():
    return [AC.virtual_queries(scene(k), 2)[1] for k in (0, 1)]


def teach(predictor, rewards):
    for episode in range(2):
        for q, reward in zip(queries(), rewards):
            predictor.learn(q, reward, episode)


def test_public_queries_are_invariant_to_coordinates_and_personal_annotations():
    a = scene()
    b = a.copy()
    b[:2] = [99, -123]
    b[8:20] = 7
    b[20:].reshape(13, 11, 11)[[4, 12]] = 1
    q1 = AC.experience_query(a, 1, a, 2)
    q2 = AC.experience_query(b, 1, b, 2)
    assert q1.key == q2.key
    assert np.array_equal(AC.strip_coordinates(a)[2:], a[2:])
    assert a[0] == 0 and b[0] == 99  # no mutation


def test_virtual_forward_moves_self_and_shelf_to_observed_goal():
    q = AC.virtual_queries(scene(), 2)[1]
    after = q.after[6:].reshape(len(AC.PUBLIC_CHANNELS), 5, 5)
    assert q.action == 1 and q.source == "virtual_local"
    for channel in (3, 5, 6, 10, 11):
        assert after[AC.PUBLIC_CHANNELS.index(channel), 2, 2] == 1
    # Opposite rotations follow the environment's enum ordering (UP,DOWN,LEFT,RIGHT).
    qs = AC.virtual_queries(scene(), 2)
    assert np.argmax(qs[2].after[1:5]) == 2
    assert np.argmax(qs[3].after[1:5]) == 3


def test_virtual_forward_rejects_hidden_border_and_collisions():
    assert AC.virtual_queries(scene(side=5), 2) == []
    for channel in (0, 3, 5):
        obs = scene()
        obs[20:].reshape(13, 11, 11)[channel, 4, 5] = 1
        assert 1 not in [q.action for q in AC.virtual_queries(obs, 2)]


def test_only_personal_rewards_are_learned_unknown_and_aliases_abstain():
    p = AC.LocalRewardPredictor(options())
    assert p.predict(queries()[0]) is None
    teach(p, [1, 0])
    assert [p.predict(q)[0] for q in queries()] == [1, 0]
    other = AC.LocalRewardPredictor(options())
    teach(other, [0, 1])
    assert [other.predict(q)[0] for q in queries()] == [0, 1]
    # Same observable event with conflicting hidden meaning must not fake certainty.
    p.learn(queries()[0], 0, 3)
    assert p.predict(queries()[0]) is None
    controller = AC.ACController(2, options())
    with pytest.raises(ValueError, match="separate scalar"):
        controller.observe([scene(), scene()], [1, 1], [scene(), scene()], 2.)


def test_gate_selects_compatible_responses_never_zeros_unknown_or_out_of_range():
    ctrl = AC.ACController(4, options())
    for predictor, rewards in zip(ctrl.predictors, ([1, 0], [0, 1], [1, 0], [0, 1])):
        teach(predictor, rewards)
    neighbors = np.ones((4, 4), bool)
    allocation, log = ctrl(1, neighbors, active=True)
    assert allocation.tolist() == [[0, 0, 1, 0], [0, 0, 0, 1], [1, 0, 0, 0], [0, 1, 0, 0]]
    assert log["ac_pairs"][0]["raw_distance"] == 1
    neighbors[0, 2] = neighbors[2, 0] = False
    assert not ctrl(2, neighbors, active=True)[0][0].any()
    assert not ctrl(3, neighbors, active=False)[0].any()
    zeros = AC.ACController(2, options())
    for p in zeros.predictors:
        teach(p, [0, 0])
    assert not zeros(1, np.ones((2, 2)), active=True)[0].any()
    unknown = AC.ACController(2, options())
    teach(unknown.predictors[0], [1, 0])
    assert not unknown(1, np.ones((2, 2)), active=True)[0].any()


def test_repeated_probe_calls_do_not_manufacture_evidence():
    ctrl = AC.ACController(2, options(min_rounds=2))
    for p in ctrl.predictors:
        teach(p, [1, 0])
    graph = np.ones((2, 2))
    assert not ctrl(1, graph, True)[0].any()
    assert not ctrl(2, graph, True)[0].any()
    assert ctrl.rounds[0, 1] == 1
    for p in ctrl.predictors:
        p.learn(queries()[0], 1, 2)
    assert ctrl(3, graph, True)[0][0, 1] == 1


class TinyEnv(gym.Env):
    """Provides no team maps, identities, positions, reward function or event info."""
    def __init__(self):
        self.observation_space = gym.spaces.Tuple(tuple(gym.spaces.Box(-100, 100, (1593,)) for _ in range(2)))
        self.action_space = gym.spaces.Tuple((gym.spaces.Discrete(5), gym.spaces.Discrete(5)))

    def reset(self, **kwargs):
        obs = scene()
        obs[:2] = [4, 6]
        return (obs.copy(), obs.copy()), {}

    def step(self, actions):
        obs = scene()
        obs[:2] = [4, 5]
        return (obs.copy(), obs.copy()), [1., 0.], False, False, {}


def test_scoped_adapter_captures_true_post_step_and_excludes_evaluation(monkeypatch, tmp_path):
    from examples import train_recurrent_ippo_consensus as T
    original_make = lambda *a, **kw: TinyEnv()
    monkeypatch.setattr(T, "make_env", original_make)
    monkeypatch.setattr(T, "collect_rollout", lambda env, *a, **kw: env.step([1, 1]))
    cfg = SimpleNamespace(env_kwargs_json=json.dumps(dict(team_reward_mode="individual", n_agents=2)),
                          observation_format="semantic", sensor_range=5, env_id="test", agent_count=2)
    with AC.install(cfg, tmp_path, options()) as ctrl:
        env = T.make_env()
        obs, _ = env.reset()
        assert np.asarray(obs)[:, :2].sum() == 0
        rewards = T.collect_rollout(env)[1]
        assert rewards == [1., 0.] and [p.version for p in ctrl.predictors] == [1, 1]
        # Post-step capture happens before a caller resets. Evaluation is not ingestion.
        env.reset()
        env.step([1, 1])
        assert [p.version for p in ctrl.predictors] == [1, 1]
        q = next(iter(ctrl.predictors[0].memory))
        assert ctrl.predictors[0].memory[q].mean == 1
        assert ctrl.predictors[1].memory[q].mean == 0
    assert T.make_env is original_make
    with pytest.raises(ValueError, match="individual"):
        with AC.install(SimpleNamespace(**{**vars(cfg), "env_kwargs_json": '{"team_reward_mode":"team"}'})):
            pass


def ac_plan(tmp_path):
    from latent_team.train_from_scratch import plan
    from latent_team.common import REFERENCE_EXPERIMENT
    if not (REFERENCE_EXPERIMENT / "experiment.json").exists():
        pytest.skip("Local reference configuration missing")
    args = SimpleNamespace(reference_experiment=REFERENCE_EXPERIMENT, output_dir=tmp_path/"plan", maps=["e1"],
                           seeds=[1], modes=["ippo", "pgct", "all"], timesteps=64, home_ratio=.5,
                           learning_rate=2e-4, entropy_coef=.05, save_interval=0, track=False, save_video=False,
                           smoke_test=True, stage_label="ac_test", wandb_project="test", wandb_mode="offline",
                           env_variant="b_e1f6_comm3_pen005", team_reward="individual", probe_bank="canonical",
                           eval_seed=5000001, reward_probe_ac=True)
    _, manifest = plan(args)
    return tmp_path/"plan", manifest


def test_plan_has_personal_rewards_for_all_baselines_and_masks_bank(tmp_path):
    _, manifest = ac_plan(tmp_path)
    for job in manifest["jobs"]:
        assert json.loads(job["config"]["env_kwargs_json"])["team_reward_mode"] == "individual"
        assert job["reward_probe_ac"] and "_ac_" in job["name"]
        if job["canonical_bank"]:
            bank = torch.load(job["canonical_bank"], weights_only=False)
            assert not bank["sequences"][..., :2].any()
    pgct = next(j for j in manifest["jobs"] if j["mode"] == "pgct")
    assert pgct["config"]["graph_mode"] == "unrestricted"  # donor override is A+C, not policy JS


def test_runner_passes_actual_ac_allocation_and_saves_state_without_optimising(monkeypatch, tmp_path):
    from latent_team import train_from_scratch as S
    from examples import train_recurrent_ippo_consensus as T
    output, manifest = ac_plan(tmp_path)
    job = next(j for j in manifest["jobs"] if j["mode"] == "pgct")
    seen = []

    def collect(env, *args, **kwargs):
        return env.step([0]*env.unwrapped.n_agents)

    def update(*args, **kwargs):
        seen.append(kwargs["peer_allocation"].copy())
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
            T.update_ippo(agents, [], None, cfg, graph, torch.device("cpu"), np.random.default_rng(1),
                          peer_allocation=np.ones((n, n), np.float32))
            T.save_checkpoint(output/job["name"]/"final.pt", cfg, agents, [], graph, 1, 1, env_kwargs=kwargs)
        finally:
            env.close()

    monkeypatch.setattr(T, "collect_rollout", collect)
    monkeypatch.setattr(T, "update_ippo", update)
    monkeypatch.setattr(T, "train", no_optimisation_train)
    S.run_job(output, manifest, job["name"], manifest["eval_seed"])
    assert len(seen) == 1 and not seen[0].any()  # insufficient own experience -> no donor
    saved = torch.load(output/job["name"]/"final.pt", weights_only=False)
    assert saved["reward_probe_ac"]["predictors"][0]["version"] == 1
    assert saved["effective_env_kwargs"]["team_reward_mode"] == "individual"
    log = json.loads((output/job["name"]/"transfer.jsonl").read_text().strip())
    assert log["ac_model"] == "local_empirical_reward"
    assert not np.asarray(log["actual_allocation"]).any()
    assert T.update_ippo is update and T.collect_rollout is collect


def test_virtual_actions_match_real_environment_without_policy_training():
    from latent_team import probes_v2 as P
    from latent_team.common import REFERENCE_EXPERIMENT
    path = REFERENCE_EXPERIMENT / "experiment.json"
    if not path.exists():
        pytest.skip("Local reference configuration missing")
    cfg = next(j["config"] for j in json.loads(path.read_text(encoding="utf-8"))["jobs"] if j["mode"] == "pgct")
    kwargs = json.loads(cfg["env_kwargs_json"])
    kwargs.update(team_reward_mode="individual", team_goal_positions=[[[0, 10]], [[9, 10]]])
    real = P.ProbeRealiser(cfg["env_id"], kwargs, "semantic")
    try:
        # Travel into a delivery goal; use ground truth only to score this test.
        for direction in range(4):
            for carrying in (False, True):
                sc = P.Scenario("transfer", "carry_goal_visible", 0, 0, 9, direction, carrying=carrying)
                before = real.realise(sc)
                for query in AC.virtual_queries(before, 2):
                    real.realise(sc)
                    obs, *_ = real.env.step([query.action, 0, 0, 0])
                    np.testing.assert_array_equal(query.after, AC.public_view(obs[0], 2))
    finally:
        real.close()


def test_same_physical_delivery_has_shared_public_query_and_personal_reward():
    # Oracle state construction is confined to this offline test, never the gate.
    from latent_team import probes_v2 as P
    from latent_team.common import REFERENCE_EXPERIMENT
    from rware.utils.semantic_observation import build_semantic_observation
    path = REFERENCE_EXPERIMENT / "experiment.json"
    if not path.exists():
        pytest.skip("Local reference configuration missing")
    cfg = next(j["config"] for j in json.loads(path.read_text(encoding="utf-8"))["jobs"] if j["mode"] == "pgct")
    kwargs = json.loads(cfg["env_kwargs_json"])
    kwargs.update(team_reward_mode="individual", team_goal_positions=[[[0, 10]], [[9, 10]]],
                  require_delivered_shelf_return=True)
    real = P.ProbeRealiser(cfg["env_id"], kwargs, "semantic")
    task = np.zeros((2, 2))
    predictors = [AC.LocalRewardPredictor(options()) for _ in range(2)]
    shared = []
    try:
        for owner, goal_x in ((0, 0), (1, 9)):
            keys = []
            for carrier_team in (0, 1):
                sc = P.Scenario("transfer", "carry_goal_visible", owner, goal_x, 9, 1, carrying=True)
                real.realise(sc)
                real.u.agent_team_ids[0] = carrier_team
                real.u._reset_progress_baselines()
                before = build_semantic_observation(real.u, real.u.agents[0])
                obs, rewards, _, _, info = real.env.step([1, 0, 0, 0])
                q = AC.experience_query(before, 1, obs[0], 2)
                keys.append(q.key)
                task[carrier_team, owner] = rewards[0] - info["reward_shaping"][0]
                for episode in range(2):
                    predictors[carrier_team].learn(q, rewards[0], episode)
                shared.append(q)
            assert keys[0] == keys[1]  # no donor's private annotation copied
        np.testing.assert_allclose(task, [[1, 0], [0, 1]])
        for p in predictors:
            assert all(p.predict(q) is not None for q in shared)
    finally:
        real.close()
