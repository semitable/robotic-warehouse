"""pgct_core unit tests: environment-independent gate, Max-GAE, metrics and the generic trainer."""
import json

import numpy as np
import pytest
import torch

import pgct_core
from pgct_core import interfaces, metrics, trainer
from pgct_core.max_gae import MaxGAEConfig, select_donors, shuffled_gate_donors
from pgct_core.ppo import PPOConfig
from pgct_core.reward_gate import RewardGateConfig, RewardProbeGate

from toy_env import ToyAdapter, ToyProjection


def toy_gate(**kw):
    cfg = RewardGateConfig(**dict(dict(interval=1, batch_size=32, train_steps=30, min_events=5, min_rounds=2,
                                       contact_snapshot=True, distance_mode="action_max", min_probes=4), **kw))
    return RewardProbeGate(4, cfg, ToyProjection(4))


def feed(gate, adapter, rng, steps):
    obs = adapter.reset()
    for _ in range(steps):
        actions = rng.integers(0, 3, 4)
        after, rewards, *_ = adapter.step(actions)
        gate.observe(list(obs), actions, list(after), rewards)
        obs = after


def test_projection_protocol_and_defaults():
    assert isinstance(ToyProjection(4), interfaces.ProbeProjection)
    assert isinstance(ToyAdapter(), interfaces.EnvAdapter)
    assert RewardGateConfig().mirror is False  # Q_ij = E_i u E_j, no augmentation by default
    with pytest.raises(ValueError, match="ProbeProjection"):
        RewardProbeGate(2, RewardGateConfig())
    gate = toy_gate(mirror=True, contact_snapshot=False)
    feed(gate, ToyAdapter(), np.random.default_rng(0), 300)
    with pytest.raises(ValueError, match="mirror"):
        gate.probes(0, 1)  # an explicit mirror request needs a projection that defines one
    gate = toy_gate()
    feed(gate, ToyAdapter(), np.random.default_rng(0), 300)
    gate.snapshot()  # with contact_snapshot the compared probes are the rollout-start copies
    b, a, f = gate.predictors[0].event_probes()
    assert len(gate.probes(0, 1)[0]) == len(b) + len(gate.predictors[1].event_probes()[0])


def test_gate_finds_the_toy_teams_without_labels():
    gate, adapter, rng = toy_gate(), ToyAdapter(teams=(1, 0, 0, 1)), np.random.default_rng(1)
    for update in range(1, 7):
        gate.snapshot()
        feed(gate, adapter, rng, 200)
        gate(update, np.ones((4, 4), bool), active=True)
    assert np.array_equal(gate.approved(), metrics.same_team_matrix(adapter.teams))


def test_float_views_are_supported():
    class FloatProjection(ToyProjection):
        view_dtype = np.float32
    gate = RewardProbeGate(4, RewardGateConfig(), FloatProjection(4))
    gate.observe_views(np.full((4, 6), .25, np.float32), np.zeros(4, int), np.full((4, 6), .5, np.float32), np.ones(4))
    assert gate.predictors[0].buffer.before.dtype == np.float32 and gate.predictors[0].buffer.before[0, 0] == .25


def test_gate_leaves_the_global_random_streams_untouched():
    torch.manual_seed(0)
    np.random.seed(0)
    expected = (torch.rand(3), np.random.rand(3))
    torch.manual_seed(0)
    np.random.seed(0)
    gate, adapter, rng = toy_gate(), ToyAdapter(), np.random.default_rng(2)
    gate.snapshot()
    feed(gate, adapter, rng, 100)
    gate(1, np.ones((4, 4), bool), active=True)
    assert torch.equal(torch.rand(3), expected[0]) and np.array_equal(np.random.rand(3), expected[1])


def test_shuffled_gate_matches_the_gate_count_and_ignores_labels():
    contact = np.ones((5, 5), bool)
    approved = np.zeros((5, 5), bool)
    approved[0, 1] = approved[1, 0] = approved[2, 3] = approved[3, 2] = approved[2, 4] = approved[4, 2] = True
    donors = shuffled_gate_donors(contact, approved, np.random.default_rng(0))
    assert donors.sum(1).tolist() == (contact & approved & ~np.eye(5, dtype=bool)).sum(1).tolist()
    assert not donors.diagonal().any()
    a = select_donors("gate", np.zeros(5), contact, None, approved=approved)
    b = select_donors("gate", np.arange(5), contact, None, approved=approved)
    assert np.array_equal(a, b)  # the gate rule does not depend on team ids
    with pytest.raises(ValueError):
        MaxGAEConfig(eta=1.5)


def test_identification_metrics():
    assert metrics.auroc([3, 2, 1, 0], [1, 1, 0, 0]) == 1.0 and metrics.auroc([0, 0], [1, 0]) == .5
    teams = [0, 0, 1, 1]
    right = metrics.same_team_matrix(teams)
    wrong = right.copy()
    wrong[0, 2] = wrong[2, 0] = True
    conf = metrics.pair_confusion(wrong, teams)
    assert (conf["tp"], conf["fp"], conf["fn"], conf["tn"]) == (2, 1, 0, 3)
    assert conf["precision"] == pytest.approx(2 / 3) and conf["false_approval_rate"] == pytest.approx(1 / 4)
    empty = np.zeros((4, 4), bool)
    rows = [dict(env_steps=10, approved=empty), dict(env_steps=20, approved=right),
            dict(env_steps=30, approved=wrong, pairs=[dict(receiver=0, donor=1, distance_ema=.1),
                                                      dict(receiver=0, donor=2, distance_ema=.9)])]
    s = metrics.identification_summary(rows, teams)
    assert s["first_correct_approval_step"] == 20 and s["first_wrong_approval_step"] == 30
    assert s["approval_flips"] == 3 and s["gate_stability"] == 0 and s["auroc"] == 1.0


@pytest.mark.parametrize("method", sorted(trainer.METHODS))
def test_generic_trainer_on_the_toy_environment(tmp_path, method):
    cfg = trainer.TrainerConfig(method=method, seed=3, total_timesteps=128, warmup_updates=0, eval_interval=1,
                                eval_episodes=1, save_interval=1, log_interval=0)
    ppo = PPOConfig(rollout_steps=64, sequence_length=16, minibatch_chunks=2, ppo_epochs=1, mlp_hidden_dim=16,
                    recurrent_hidden_dim=8)
    gate = RewardGateConfig(interval=1, batch_size=16, train_steps=5, min_events=1, min_event_response=-1.0,
                            min_rounds=1, min_probes=1, distance_threshold=1.0)
    result = trainer.run_training(ToyAdapter, tmp_path / method, cfg, ppo, gate)
    assert result["status"] == "complete" and result["env_steps"] == 128
    files = {p.name for p in (tmp_path / method).iterdir()}
    assert {"config.json", "metrics.jsonl", "gate.jsonl", "eval.jsonl", "final.pt", "checkpoint_00001.pt"} <= files
    rows = [json.loads(s) for s in (tmp_path / method / "gate.jsonl").read_text().splitlines()]
    if method == "pgct_d2":
        assert any(np.asarray(r["donors"]).any() for r in rows)
        assert all(np.array_equal(np.asarray(r["donors"]), np.asarray(r["contact"]) & np.asarray(r["approved"])
                                  & ~np.eye(4, dtype=bool)) for r in rows)
    if method in ("ippo", "pgct_id_only"):
        assert not any(np.asarray(r["donors"]).any() for r in rows)
    with pytest.raises(FileExistsError):
        trainer.run_training(ToyAdapter, tmp_path / method, cfg, ppo, gate)
    agents, payload = trainer.load_agents(tmp_path / method / "final.pt", 6, 3, ppo)
    assert len(agents) == 4 and payload["format"] == "pgct_core.trainer/1"


def relative_predictor(scale=1.0, **kw):
    from pgct_core.reward_gate import RewardPredictor
    cfg = RewardGateConfig(event_threshold="relative", event_scale=.1, event_buffer_size=8, **kw)
    p = RewardPredictor(cfg, 0, 3, 3)
    view = np.zeros(3, np.uint8)
    for r in (0.0, .1, .05, -1.0, .2, 1.0, .1, .05):
        p.learn(view, 0, view, r * scale)
    return p


def test_relative_event_rule_selects_and_purges_by_own_reward_scale():
    p = relative_predictor()
    rewards = p.events.recent(p.events.count)[3]
    # .05 was an event while the maximum was .1/.2, and was dropped once a reward of 1.0 raised the threshold to .1
    assert np.allclose(rewards, [.1, 1.0, .2, .1]) and p.events_seen == 4 and p.reward_max == 1.0
    assert p.event_threshold() == pytest.approx(.1) and "reward_max" in p.state_dict()
    scaled = relative_predictor(scale=7.0)  # scale-free: multiplying every reward keeps the same events
    assert np.allclose(scaled.events.recent(scaled.events.count)[3], 7.0 * rewards)


def test_absolute_rule_is_the_default_and_unchanged():
    from pgct_core.reward_gate import RewardPredictor
    p = RewardPredictor(RewardGateConfig(event_buffer_size=8), 0, 3, 3)
    view = np.zeros(3, np.uint8)
    for r in (0.0, .1, .05, -1.0, .2, 1.0, .1, .05, .5):
        p.learn(view, 0, view, r)
    assert np.allclose(p.events.recent(p.events.count)[3], [.5, 1.0]) and "reward_max" not in p.state_dict()
    with pytest.raises(ValueError):
        RewardGateConfig(event_threshold="quantile")


def test_relative_maturity_uses_the_own_event_reward_scale():
    gate = RewardProbeGate(4, RewardGateConfig(event_threshold="relative", interval=1, batch_size=32, train_steps=40,
                                               min_events=5, min_rounds=2, contact_snapshot=True, min_probes=4,
                                               distance_mode="action_max"), ToyProjection(4))
    adapter, rng = ToyAdapter(teams=(1, 0, 0, 1)), np.random.default_rng(1)
    for update in range(1, 7):
        gate.snapshot()
        obs = adapter.reset()
        for _ in range(200):  # toy rewards scaled down 20x: the absolute rule (0.5) would see no events at all
            actions = rng.integers(0, 3, 4)
            after, rewards, *_ = adapter.step(actions)
            gate.observe(list(obs), actions, list(after), rewards * .05)
            obs = after
        gate(update, np.ones((4, 4), bool), active=True)
    assert gate.maturity().all()
    assert np.array_equal(gate.approved(), metrics.same_team_matrix(adapter.teams))
