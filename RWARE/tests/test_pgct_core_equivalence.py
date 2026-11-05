"""Migration regression: the pgct_core extraction must not change RWARE PGCT-D2 behaviour.

Every check runs the frozen pre-split implementation (tests/legacy/, commit 122615f) and the refactored code
(latent_team -> pgct_core) on identical inputs and requires EXACT equality (np.array_equal / torch.equal /
identical logs). Deterministic components have no tolerance; the trainer-level check compares final weights
bit for bit on the same machine.
"""
import copy
import json
from pathlib import Path
from types import SimpleNamespace
import sys

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from examples import train_recurrent_ippo_consensus as T  # noqa: E402
from latent_team import max_gae as MG  # noqa: E402
from latent_team import reward_probe_ac_v2 as V2  # noqa: E402
from legacy import max_gae_legacy as LMG  # noqa: E402
from legacy import reward_probe_ac_v2_legacy as LV2  # noqa: E402
import pgct_core  # noqa: E402
from pgct_core import networks as CN, ppo as CP  # noqa: E402

GATE_SETTINGS = [dict(distance_mode=d, mirror=m, contact_snapshot=s)
                 for d in ("pooled", "action_max") for m in (True, False) for s in (False, True)]


def semantic_obs(rng, n, sensor=2):
    side = 2 * sensor + 1
    obs = rng.integers(0, 2, size=(n, 20 + 13 * side * side)).astype(np.float32)
    obs[:, :2] = 0
    return obs


def run_gate(module, settings, observe_raw):
    cfg = module.ACV2Config(interval=2, batch_size=32, train_steps=3, min_events=4, min_event_response=-1.0,
                            event_reward=.5, probes_per_agent=8, min_probes=2, min_rounds=2, distance_threshold=.9,
                            seed=3, **settings)
    ctrl = module.ACV2Controller(4, cfg)
    rng = np.random.default_rng(11)
    contacts = [np.ones((4, 4), bool), np.array([[1, 1, 0, 0], [1, 1, 0, 0], [0, 0, 1, 1], [0, 0, 1, 1]], bool)]
    trace = []
    for update in range(1, 9):
        if cfg.contact_snapshot:
            ctrl.snapshot()
        for _ in range(40):
            rewards = rng.choice([0., .2, 1.], size=4, p=[.6, .2, .2])
            actions = rng.integers(0, 5, size=4)
            if observe_raw:
                ctrl.observe(semantic_obs(rng, 4), actions, semantic_obs(rng, 4), rewards)
            else:
                size = module.view_size(cfg.radius)
                views = rng.integers(0, 2, size=(2, 4, size)).astype(np.uint8)
                ctrl.observe_views(views[0], actions, views[1], rewards)
        alloc, log = ctrl(update, contacts[update % 2], active=update > 2)
        trace.append(dict(alloc=alloc.copy(), log=json.dumps(log, sort_keys=True, default=str),
                          metrics=json.dumps(ctrl.last_metrics, sort_keys=True), ema=ctrl.ema.copy(),
                          rounds=ctrl.rounds.copy(), gate=ctrl.gate.copy(), approved=ctrl.approved(),
                          mature=ctrl.maturity()))
    weights = [{k: v.clone() for k, v in p.net.state_dict().items()} for p in ctrl.predictors]
    return trace, weights, ctrl.state_dict()


@pytest.mark.parametrize("settings", GATE_SETTINGS, ids=lambda s: "-".join(f"{v}" for v in s.values()))
@pytest.mark.parametrize("observe_raw", [False, True])
def test_gate_matches_legacy_exactly(settings, observe_raw):
    new, new_w, new_state = run_gate(V2, settings, observe_raw)
    old, old_w, old_state = run_gate(LV2, settings, observe_raw)
    assert any(t["log"].count('"accepted": true') for t in old)  # the comparison path really ran
    for a, b in zip(new, old):
        for key in ("alloc", "ema", "rounds", "gate", "approved", "mature"):
            assert np.array_equal(a[key], b[key]), key
        assert a["log"] == b["log"] and a["metrics"] == b["metrics"]
    for a, b in zip(new_w, old_w):
        assert all(torch.equal(a[k], b[k]) for k in a)
    assert new_state["config"] == old_state["config"]
    assert new_state["last_versions"] == old_state["last_versions"]


def test_config_defaults_and_helpers_match_legacy():
    from dataclasses import asdict, fields
    assert asdict(V2.ACV2Config()) == asdict(LV2.ACV2Config())
    assert [f.name for f in fields(V2.ACV2Config)] == [f.name for f in fields(LV2.ACV2Config)]
    assert V2.ACV2Config().mirror is True and pgct_core.RewardGateConfig().mirror is False
    rng = np.random.default_rng(0)
    views = rng.integers(0, 2, size=(5, V2.view_size(2))).astype(np.uint8)
    assert np.array_equal(V2.mirror_view(views, 2), LV2.mirror_view(views, 2))
    assert np.array_equal(V2.features(views, np.arange(5), views), LV2.features(views, np.arange(5), views))
    p, q = rng.random(12), rng.random(12)
    assert V2.response_distance(p, q) == LV2.response_distance(p, q)
    acts, own = rng.integers(0, 5, 12), rng.integers(0, 5, 12)
    assert V2.action_max_distance(p, q, acts, own, own, 1) == LV2.action_max_distance(p, q, acts, own, own, 1)


def rollout(steps=24, n=3, seed=0, dones=None, dim=6):
    g = torch.Generator().manual_seed(seed)
    return SimpleNamespace(rewards=torch.rand(steps, n, generator=g), values=torch.rand(steps, n, generator=g),
                           last_values=torch.rand(n, generator=g), timeout_values=torch.rand(steps, n, generator=g),
                           dones=torch.zeros(steps) if dones is None else dones,
                           obs=torch.rand(steps, n, dim, generator=g))


def nets(n=3, dim=6, seed=0):
    torch.manual_seed(seed)
    return [T.RecurrentActorCritic(dim, 5, 16, 8) for _ in range(n)]


def test_gae_and_advantage_math_match_legacy():
    r = rollout(dones=torch.tensor([0.] * 10 + [1.] + [0.] * 13))
    assert all(torch.equal(a, b) for a, b in zip(
        pgct_core.compute_gae(r.rewards, r.dones, r.values, r.last_values, .99, .95, r.timeout_values),
        T.compute_gae(r.rewards, r.dones, r.values, r.last_values, .99, .95, r.timeout_values)))
    f = r.values + torch.rand(r.values.shape)
    valid, _ = MG.warm_mask(r.dones, 8)
    lvalid, _ = LMG.warm_mask(r.dones, 8)
    assert torch.equal(valid, lvalid)
    for scale in ("raw", "shrink", "relative"):
        args = (r, .99, .95, .9, .5, f, r.last_values + .3, r.timeout_values, np.array([True, False, True]))
        new = MG.blended_advantages(*args, scale=scale, valid=valid)
        old = LMG.blended_advantages(*args, scale=scale, valid=valid)
        assert all(torch.equal(a, b) for a, b in zip(new, old)), scale
    assert torch.equal(MG.explained_variance(r.values, f), LMG.explained_variance(r.values, f))


def test_donor_selection_matches_legacy():
    contact = np.ones((6, 6), bool)
    teams = np.array([0, 1, 0, 1, 0, 1])
    approved = np.random.default_rng(1).random((6, 6)) > .5
    mature = np.array([1, 1, 0, 1, 1, 1], bool)
    critic = np.array([1, 0, 1, 1, 1, 1], bool)
    for mode in MG.DONOR_MODES:
        a = MG.select_donors(mode, teams, contact, np.random.default_rng(5), approved=approved, mature=mature,
                             critic_mature=critic)
        b = LMG.select_donors(mode, teams, contact, np.random.default_rng(5), approved=approved, mature=mature,
                              critic_mature=critic)
        assert np.array_equal(a, b), mode


def test_donor_critics_and_max_targets_match_legacy():
    agents = nets()
    r = rollout(dones=torch.tensor([0.] * 15 + [1.] + [0.] * 8))
    timeout_obs = {15: torch.rand(3, 6)}
    last = torch.rand(3, 6)
    donors = np.array([[0, 1, 1], [0, 0, 1], [0, 0, 0]], bool)
    for cap in (None, .05):
        config = MG.MaxGAEConfig(donors="gate", cap=cap)
        new = MG.max_targets(agents, r, donors, timeout_obs, last, config, torch.device("cpu"))
        old = LMG.max_targets(agents, r, donors, timeout_obs, last, LMG.MaxGAEConfig(donors="gate", cap=cap),
                              torch.device("cpu"))
        assert all(torch.equal(torch.as_tensor(a), torch.as_tensor(b)) for a, b in zip(new, old))


def test_core_network_matches_trainer_network():
    torch.manual_seed(7)
    trainer_net = T.RecurrentActorCritic(11, 5, 32, 16, encoder_type="mlp")
    torch.manual_seed(7)
    core_net = CN.RecurrentActorCritic(11, 5, 32, 16)
    a, b = trainer_net.state_dict(), core_net.state_dict()
    assert list(a) == list(b) and all(torch.equal(a[k], b[k]) for k in a)
    obs, h = torch.rand(4, 11), torch.rand(4, 16)
    assert all(torch.equal(x, y) for x, y in zip(trainer_net._critic_step(obs, h), core_net._critic_step(obs, h)))


def ppo_inputs(seed=3, steps=32, n=2, dim=7):
    agents = nets(n, dim, seed)
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        logp = torch.log(torch.full((steps, n), .2))
    r = T.Rollout(obs=torch.rand(steps, n, dim, generator=g), actions=torch.randint(0, 5, (steps, n), generator=g),
                  action_masks=torch.ones(steps, n, 5), old_log_probs=logp,
                  rewards=(torch.rand(steps, n, generator=g) > .8).float(), dones=(torch.rand(steps, generator=g) > .9).float(),
                  values=torch.rand(steps, n, generator=g), actor_h=torch.rand(steps, n, 8, generator=g),
                  critic_h=torch.rand(steps, n, 8, generator=g), last_values=torch.rand(n, generator=g),
                  timeout_values=torch.zeros(steps, n))
    r.rewards[:, 1] = 0  # agent 1 exercises the zero-reward actor skip
    return agents, r


def test_core_ppo_update_matches_trainer_update_bit_for_bit():
    cfg = CP.PPOConfig(rollout_steps=32, sequence_length=8, minibatch_chunks=2, ppo_epochs=2)
    tcfg = SimpleNamespace(**{k: getattr(cfg, k) for k in vars(cfg)}, peer_transfer_mode="none", pgct_peer_loss_coef=.01)
    estimator = SimpleNamespace(update_value_uncertainty=lambda x: None)
    for injected in (False, True):
        agents, r = ppo_inputs()
        twins = copy.deepcopy(agents)
        opts = [torch.optim.Adam(a.parameters(), lr=2e-4, eps=1e-5) for a in agents]
        topts = [torch.optim.Adam(a.parameters(), lr=2e-4, eps=1e-5) for a in twins]
        adv = ret = None
        original = T.compute_gae
        if injected:  # the Max-GAE path: actor advantages differ from the critic's own returns
            adv, ret = pgct_core.compute_gae(r.rewards, r.dones, r.values, r.last_values, .99, .95, r.timeout_values)
            adv = adv * 1.7 + .1
            T.compute_gae = lambda *a, **k: (adv, ret)
        try:
            trainer_metrics = T.update_ippo(twins, topts, r, tcfg, estimator, torch.device("cpu"), np.random.default_rng(9))
        finally:
            T.compute_gae = original
        core_metrics = CP.ppo_update(agents, opts, r, cfg, torch.device("cpu"), np.random.default_rng(9),
                                     advantages=adv, returns=ret)
        for a, b in zip(agents, twins):
            sa, sb = a.state_dict(), b.state_dict()
            assert all(torch.equal(sa[k], sb[k]) for k in sa)
        for key in ("policy_loss", "value_loss", "entropy", "approx_kl", "agent_1/actor_update_skipped"):
            assert core_metrics[key] == trainer_metrics[key], key


def infer_network(state):
    """Rebuild a trainer network from a saved state dict (MLP or semantic CNN encoder)."""
    if "actor_encoder.spatial_cnn.0.weight" in state:
        channels = state["actor_encoder.spatial_cnn.0.weight"].shape[1]
        ego = state["actor_encoder.ego_encoder.0.weight"].shape[1]
        hidden = state["actor_encoder.spatial_fc.0.weight"].shape[0]
        side = int(round((state["actor_encoder.spatial_fc.0.weight"].shape[1] / 32) ** .5))
        rec = state["actor_rnn.weight_hh"].shape[1]
        return T.RecurrentActorCritic(ego + channels * side * side, state["actor_head.weight"].shape[0], hidden, rec,
                                      encoder_type="cnn", spatial_channels=channels, spatial_size=side, ego_dim=ego)
    obs_dim = state["actor_encoder.0.weight"].shape[1]
    return T.RecurrentActorCritic(obs_dim, state["actor_head.weight"].shape[0], state["actor_encoder.0.weight"].shape[0],
                                  state["actor_rnn.weight_hh"].shape[1])


@pytest.mark.parametrize("path", sorted(str(p.relative_to(ROOT)) for p in (ROOT / "runs").rglob("*.pt")
                                        if "sp_2026" not in str(p)) or ["<none>"])
def test_existing_checkpoints_still_load(path):
    """Checkpoints hold only tensors/arrays/builtins, so moving modules cannot break unpickling."""
    if path == "<none>":
        pytest.skip("no local checkpoints")
    payload = torch.load(ROOT / path, map_location="cpu", weights_only=False)
    states = payload["agents"] if isinstance(payload, dict) and "agents" in payload else None
    if states is None:
        pytest.skip(f"{path} is not an agent checkpoint")
    for state in states:
        infer_network(state).load_state_dict(state, strict=True)


def test_trainer_level_pgct_d2_matches_legacy(tmp_path, monkeypatch):
    """Real RWARE trainer, 64-step PGCT-D2 jobs: legacy modules vs refactored modules, identical outputs."""
    from latent_team import train_from_scratch as S
    if not (S.C.REFERENCE_EXPERIMENT / "experiment.json").exists():
        pytest.skip("Local reference configuration missing")
    import characterize_pgct_d2 as CH
    torch.set_num_threads(1)
    results = {}
    for label, v2, mg in (("new", V2, MG), ("legacy", LV2, LMG)):
        monkeypatch.setattr(S, "AC2", v2)
        monkeypatch.setattr(S, "MG", mg)
        options = tmp_path / f"{label}.json"
        options.write_text(json.dumps(CH.SETTINGS["natural"]), encoding="utf-8")
        out, manifest = CH.plan(tmp_path / label, "b6_corner_indiv_comm5_pen005", options,
                                ["pgct_max", "oracle_max"], "action_max")
        for job in manifest["jobs"]:
            S.run_job(out, manifest, job["name"], manifest["eval_seed"])
            results[label, job["mode"]] = CH.fingerprint(out / job["name"])
    assert results["new", "pgct_max"]["compared"] > 0
    for mode in ("pgct_max", "oracle_max"):
        assert results["new", mode] == results["legacy", mode], mode
