"""Max-GAE (D2/D1) advantage path. No policy optimisation or long-running training."""
import json
from pathlib import Path
from types import SimpleNamespace
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from examples import train_recurrent_ippo_consensus as T
from latent_team import max_gae as MG

GAMMA, LAM = .99, .95


def rollout(steps=12, n=2, seed=0, dones=None):
    g = torch.Generator().manual_seed(seed)
    return SimpleNamespace(rewards=torch.rand(steps, n, generator=g), values=torch.rand(steps, n, generator=g),
                           last_values=torch.rand(n, generator=g), timeout_values=torch.zeros(steps, n),
                           dones=torch.zeros(steps) if dones is None else dones,
                           obs=torch.zeros(steps, n, 3))


def test_max_advantage_matches_the_closed_form_and_leaves_other_columns_exact():
    r = rollout()
    b = torch.rand(12)
    b_last = .7
    f_values, f_last = r.values.clone(), r.last_values.clone()
    f_values[:, 0] += b
    f_last[0] += b_last
    actor, returns, adv_self, adv_max = MG.blended_advantages(r, GAMMA, LAM, LAM, 1.0, f_values, f_last,
                                                             r.timeout_values, np.array([True, False]))
    steps, gl = 12, GAMMA * LAM
    for t in range(steps):
        future = sum(gl ** (k - 1) * float(b[t + k]) for k in range(1, steps - t))
        expected = -float(b[t]) + GAMMA * (1 - LAM) * future + GAMMA * gl ** (steps - 1 - t) * b_last
        assert float(adv_max[t, 0] - adv_self[t, 0]) == pytest.approx(expected, abs=1e-5)
    ref_adv, ref_ret = T.compute_gae(r.rewards, r.dones, r.values, r.last_values, GAMMA, LAM, r.timeout_values)
    assert torch.equal(actor[:, 1], ref_adv[:, 1]) and torch.equal(returns, ref_ret)  # critic targets unchanged
    assert torch.equal(actor[:, 0], adv_max[:, 0])


def test_d2_blend():
    r = rollout(seed=3)
    f_values = r.values + .4
    actor, _, adv_self, adv_max = MG.blended_advantages(r, GAMMA, LAM, LAM, .3, f_values, r.last_values + .4,
                                                       r.timeout_values, np.array([True, True]))
    assert torch.allclose(actor, adv_self + .3 * (adv_max - adv_self))


def test_relative_scale_gives_the_max_term_eta_times_the_own_spread():
    r = rollout(seed=4)
    f_values = r.values + 3 * torch.rand(12, 2, generator=torch.Generator().manual_seed(9))  # large uplift
    actor, _, adv_self, adv_max = MG.blended_advantages(r, GAMMA, LAM, LAM, .5, f_values, r.last_values,
                                                       r.timeout_values, np.array([True, False]),
                                                       scale="relative")
    added = actor[:, 0] - adv_self[:, 0]
    assert float(added.std(unbiased=False)) == pytest.approx(.5 * float(adv_self[:, 0].std(unbiased=False)),
                                                             rel=1e-4)
    assert float(torch.corrcoef(torch.stack([added, adv_max[:, 0] - adv_self[:, 0]]))[0, 1]) == pytest.approx(1.)
    assert torch.equal(actor[:, 1], adv_self[:, 1])


def uplift_shift(r, b, scale, eta=.5):
    f_values = r.values.clone()
    f_values[:, 0] += b
    actor, _, adv_self, _ = MG.blended_advantages(r, GAMMA, LAM, LAM, eta, f_values, r.last_values,
                                                 r.timeout_values, np.array([True, False]), scale=scale)
    return actor[:, 0] - adv_self[:, 0], adv_self[:, 0]


@pytest.mark.parametrize("size", [1e-6, 1e-3, .3, 3.0])
def test_shrink_never_exceeds_raw_and_caps_the_spread_at_eta_times_the_own_spread(size):
    r = rollout(steps=64, seed=6)
    b = size * torch.rand(64, generator=torch.Generator().manual_seed(7))
    raw, adv_self = uplift_shift(r, b, "raw")
    shrink, _ = uplift_shift(r, b, "shrink")
    own = float(adv_self.std(unbiased=False))
    assert float(shrink.std(unbiased=False)) <= min(float(raw.std(unbiased=False)), .5 * own) * (1 + 1e-5)
    assert float(torch.corrcoef(torch.stack([shrink, raw]))[0, 1]) == pytest.approx(1., abs=1e-4)


def test_shrink_returns_to_ippo_as_the_uplift_vanishes_but_relative_does_not():
    r = rollout(steps=64, seed=8)
    for size in [1e-2, 1e-4, 1e-6]:
        b = size * torch.rand(64, generator=torch.Generator().manual_seed(9))
        shrink, adv_self = uplift_shift(r, b, "shrink")
        relative, _ = uplift_shift(r, b, "relative")
        own = float(adv_self.std(unbiased=False))
        assert float(shrink.abs().max()) < 10 * size  # vanishes with b (raw-sized)
        assert float(relative.std(unbiased=False)) == pytest.approx(.5 * own, rel=1e-3)  # E66: inflated


def test_masked_steps_keep_the_own_advantage_and_later_steps_are_unchanged():
    r = rollout(steps=64, seed=10)
    f_values = r.values.clone()
    f_values[7:, 0] += .8  # uplift only where the donor is warm (window 8)
    valid, _ = MG.warm_mask(r.dones, 8)
    args = (r, GAMMA, LAM, LAM, 1.0, f_values, r.last_values, r.timeout_values, np.array([True, False]))
    masked, _, adv_self, _ = MG.blended_advantages(*args, valid=valid)
    plain, _, _, _ = MG.blended_advantages(*args)
    assert torch.equal(masked[:7, 0], adv_self[:7, 0])
    assert float((plain[:7, 0] - adv_self[:7, 0]).min()) > .3  # the spurious step the mask removes
    assert torch.equal(masked[7:, 0], plain[7:, 0]) and torch.equal(masked[:, 1], adv_self[:, 1])


def test_explained_variance():
    returns = torch.tensor([[1., 2.], [2., 2.], [3., 2.], [4., 2.]])
    ev = MG.explained_variance(returns.clone(), returns)
    assert float(ev[0]) == pytest.approx(1.) and torch.isnan(ev[1])
    assert float(MG.explained_variance(torch.zeros(4, 1) + 2.5, returns[:, :1])[0]) == pytest.approx(0.)


def test_donor_selection_rules():
    teams = np.array([0, 1, 0, 1, 0, 1])
    contact = np.zeros((6, 6), bool)
    for i, j in [(0, 1), (0, 2), (0, 3), (1, 3), (4, 5)]:
        contact[i, j] = contact[j, i] = True
    np.fill_diagonal(contact, True)
    oracle = MG.select_donors("oracle", teams, contact, None)
    assert oracle[0].tolist() == [False, False, True, False, False, False] and not oracle[4].any()
    everyone = MG.select_donors("all", teams, contact, None)
    assert not everyone.diagonal().any() and everyone[0, [1, 2, 3]].all()
    k = MG.select_donors("allk", teams, contact, np.random.default_rng(5))
    assert (k.sum(1) == oracle.sum(1)).all() and not (k & ~everyone).any()
    assert (k == MG.select_donors("allk", teams, contact, np.random.default_rng(5))).all()


def test_donor_values_follow_the_receivers_stream_with_resets_and_timeouts():
    torch.manual_seed(0)
    net = T.RecurrentActorCritic(obs_dim=6, action_dim=5, mlp_hidden_dim=8, recurrent_hidden_dim=4)
    obs = torch.randn(10, 2, 6)
    dones = torch.zeros(10)
    dones[4] = 1
    timeout = {4: torch.randn(2, 6)}
    last = torch.randn(2, 6)
    values, v_timeout, v_last = MG.donor_values(net, obs, dones, timeout, last, torch.device("cpu"))
    with torch.no_grad():
        for b in range(2):
            h = torch.zeros(1, 4)
            for t in range(10):
                v, h = net._critic_step(obs[t, b:b + 1], h)
                assert float(values[t, b]) == pytest.approx(float(v), abs=1e-6)
                if t == 4:
                    vt, _ = net._critic_step(timeout[4][b:b + 1], h)
                    assert float(v_timeout[t, b]) == pytest.approx(float(vt), abs=1e-6)
                    h = torch.zeros_like(h)
            vl, _ = net._critic_step(last[b:b + 1], h)
            assert float(v_last[b]) == pytest.approx(float(vl), abs=1e-6)
    assert not v_timeout[[0, 1, 2, 3, 5, 6, 7, 8, 9]].any()


def test_warm_mask_waits_for_the_window_or_a_reset():
    valid, last = MG.warm_mask(torch.zeros(6), 3)
    assert valid.tolist() == [0, 0, 1, 1, 1, 1] and last
    dones = torch.zeros(6)
    dones[0] = 1
    valid, _ = MG.warm_mask(dones, 5)
    assert valid.tolist() == [0, 1, 1, 1, 1, 1]


class Constant:
    def __init__(self, value):
        self.value = value

    def initial_hidden(self, batch, device):
        return torch.zeros(batch, 1), torch.zeros(batch, 1)

    def _critic_step(self, obs, h):
        return torch.full((obs.shape[0],), self.value), h


@pytest.mark.parametrize("cap, expected", [(None, .8), (.5, .5)])
def test_uplift_only_for_receivers_with_donors_after_warmup(cap, expected):
    r = rollout(steps=6, n=3)
    r.values[:] = .2
    agents = [Constant(0.), Constant(5.), Constant(1.)]
    donors = np.zeros((3, 3), bool)
    donors[0, 2] = True
    config = MG.MaxGAEConfig(cap=cap, window=2)
    f_values, f_last, f_timeout, active, b, _ = MG.max_targets(agents, r, donors, {}, torch.zeros(3, 3), config,
                                                             torch.device("cpu"))
    assert active.tolist() == [True, False, False]
    assert b[:, 0].tolist() == pytest.approx([0.] + [expected] * 5)
    assert torch.equal(f_values[:, 1:], r.values[:, 1:]) and torch.equal(f_timeout, r.timeout_values)
    assert float(f_last[0] - r.last_values[0]) == pytest.approx(max(0., min(1. - float(r.last_values[0]),
                                                                            cap or 9.)))


def test_install_refuses_team_reward():
    cfg = SimpleNamespace(env_kwargs_json=json.dumps(dict(team_reward_mode="team")), gae_lambda=.95, sensor_range=5)
    with pytest.raises(ValueError, match="individual"):
        with MG.install(cfg):
            pass


def max_plan(tmp_path, **overrides):
    from latent_team.train_from_scratch import plan
    from latent_team.common import REFERENCE_EXPERIMENT
    if not (REFERENCE_EXPERIMENT / "experiment.json").exists():
        pytest.skip("Local reference configuration missing")
    ckpt = tmp_path / "checkpoint_01000.pt"
    # Layout of the real A+C IPPO s1 checkpoint (seed 1: team 0 bottom-left, team 1 bottom-right).
    torch.save(dict(effective_env_kwargs=dict(team_goal_positions=[[[0, 10]], [[9, 10]]])), ckpt)
    args = dict(reference_experiment=REFERENCE_EXPERIMENT, output_dir=tmp_path/"plan", maps=["e1"], seeds=[1],
                modes=["ippo", "oracle_max", "oracle_maxd1", "all_max", "allk_max"], timesteps=64, home_ratio=.5,
                learning_rate=2e-4, entropy_coef=.05, save_interval=0, track=False, save_video=False,
                smoke_test=True, stage_label="max_test", wandb_project="test", wandb_mode="offline",
                env_variant="b6_corner_indiv_pen005", team_reward="individual", probe_bank="canonical",
                eval_seed=5000001, reward_probe_ac=True, ac_gate="v1", init_checkpoint=ckpt, transfer_warmup=0,
                max_gae_eta=.5, max_gae_scale="shrink", max_gae_lambda=None, max_gae_cap=None)
    args.update(overrides)
    return plan(SimpleNamespace(**args))


def test_plan_max_modes_branch_from_checkpoint(tmp_path):
    _, manifest = max_plan(tmp_path)
    jobs = {j["mode"]: j for j in manifest["jobs"]}
    assert all("_ac_br01000_" in j["name"] for j in manifest["jobs"])
    assert manifest["init_checkpoint"].endswith("checkpoint_01000.pt") and manifest["transfer_warmup_override"] == 0
    assert manifest["initialization"].startswith("branch")
    assert "max_gae" not in jobs["ippo"]
    assert (jobs["oracle_max"]["max_gae"]["donors"], jobs["oracle_max"]["max_gae"]["eta"]) == ("oracle", .5)
    assert jobs["oracle_maxd1"]["max_gae"]["eta"] == 1.0 and jobs["allk_max"]["max_gae"]["donors"] == "allk"
    assert jobs["oracle_maxd1"]["max_gae"]["scale"] == "raw" and jobs["oracle_max"]["max_gae"]["scale"] == "shrink"
    for mode, job in jobs.items():
        cfg = job["config"]
        assert cfg["init_checkpoint"].endswith("checkpoint_01000.pt") and job["transfer_warmup_updates"] == 0
        assert not cfg["communication_enabled"] and cfg["pgct_peer_loss_coef"] == 0  # same trainer path as IPPO
        assert job["canonical_bank"] is None
        assert json.loads(cfg["env_kwargs_json"])["team_reward_mode"] == "individual"


def test_branch_plan_refuses_a_seed_with_swapped_goal_sides(tmp_path):
    _, manifest = max_plan(tmp_path, seeds=[1, 3], modes=["ippo", "oracle_max"])  # odd seeds keep the layout
    assert {j["seed"] for j in manifest["jobs"]} == {1, 3}
    with pytest.raises(ValueError, match="goal layout"):
        max_plan(tmp_path, output_dir=tmp_path / "plan_even", seeds=[1, 2], modes=["ippo"])
    assert not (tmp_path / "plan_even").exists()  # refused before anything is written


def test_plan_refuses_team_reward_for_max_modes(tmp_path):
    with pytest.raises(ValueError):
        max_plan(tmp_path, reward_probe_ac=False, team_reward="team", init_checkpoint=None)
