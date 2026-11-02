"""PGCT-D2 approval, contact/maturity isolation, runner wiring and 64-step trainer checks."""
import json
from pathlib import Path
from types import SimpleNamespace
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from latent_team import max_gae as MG
from latent_team import reward_probe_ac_v2 as V2
from latent_team import train_from_scratch as S
from examples import train_recurrent_ippo_consensus as T
from test_max_gae import max_plan, rollout


def evidence_controller():
    ctrl = V2.ACV2Controller(3)
    flags = [True, True, True]
    ctrl.predictors = [SimpleNamespace(mature=lambda i=i: flags[i], events_seen=20) for i in range(3)]
    ctrl.rounds[0, 1], ctrl.ema[0, 1] = 3, .2
    ctrl.rounds[0, 2], ctrl.ema[0, 2] = 2, .1
    ctrl.rounds[1, 2], ctrl.ema[1, 2] = 3, .5
    ctrl.gate[0, 1] = ctrl.gate[1, 0] = True
    return ctrl, flags


def test_approval_survives_lost_contact_but_rechecks_maturity_and_evidence():
    ctrl, flags = evidence_controller()
    expected = np.array([[0, 1, 0], [1, 0, 0], [0, 0, 0]], dtype=bool)
    assert np.array_equal(ctrl.approved(), expected)
    ctrl(1, np.zeros((3, 3), bool), active=True)  # not a refit round: legacy gate clears
    assert not ctrl.gate.any() and np.array_equal(ctrl.approved(), expected)
    result = ctrl.approved()
    result[:] = False
    assert np.array_equal(ctrl.approved(), expected)  # returned copy
    flags[0] = False
    assert not ctrl.approved().any() and ctrl.maturity().tolist() == [False, True, True]
    flags[0] = True
    ctrl.ema[0, 1] = .51
    assert not ctrl.approved().any()


def snapshot_controller(snapshot=True):
    cfg = V2.ACV2Config(interval=1, batch_size=4, train_steps=1, min_events=1, min_event_response=-1.0,
                        event_reward=.1, min_rounds=1, distance_threshold=1.0, probes_per_agent=4, min_probes=1,
                        contact_snapshot=snapshot)
    return V2.ACV2Controller(2, cfg)


def feed(ctrl, rng, steps):
    size = V2.view_size(ctrl.config.radius)
    for _ in range(steps):
        before, after = (rng.integers(0, 2, size=(2, size)).astype(np.uint8) for _ in range(2))
        ctrl.observe_views(before, rng.integers(0, 5, size=2), after, rng.choice([0., .2, 1.], size=2))


def test_contact_snapshot_compares_the_rollout_start_state_then_refits():
    rng = np.random.default_rng(0)
    ctrl = snapshot_controller()
    contact = np.ones((2, 2), bool)
    assert not ctrl.maturity().any()  # nothing frozen yet
    ctrl.snapshot()                   # rollout 1 starts
    feed(ctrl, rng, 12)
    ctrl(1, contact, active=True)     # immature at the rollout start: no comparison, then refit
    assert ctrl.last_log["ac_pairs"][0]["reason"] == "immature"
    assert [p.version for p in ctrl.predictors] == [1, 1]
    ctrl.snapshot()                   # rollout 2 starts: this is what either agent can send on a contact
    b, a, f = ctrl.probes(0, 1)
    expected = V2.response_distance(ctrl.predictors[0].predict(b, a, f), ctrl.predictors[1].predict(b, a, f))
    frozen_mature = ctrl.maturity().copy()
    feed(ctrl, rng, 12)               # experience after the contact
    assert np.array_equal(ctrl.maturity(), frozen_mature) and frozen_mature.all()
    ctrl(2, contact, active=True)
    row = ctrl.last_log["ac_pairs"][0]
    assert row["distance"] == pytest.approx(expected, abs=1e-12) and ctrl.last_versions[0, 1] == (1, 1)
    assert [p.version for p in ctrl.predictors] == [2, 2]  # refitted only after the comparison
    live = [p.event_probes() for p in ctrl.predictors]
    lb, la, lf = (np.concatenate(x) for x in zip(*live))
    after_refit = V2.response_distance(ctrl.predictors[0].predict(lb, la, lf), ctrl.predictors[1].predict(lb, la, lf))
    assert after_refit != pytest.approx(expected, abs=1e-6)  # the old order would have used other information


def test_contact_snapshot_refuses_a_refit_between_snapshot_and_comparison():
    rng = np.random.default_rng(1)
    ctrl = snapshot_controller()
    ctrl.snapshot()
    feed(ctrl, rng, 12)
    ctrl(1, np.ones((2, 2), bool), active=True)
    ctrl.snapshot()
    ctrl.predictors[0].train()
    with pytest.raises(RuntimeError, match="snapshot"):
        ctrl(2, np.ones((2, 2), bool), active=True)


def test_plan_freezes_contact_state_only_for_pgct_d2_modes(tmp_path):
    _, manifest = max_plan(tmp_path, modes=["pgct", "pgct_max", "oracle_mature_max"], ac_gate="v2",
                           init_checkpoint=None, transfer_warmup=None)
    flags = {j["mode"]: j["reward_probe_ac_v2"]["contact_snapshot"] for j in manifest["jobs"]}
    assert flags == {"pgct": False, "pgct_max": True, "oracle_mature_max": True}


def test_run_job_refuses_a_pgct_d2_job_planned_without_the_snapshot(tmp_path):
    output, manifest = max_plan(tmp_path, modes=["pgct_max"], ac_gate="v2", init_checkpoint=None,
                                transfer_warmup=None)
    manifest["jobs"][0]["reward_probe_ac_v2"]["contact_snapshot"] = False
    with pytest.raises(ValueError, match="re-plan"):
        S.run_job(output, manifest, manifest["jobs"][0]["name"], manifest["eval_seed"])


def test_real_trainer_natural_approval(tmp_path):
    """Permissive v2 settings (wiring only, not research values): real compare -> approve -> donor path."""
    checkpoint = S.C.ROOT / "runs/ac_20261023_185556/b6_corner_indiv_pen005_ac_ippo_s1/checkpoint_01000.pt"
    if not checkpoint.is_file():
        pytest.skip("Local 1M checkpoint missing")
    options = tmp_path / "ac.json"
    options.write_text(json.dumps(dict(interval=1, batch_size=16, train_steps=2, min_events=1,
                                       min_event_response=-1.0, event_reward=.001, min_rounds=1,
                                       distance_threshold=1.0, min_probes=1)))
    output, manifest = max_plan(tmp_path, modes=["pgct_max"], ac_gate="v2", ac_config=options,
                                init_checkpoint=checkpoint)
    job = manifest["jobs"][0]
    S.run_job(output, manifest, job["name"], manifest["eval_seed"])
    folder = output / job["name"]
    rows = [json.loads(s) for s in (folder / "max_gae.jsonl").read_text().splitlines()]
    compared = [p for r in rows for p in r["ac_pairs"] if p["distance"] is not None]
    assert compared  # a real comparison happened on rollout contacts
    for r in rows:
        contact = np.asarray(r["contact"], bool) & ~np.eye(6, dtype=bool)
        assert np.array_equal(np.asarray(r["donors"], bool), contact & np.asarray(r["approved"], bool))
    assert any(np.asarray(r["donors"]).any() for r in rows)
    assert all(not np.asarray(json.loads(s)["actual_allocation"]).any()
               for s in (folder / "transfer.jsonl").read_text().splitlines())
    state = torch.load(folder / "final.pt", weights_only=False)["reward_probe_ac_v2"]
    # Compared at update 2 with the version used during rollout 2 (1), then refitted to 2.
    assert all(v == (1, 1) for v in state["last_versions"].values())
    assert [p["version"] for p in state["predictors"]] == [2] * 6
    assert state["config"]["contact_snapshot"]


def test_gate_donors_use_approval_and_contacts_without_team_labels():
    contact = np.ones((3, 3), bool)
    approval = np.array([[0, 1, 1], [1, 0, 0], [1, 0, 0]], dtype=bool)
    contact[0, 2] = contact[2, 0] = False
    a = MG.select_donors("gate", [0, 0, 0], contact, None, approved=approval)
    b = MG.select_donors("gate", [0, 1, 2], contact, None, approved=approval)
    assert np.array_equal(a, b) and a.tolist() == [[False, True, False], [True, False, False], [False]*3]
    assert approval[0, 2] and contact.diagonal().all()  # inputs untouched
    filtered = MG.select_donors("gate", [0, 1, 2], contact, None, approved=approval,
                                 critic_mature=[False, True, True])
    assert filtered[0, 1] and not filtered[1].any()  # critic condition is donor-only


def test_oracle_mature_requires_both_reward_predictors_but_only_donor_critic():
    teams = [0, 1, 0, 1]
    contact = np.ones((4, 4), bool)
    donors = MG.select_donors("oracle_mature", teams, contact, None, mature=[True, False, True, True],
                               critic_mature=[False, True, True, True])
    assert donors.sum() == 1 and donors[0, 2]
    assert np.array_equal(MG.select_donors("oracle_mature", teams, contact, None, mature=[True]*4),
                          MG.select_donors("oracle", teams, contact, None))


@pytest.mark.parametrize("mode, kwargs", [("gate", {}), ("gate", {"approved": np.ones((2, 2))}),
                                         ("oracle_mature", {}), ("unknown", {})])
def test_donor_rules_fail_closed_for_missing_or_invalid_inputs(mode, kwargs):
    with pytest.raises(ValueError):
        MG.select_donors(mode, [0, 1, 0], np.ones((3, 3), bool), None, **kwargs)


@pytest.mark.parametrize("mode", ["gate", "oracle_mature"])
def test_new_install_requires_a_controller(mode):
    with pytest.raises(ValueError, match="controller"):
        with MG.install(SimpleNamespace(), config=MG.MaxGAEConfig(donors=mode)):
            pass


@pytest.mark.parametrize("overrides", [{"reward_probe_ac": False}, {"ac_gate": "v1"}])
def test_new_modes_refuse_non_v2_plans_before_creating_folders(tmp_path, overrides):
    with pytest.raises(ValueError, match="require.*v2"):
        max_plan(tmp_path, modes=["pgct_max", "oracle_mature_max"], **overrides)
    assert not (tmp_path / "plan").exists()


def test_new_plan_keeps_ippo_path_and_records_optional_critic_filter(tmp_path):
    _, manifest = max_plan(tmp_path, modes=["pgct_max", "oracle_mature_max"], ac_gate="v2",
                            max_gae_min_critic_ev=.1)
    assert [j["max_gae"]["donors"] for j in manifest["jobs"]] == ["gate", "oracle_mature"]
    for job in manifest["jobs"]:
        assert job["max_gae"]["min_critic_ev"] == .1 and job["max_gae"]["scale"] == "shrink"
        assert job["config"]["pgct_peer_loss_coef"] == 0 and not job["config"]["communication_enabled"]
        assert job["reward_probe_ac_v2"] and job["canonical_bank"] is None


def test_critic_filter_rejects_nonfinite_config():
    with pytest.raises(ValueError, match="finite"):
        MG.MaxGAEConfig(min_critic_ev=float("nan"))


def test_copied_critic_gives_negligible_uplift_with_resets():
    torch.manual_seed(37)
    net = T.RecurrentActorCritic(obs_dim=3, action_dim=5, mlp_hidden_dim=8, recurrent_hidden_dim=4)
    r = rollout(steps=32)
    r.dones[15] = 1
    r.obs = torch.rand(32, 2, 3)
    last_obs = torch.rand(2, 3)
    r.values, r.timeout_values, r.last_values = MG.donor_values(net, r.obs, r.dones, {}, last_obs, "cpu")
    donors = np.array([[False, True], [True, False]])
    f, last, timeout, active, _, _ = MG.max_targets([net, net], r, donors, {}, last_obs,
                                                   MG.MaxGAEConfig(), "cpu")
    valid, _ = MG.warm_mask(r.dones, 8)
    for scale in ("raw", "shrink"):
        actor, _, own, _ = MG.blended_advantages(r, .99, .95, .95, .5, f, last, timeout, active,
                                                 scale=scale, valid=valid)
        assert torch.allclose(actor, own, atol=1e-6, rtol=0)


def test_real_trainer_closed_gate_and_oracle_injection_are_bit_identical(tmp_path, monkeypatch):
    """Diagnostic bypass only in this test; production gate has no label access. No long training."""
    checkpoint = S.C.ROOT / "runs/ac_20261023_185556/b6_corner_indiv_pen005_ac_ippo_s1/checkpoint_01000.pt"
    if not checkpoint.is_file():
        pytest.skip("Local 1M checkpoint missing")
    ac_options = tmp_path / "ac_options.json"
    # Actually refit v2 during both updates, without collecting enough reward events to mature.
    ac_options.write_text(json.dumps(dict(interval=1, batch_size=16, train_steps=2, min_events=1000000)))
    output, manifest = max_plan(tmp_path, modes=["ippo", "pgct_max", "oracle_mature_max", "oracle_max"],
                                ac_gate="v2", ac_config=ac_options, init_checkpoint=checkpoint)
    originals = {k: getattr(T, k) for k in ("make_env", "collect_rollout", "update_ippo", "compute_gae")}
    saved = {}
    for job in manifest["jobs"]:
        S.run_job(output, manifest, job["name"], manifest["eval_seed"])
        saved[job["mode"]] = torch.load(output / job["name"] / "final.pt", weights_only=False)
    def identical(a, b):
        return all(torch.equal(x[k], y[k]) for x, y in zip(a["agents"], b["agents"]) for k in x)
    assert identical(saved["ippo"], saved["pgct_max"])
    assert identical(saved["ippo"], saved["oracle_mature_max"])
    assert all(p["version"] == 2 for p in saved["pgct_max"]["reward_probe_ac_v2"]["predictors"])

    for job in manifest["jobs"]:
        if job["mode"] not in S.AC_MAX_MODES:
            continue
        folder = output / job["name"]
        rows = [json.loads(s) for s in (folder / "max_gae.jsonl").read_text().splitlines()]
        assert len(rows) == 2 and all(not np.asarray(r["donors"]).any() for r in rows)
        assert rows[-1]["same_team_possible_arcs"] == 12 and rows[-1]["reward_mature"] == [False]*6
        assert all(not np.asarray(json.loads(s)["actual_allocation"]).any()
                   for s in (folder / "transfer.jsonl").read_text().splitlines())

    with S.EC.agent_count_support(6):
        env = T.make_env(job["config"]["env_id"], saved["oracle_max"]["effective_env_kwargs"], "semantic")
        try:
            env.reset(seed=1)
            teams = np.asarray(env.unwrapped.get_oracle_team_assignments())
        finally:
            env.close()
    oracle = (teams[:, None] == teams[None, :]) & ~np.eye(len(teams), dtype=bool)
    monkeypatch.setattr(V2.ACV2Controller, "approved", lambda self: oracle.copy())
    monkeypatch.setattr(V2.ACV2Controller, "maturity", lambda self: np.ones(len(teams), bool))
    injected_output, injected = max_plan(tmp_path, output_dir=tmp_path / "injected",
                                        modes=["pgct_max", "oracle_mature_max"],
                                        ac_gate="v2", ac_config=ac_options, init_checkpoint=checkpoint)
    for job in injected["jobs"]:
        S.run_job(injected_output, injected, job["name"], injected["eval_seed"])
        payload = torch.load(injected_output / job["name"] / "final.pt", weights_only=False)
        assert identical(saved["oracle_max"], payload)
        rows = [json.loads(s) for s in (injected_output / job["name"] / "max_gae.jsonl").read_text().splitlines()]
        assert any(r["max_gae_donor_arcs"] > 0 and r["max_gae_adv_shift"] > 0 for r in rows)
        assert all(r["max_gae_cross_team_arcs"] == 0 and r["max_gae_same_team_approval_coverage"] == 1 for r in rows)

    filtered_output, filtered = max_plan(tmp_path, output_dir=tmp_path / "filtered",
                                        modes=["pgct_max", "oracle_mature_max"], ac_gate="v2", ac_config=ac_options,
                                        init_checkpoint=checkpoint, max_gae_min_critic_ev=1e9)
    for job in filtered["jobs"]:
        S.run_job(filtered_output, filtered, job["name"], filtered["eval_seed"])
        payload = torch.load(filtered_output / job["name"] / "final.pt", weights_only=False)
        assert identical(saved["ippo"], payload)
        rows = [json.loads(s) for s in (filtered_output / job["name"] / "max_gae.jsonl").read_text().splitlines()]
        assert all(r["critic_ev_filter_enabled"] and not any(r["critic_mature"])
                   and not np.asarray(r["donors"]).any() and np.asarray(r["approved"]).any() for r in rows)

    held_output, held = max_plan(tmp_path, output_dir=tmp_path / "held", modes=["pgct_max"],
                                ac_gate="v2", ac_config=ac_options, init_checkpoint=checkpoint, transfer_warmup=1)
    job = held["jobs"][0]
    S.run_job(held_output, held, job["name"], held["eval_seed"])
    rows = [json.loads(s) for s in (held_output / job["name"] / "max_gae.jsonl").read_text().splitlines()]
    assert rows[0]["held"] and not np.asarray(rows[0]["donors"]).any()
    assert not rows[1]["held"] and np.asarray(rows[1]["donors"]).any()
    assert all(getattr(T, k) is v for k, v in originals.items())
