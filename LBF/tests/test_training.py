"""Generic trainer on multi-team LBF: every method runs, methods share initial weights, the label rule holds,
the gate methods reduce exactly to IPPO / Oracle, and the shared command lines (plan, evaluate, video,
aggregate) work end to end."""
import json

import numpy as np
import pytest
import torch

from pgct_core import reward_gate as RG
from pgct_core.trainer import METHODS, run_training
from pgct_lbf import aggregate as AG
from pgct_lbf import evaluate as EV
from pgct_lbf import train as TR
from pgct_lbf import video as VI
from pgct_lbf.adapter import ForagingAdapter
from pgct_lbf.experiment import SMOKE, apply_overrides, build, load_config

TINY = list(SMOKE) + ["trainer.total_timesteps=128", "trainer.eval_interval=0", "trainer.log_interval=0"]


def job(tmp_path, method, extra=(), name=None, config="main_4p2t.json"):
    data = apply_overrides(load_config(config), TINY + list(extra))
    make_adapter, cfg, ppo, gate, mg, resolved = build(data, method, 1)
    out = tmp_path / (name or method)
    run_training(make_adapter, out, cfg, ppo, gate, mg)
    return out, torch.load(out / "final.pt", weights_only=False)


def same_weights(a, b):
    return all(torch.equal(x[k], y[k]) for x, y in zip(a["agents"], b["agents"]) for k in x)


@pytest.mark.parametrize("config", ["main_4p2t.json", "ind_4p2t.json", "coop_4p2t.json"])
def test_configs_build(config):
    make_adapter, cfg, ppo, gate, mg, resolved = build(load_config(config), "pgct_d2", 2)
    adapter = make_adapter()
    assert adapter.n_agents == 4 and adapter.obs_dim == 100 and adapter.projection.view_size == 100
    assert gate.event_threshold == "relative" and gate.distance_mode == "action_max" and not gate.mirror
    assert resolved["foraging"]["team_seed"] == 2


def test_every_method_runs_and_shares_the_initial_weights(tmp_path):
    hashes = set()
    for method in METHODS:
        out, payload = job(tmp_path, method)
        hashes.add((out / "initial_weights_sha256.txt").read_text())
        rows = [json.loads(s) for s in (out / "metrics.jsonl").read_text().splitlines()]
        assert len(rows) == 2 and all("max_gae_donor_arcs" in r and "cost_critic_bytes" in r for r in rows)
        gate_rows = [json.loads(s) for s in (out / "gate.jsonl").read_text().splitlines()]
        assert ("approved" in gate_rows[0]) == METHODS[method]["gate"]
        assert (payload["reward_gate"] is not None) == METHODS[method]["gate"]
        assert json.loads((out / "status.json").read_text())["status"] == "complete"
    assert len(hashes) == 1


def test_non_oracle_methods_never_read_team_labels_without_audit(tmp_path, monkeypatch):
    def forbidden(self):
        raise AssertionError("ground-truth teams read by a non-oracle training path")
    monkeypatch.setattr(ForagingAdapter, "oracle_team_ids", forbidden)
    for method in sorted(set(METHODS) - {"oracle"}):
        job(tmp_path, method, ["trainer.audit=false"])
    with pytest.raises(AssertionError, match="ground-truth"):
        job(tmp_path, "oracle", ["trainer.audit=false"], name="oracle_forbidden")


def test_gate_methods_reduce_to_ippo_or_oracle_exactly(tmp_path, monkeypatch):
    closed = ["gate.min_events=1000000"]  # predictors refit but never mature: no approval
    _, ippo = job(tmp_path, "ippo")
    for method in ("pgct_d2", "pgct_id_only", "random_gate"):
        _, payload = job(tmp_path, method, closed, name=f"{method}_closed")
        assert same_weights(ippo, payload), method  # the gate never perturbs the training streams
    _, oracle = job(tmp_path, "oracle")
    teams = build(apply_overrides(load_config("main_4p2t.json"), TINY), "oracle", 1)[0]().oracle_team_ids()
    oracle_matrix = (teams[:, None] == teams[None, :]) & ~np.eye(4, dtype=bool)
    monkeypatch.setattr(RG.RewardProbeGate, "approved", lambda self: oracle_matrix.copy())
    monkeypatch.setattr(RG.RewardProbeGate, "maturity", lambda self: np.ones(4, bool))
    out, injected = job(tmp_path, "pgct_d2", closed, name="pgct_injected")
    assert same_weights(oracle, injected)
    rows = [json.loads(s) for s in (out / "metrics.jsonl").read_text().splitlines()]
    assert any(r["max_gae_donor_arcs"] > 0 for r in rows) and all(r["max_gae_cross_team_arcs"] == 0 for r in rows)


def test_command_lines_end_to_end(tmp_path):
    common = ["--config", "main_4p2t.json", "--seeds", "1", "--name", "p", "--output-root", str(tmp_path), "--smoke"]
    assert TR.main(common + ["--methods", "ippo", "pgct_d2", "--plan-only"]) == 0
    plan = json.loads((tmp_path / "p" / "plan.json").read_text())
    assert plan["experiment"] == "LBF" and plan["jobs"][0]["config"]["foraging"]["team_seed"] == 1
    with pytest.raises(SystemExit):  # same plan name, different configuration
        TR.main(["--config", "ind_4p2t.json", "--seeds", "1", "--name", "p", "--output-root", str(tmp_path),
                 "--smoke", "--methods", "ippo", "--plan-only"])
    pytest.importorskip("imageio")
    assert TR.main(common + ["--methods", "pgct_d2", "--video-every", "1"]) == 0
    job_dir = tmp_path / "p" / "pgct_d2_s1"
    assert (job_dir / "final.pt").exists() and (job_dir / "videos" / "final.mp4").exists()
    assert not (job_dir / "video_error.json").exists()
    assert TR.main(common + ["--methods", "pgct_d2"]) == 0          # finished job is skipped
    assert EV.main(["--job", str(job_dir), "--episodes", "2"]) == 0
    assert (job_dir / "eval_final_seed7000001_det.json").exists()
    with pytest.raises(SystemExit):                                  # never overwrites
        EV.main(["--job", str(job_dir), "--episodes", "2"])
    assert VI.main(["--job", str(job_dir), "--max-steps", "5"]) == 0
    assert (job_dir / "videos" / "final_seed5000001_det.mp4").exists()
    assert AG.main(["--plan", str(tmp_path / "p")]) == 0
    summary = json.loads((tmp_path / "p" / "summary.json").read_text())
    assert {j["job"] for j in summary["jobs"]} == {"pgct_d2_s1"}
    assert "final_eval_collected_own" in summary["jobs"][0] and summary["jobs"][0]["status"] == "complete"
