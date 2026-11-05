"""W&B adapter tests use a fake module and never create a network run."""
import json
import sys
from types import SimpleNamespace

import numpy as np

from pgct_overcooked.tracking import WandbTracker, scalar_payload


class FakeRun:
    def __init__(self):
        self.id = "offline-test-id"
        self.url = "https://wandb.invalid/offline-test-id"
        self.summary = {}
        self.defined = []
        self.logged = []
        self.exit_code = None

    def define_metric(self, *args, **kwargs):
        self.defined.append((args, kwargs))

    def log(self, row):
        self.logged.append(row)

    def finish(self, exit_code=0):
        self.exit_code = exit_code


def test_scalar_payload_flattens_agent_vectors_and_omits_nested_values():
    row = dict(env_steps=128, update=np.int64(2), method="pgct_d2", loss=np.float32(0.5),
               per_agent=[1.0, 2.0], nested=[[1], [2]])
    assert scalar_payload(row, "eval/") == {
        "env_steps": 128,
        "update": 2,
        "eval/loss": 0.5,
        "eval/per_agent/agent_0": 1.0,
        "eval/per_agent/agent_1": 2.0,
    }


def test_tracker_writes_descriptor_logs_and_finishes(tmp_path, monkeypatch):
    fake_run = FakeRun()
    calls = []

    def init(**kwargs):
        calls.append(kwargs)
        return fake_run

    monkeypatch.setitem(sys.modules, "wandb", SimpleNamespace(init=init))
    tracker = WandbTracker(tmp_path, project="pgct-overcooked", entity=None, group="pilot",
                           run_name="pilot_ippo_s1", method="ippo", seed=1, mode="offline",
                           config={"budget": 128}, tags=["test"])
    tracker("metrics", {"env_steps": 128, "episode_return_mean": 0.25})
    tracker("eval", {"env_steps": 128, "eval_return_mean": 0.5})
    tracker.finish("complete")

    descriptor = json.loads((tmp_path / "wandb.json").read_text(encoding="utf-8"))
    assert descriptor["group"] == "pilot" and descriptor["name"] == "pilot_ippo_s1"
    assert calls[0]["tags"] == ["test", "ippo", "seed1"]
    assert fake_run.logged[0]["episode_return_mean"] == 0.25
    assert fake_run.logged[1]["eval/eval_return_mean"] == 0.5
    assert fake_run.summary["status"] == "complete" and fake_run.exit_code == 0
