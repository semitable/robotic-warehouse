"""Follower experiments (D145): start chosen agents from a saved checkpoint and keep chosen agents fixed."""
import json

import pytest
import torch

from pgct_core import trainer
from pgct_core.ppo import PPOConfig
from pgct_core.reward_gate import RewardGateConfig

from toy_env import ToyAdapter

PPO = PPOConfig(rollout_steps=64, sequence_length=16, minibatch_chunks=2, ppo_epochs=1, mlp_hidden_dim=16,
                recurrent_hidden_dim=8)
GATE = RewardGateConfig(interval=1, batch_size=16, train_steps=5, min_events=1, min_event_response=-1.0,
                        min_rounds=1, min_probes=1, distance_threshold=1.0)


def run(path, method="pgct_d2", seed=3, **kw):
    cfg = trainer.TrainerConfig(method=method, seed=seed, total_timesteps=192, warmup_updates=0, eval_interval=0,
                                eval_episodes=1, save_interval=0, log_interval=0, **kw)
    trainer.run_training(ToyAdapter, path, cfg, PPO, GATE)
    return torch.load(path / "final.pt", weights_only=False)["agents"]


def same(a, b):
    return all(torch.equal(a[k], b[k]) for k in a)


def test_expert_is_loaded_and_kept_fixed_while_the_others_learn(tmp_path):
    source = run(tmp_path / "source", method="ippo", seed=11)
    after = run(tmp_path / "follow", init_checkpoint=str(tmp_path / "source" / "final.pt"), init_agents={"2": 0},
                frozen_agents=[2])
    assert same(after[2], source[0])                   # loaded from source agent 0 and never updated
    assert not any(same(after[i], source[0]) for i in (0, 1, 3))
    config = json.loads((tmp_path / "follow" / "config.json").read_text())
    assert config["trainer"]["init_agents"] == {"2": 0} and config["trainer"]["frozen_agents"] == [2]
    assert len(config["init_checkpoint_sha256"]) == 64


def test_initial_weights_of_unloaded_agents_match_a_plain_run(tmp_path):
    run(tmp_path / "source", method="ippo", seed=11)
    cfg = dict(init_checkpoint=str(tmp_path / "source" / "final.pt"), init_agents={1: 0})
    plain = run(tmp_path / "plain", method="ippo", seed=5)
    loaded = run(tmp_path / "loaded", method="ippo", seed=5, **cfg)
    assert not same(plain[1], loaded[1])
    # Without frozen agents and with nobody loaded, the run is exactly the default run.
    again = run(tmp_path / "again", method="ippo", seed=5, frozen_agents=[])
    assert all(same(a, b) for a, b in zip(plain, again))


def test_frozen_agent_without_loading_keeps_its_initial_weights(tmp_path):
    first = run(tmp_path / "frozen", method="ippo", seed=7, frozen_agents=[0])
    plain = run(tmp_path / "plain", method="ippo", seed=7)
    payload = torch.load(tmp_path / "frozen" / "final.pt", weights_only=False)
    torch.manual_seed(7)
    initial = trainer.RecurrentActorCritic(6, 3, 16, 8).state_dict()
    assert same(first[0], initial)
    assert not same(first[0], plain[0])
    assert payload["agents"][1] is not None


def test_config_validation():
    with pytest.raises(ValueError):
        trainer.TrainerConfig(init_agents={0: 0})
    with pytest.raises(ValueError):
        trainer.TrainerConfig(init_checkpoint="x.pt")
