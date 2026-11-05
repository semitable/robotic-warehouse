import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from examples import train_recurrent_ippo_consensus as T
from examples.diagnose_pgct_rollout_compatibility import compare_histories, environment_signature, model_hash, summarize


def test_common_history_replay_matches_identical_networks_without_changing_weights_or_rng():
    agents = [T.RecurrentActorCritic(3, 5, 4, 4) for _ in range(4)]
    for net in agents[1:]:
        net.load_state_dict(agents[0].state_dict())
    cache = dict(observations=np.random.default_rng(4).normal(size=(6, 4, 3)).astype(np.float32),
                 action_masks=np.tile([1, 1, 0, 1, 0], (6, 4, 1)).astype(np.float32),
                 time_ids=np.array([0, 1, 2, 0, 1, 2]), episode_ids=np.array([0, 0, 0, 1, 1, 1]),
                 phases=np.zeros((6, 4), np.int8))
    before = model_hash(agents)
    rng = torch.get_rng_state().clone()
    result = compare_histories(agents, cache, sample_interval=2)
    assert torch.equal(rng, torch.get_rng_state())
    assert before == model_hash(agents)
    np.testing.assert_array_equal(result['observation_indices'], [0, 2, 3, 5])
    np.testing.assert_allclose(result['recurrent_masked'], 0, atol=1e-8)
    np.testing.assert_allclose(result['critic_squared'], 0)
    summary = summarize(result, cache)
    assert summary['sampled_observation_stream_instances'] == 16
    assert summary['descriptive_pair_mean_pearson'] is None


def test_matched_environment_defaults_and_ippo_communication_do_not_mask_task_changes():
    a = dict(sensor_range=5, shelf_soft_zone_ratio=.5, communication_enabled=True,
             communication_topology='physical', communication_range=5)
    b = dict(a, shelf_rows=1, shelf_columns=3, column_height=8, communication_enabled=False,
             communication_topology='none')
    assert environment_signature(a) == environment_signature(b)
    b['shelf_soft_zone_ratio'] = .75
    assert environment_signature(a) != environment_signature(b)
