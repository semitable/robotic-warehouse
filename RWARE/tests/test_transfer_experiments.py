import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from examples import train_recurrent_ippo_consensus as T
from examples.pgct_probe_diagnostics import common_probe_response, finite_horizon_returns, js_divergence, PAIRS, critic_rollouts
from examples.train_pgct_transfer_experiments import ablation_allocation


def entropy_form_js(p, q):
    def entropy(a):
        return -(a * np.log(np.maximum(a, 1e-300))).sum(-1)
    return entropy((p + q) / 2) - (entropy(p) + entropy(q)) / 2


def test_js_matches_independent_entropy_formula_and_known_limits():
    p = np.array([[1., 0, 0], [.1, .3, .6]])
    q = np.array([[0., 1, 0], [.7, .2, .1]])
    np.testing.assert_allclose(js_divergence(p, q), entropy_form_js(p, q))
    np.testing.assert_allclose(js_divergence(p, p), 0)
    assert js_divergence(p, q)[0] == pytest.approx(np.log(2))


def test_actual_trainer_js_matches_independent_reference_and_preserves_rng():
    agents = [T.RecurrentActorCritic(3, 5, 4, 4) for _ in range(4)]
    with torch.no_grad():
        for i, net in enumerate(agents):
            net.actor_head.bias[i] += 3
    probes = torch.randn(6, 2, 3)
    probes[5] = probes[0]  # duplicate weighting must be visible, not alter training gate
    before = torch.get_rng_state().clone()
    response = common_probe_response(agents, probes, torch.device("cpu"))
    actual, _ = T.policy_distance_similarity_matrices(agents, probes, torch.device("cpu"), .25)
    np.testing.assert_allclose(actual, response["mean_js"], atol=2e-7)
    assert response["unique_probe_count"] == 5
    assert torch.equal(before, torch.get_rng_state())
    p = np.asarray(response["probabilities"])
    for k, (i, j) in enumerate(PAIRS):
        np.testing.assert_allclose(response["js_per_probe"][k],
                                   entropy_form_js(p[i], p[j]).mean(1), atol=1e-12)


def test_random_matches_receiver_budget_and_is_restricted_to_neighbors():
    neighbors = np.ones((4, 4), bool)
    np.fill_diagonal(neighbors, False)
    reference = np.array([[0, .2, .8, 0], [0, 0, 0, 0], [.5, .5, 0, 0], [0, 0, 1, 0]], np.float32)
    result, exposure = ablation_allocation("random", np.zeros((4, 4)), neighbors, np.random.default_rng(7), reference)
    np.testing.assert_allclose(result.sum(1), reference.sum(1))
    np.testing.assert_array_equal((result > 0).sum(1), (reference > 0).sum(1))
    assert not result[~neighbors].any()
    assert sum(exposure["allocation_mass_shortfall"]) == 0
    # Under a donor shortage, preserve mass when possible and expose lost count.
    neighbors[0, 1] = neighbors[0, 2] = False
    result, exposure = ablation_allocation("random", np.zeros((4, 4)), neighbors, np.random.default_rng(7), reference)
    assert result[0, 3] == pytest.approx(1)
    assert exposure["donor_count_shortfall"][0] == 1
    neighbors[0, 3] = False
    _, exposure = ablation_allocation("random", np.zeros((4, 4)), neighbors, np.random.default_rng(7), reference)
    assert exposure["allocation_mass_shortfall"][0] == pytest.approx(1)


def test_edge_removal_and_refill_answer_different_dose_questions():
    neighbors = np.ones((4, 4), bool)
    np.fill_diagonal(neighbors, False)
    allocation = np.zeros((4, 4))
    allocation[2, 3] = .75
    allocation[2, 0] = .25
    allocation[3, 2] = 1
    removed, _ = ablation_allocation("drop34", allocation, neighbors, np.random.default_rng(1))
    assert removed[2, 0] == .25  # no renormalization / hidden dose compensation
    assert not removed[2, 3] and not removed[3, 2]
    refill, _ = ablation_allocation("drop34_refill", allocation, neighbors, np.random.default_rng(1))
    assert not refill[2, 3] and not refill[3, 2]
    np.testing.assert_allclose(refill.sum(1), allocation.sum(1))
    np.testing.assert_allclose(ablation_allocation("pgct", allocation, neighbors, np.random.default_rng(1))[0], allocation)


def test_finite_horizon_mc_target_has_no_implicit_bootstrap():
    np.testing.assert_allclose(finite_horizon_returns([[1, 0], [2, 3]], .5), [[2, 1.5], [2, 3]])


def test_stochastic_critic_diagnostic_preserves_model_and_training_rng():
    from types import SimpleNamespace
    cfg = SimpleNamespace(env_id="rware-multiteam-tiny-4ag-2teams-v0", observation_format="semantic",
                          recurrent_hidden_dim=4, action_mask=True, gamma=.99)
    agents = [T.RecurrentActorCritic(1593, 5, 4, 4) for _ in range(4)]
    weights = [{k: v.clone() for k, v in net.state_dict().items()} for net in agents]
    torch_before = torch.get_rng_state().clone()
    numpy_before = np.random.get_state()
    result = critic_rollouts(cfg, agents, dict(sensor_range=5, communication_enabled=False),
                             torch.device("cpu"), episodes=1, horizon=3, seed=5, make_env=T.make_env)
    assert torch.equal(torch_before, torch.get_rng_state())
    numpy_after = np.random.get_state()
    np.testing.assert_array_equal(numpy_before[1], numpy_after[1])
    assert numpy_before[2:] == numpy_after[2:]
    for saved, net in zip(weights, agents):
        assert all(torch.equal(saved[k], v) for k, v in net.state_dict().items())
    assert result["samples_per_agent"] == 3
    assert np.isfinite(result["metrics"]["critic_mc_mse"])
    np.testing.assert_allclose(np.diag(result["common_history_masked_js"]), 0)
