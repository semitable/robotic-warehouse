"""VMAS Team Discovery rules, observation/probe separation, RNG isolation and the label boundary (needs vmas)."""
import numpy as np
import pytest

pytest.importorskip("vmas")
import torch  # noqa: E402

from pgct_vmas.adapter import AdapterConfig, DiscoveryAdapter, DiscoveryConfig  # noqa: E402

STAY = 0


def place(adapter, agents, targets):
    """Put agents / targets (dict index -> (x, y)) at fixed positions with zero velocity."""
    env = adapter.env
    for i, p in agents.items():
        env.agents[i].set_pos(torch.tensor([p], dtype=torch.float32), batch_index=0)
        env.agents[i].set_vel(torch.zeros(1, 2), batch_index=0)
    for k, p in targets.items():
        env.scenario._targets[k].set_pos(torch.tensor([p], dtype=torch.float32), batch_index=0)


def far_layout(adapter):
    """Agents in the four corners and every target far from all of them."""
    n_targets = len(adapter.env.scenario._targets)
    agents = {0: (-.9, -.9), 1: (.9, -.9), 2: (-.9, .9), 3: (.9, .9)}
    targets = {k: (-.5 + .2 * k, 0.0) for k in range(n_targets)}
    place(adapter, agents, targets)


def test_observation_layout_and_probe_view():
    adapter = DiscoveryAdapter(DiscoveryConfig(team_seed=1))
    obs = adapter.reset(seed=3)
    assert obs.shape == (4, 4 + 2 * 12 + 12) and obs.dtype == np.float32
    assert np.allclose(obs[:, :2], adapter.positions(), atol=1e-6)
    lidar = obs[:, 4:]
    assert lidar.min() >= 0 and lidar.max() <= .35 + 1e-6
    view = adapter.projection.public_view(obs[0])
    assert view.dtype == np.float32 and view.shape == (38,) and np.array_equal(view, obs[0, 2:])


def test_covering_pays_only_own_type_and_respawns():
    adapter = DiscoveryAdapter(DiscoveryConfig(team_sizes=(2, 2), team_assignment="fixed", targets_per_type=1))
    adapter.reset(seed=0)
    far_layout(adapter)
    place(adapter, {0: (0.0, 0.0)}, {0: (.15, 0.0), 1: (-.15, 0.0)})   # target 0: type 0 (agent 0's), 1: type 1
    _, rewards, terminated, truncated, info = adapter.step([STAY] * 4)
    assert np.allclose(rewards, [1, 0, 0, 0])
    assert info["events"]["covered_own"].tolist() == [1, 0, 0, 0]
    assert info["events"]["covered_other"].tolist() == [1, 0, 0, 0]
    targets, types = adapter.target_state()
    assert types.tolist() == [0, 1]
    assert np.linalg.norm(targets - np.array([[.15, 0.0], [-.15, 0.0]]), axis=1).min() > .1   # both respawned
    assert not terminated and not truncated


def test_coop_variant_needs_two_agents_and_pays_only_the_team():
    adapter = DiscoveryAdapter(DiscoveryConfig(team_sizes=(2, 2), team_assignment="fixed", targets_per_type=1,
                                               agents_per_target=2))
    adapter.reset(seed=0)
    far_layout(adapter)
    place(adapter, {0: (0.0, 0.0)}, {0: (.15, 0.0)})
    _, rewards, *_ = adapter.step([STAY] * 4)
    assert np.allclose(rewards, 0)                                         # one agent is not enough
    far_layout(adapter)
    place(adapter, {0: (0.0, 0.0), 2: (.3, 0.0)}, {0: (.15, 0.0)})         # helper from team 1
    _, rewards, *_, info = adapter.step([STAY] * 4)
    assert np.allclose(rewards, [1, 0, 0, 0])
    assert info["events"]["covered_other"].tolist() == [0, 0, 1, 0]


def test_seeded_resets_are_reproducible_and_isolated_from_other_vmas_environments():
    def rollout(adapter, interleave=False):
        out = [adapter.reset(seed=5)]
        rng = np.random.default_rng(0)
        for t in range(30):
            if interleave and t % 7 == 0:   # an evaluation / video environment built and used mid-episode
                other = DiscoveryAdapter(DiscoveryConfig(team_seed=9))
                other.reset(seed=123)
                other.step(rng.integers(0, 9, 4))
            out.append(adapter.step(np.random.default_rng(t).integers(0, 9, 4))[0])
            if t == 20:
                out.append(adapter.reset())  # unseeded reset continues the adapter's own stream
        return np.stack(out)
    plain = rollout(DiscoveryAdapter(DiscoveryConfig(team_seed=1)))
    mixed = rollout(DiscoveryAdapter(DiscoveryConfig(team_seed=1)), interleave=True)
    assert np.array_equal(plain, mixed)


def test_physics_and_observations_ignore_team_labels():
    a = DiscoveryAdapter(DiscoveryConfig(team_seed=1))
    b = DiscoveryAdapter(DiscoveryConfig(team_seed=2))
    assert not np.array_equal(a.oracle_team_ids(), b.oracle_team_ids())
    oa, ob = a.reset(seed=11), b.reset(seed=11)
    assert np.array_equal(oa, ob)
    rng = np.random.default_rng(1)
    for _ in range(40):
        actions = rng.integers(0, 9, 4)
        oa, ra, *_, ia = a.step(actions)
        ob, rb, *_, ib = b.step(actions)
        covered = ia["events"]["covered_own"] + ia["events"]["covered_other"]
        assert np.array_equal(covered, ib["events"]["covered_own"] + ib["events"]["covered_other"])
        if not np.any(covered):            # a covering respawns targets at random, so compare until then
            assert np.array_equal(oa, ob)
    assert set(ia) == {"events"}


def test_contact_matrix():
    adapter = DiscoveryAdapter(DiscoveryConfig(), AdapterConfig(comm_range=.5))
    adapter.reset(seed=0)
    place(adapter, {0: (0.0, 0.0), 1: (.4, 0.0), 2: (.0, .6), 3: (.9, .9)}, {})
    contact = adapter.contact_matrix()
    assert contact[0, 1] and not contact[0, 2] and not contact[0, 3] and contact.diagonal().all()
    assert np.array_equal(contact, contact.T)
