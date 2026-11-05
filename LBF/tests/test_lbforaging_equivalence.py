"""With one team, the multi-team LBF reproduces the original lbforaging 2.0.0 step for step.

Skipped when lbforaging is not installed (pip install lbforaging==2.0.0 --no-deps; its pyglet dependency is only
needed for its own rendering). Both environments start from the state lbforaging spawned; then the same random
actions (LOAD over-represented) are applied and positions, food, rewards, the end of the episode and the grid
observations are compared after every step. The only intended difference: lbforaging reports the time limit as
``done``; here it is truncation.

lbforaging 2.0.0 cannot build its grid observation space under gymnasium 1.x (Box(low of shape (3, k, k),
shape=[3]) is refused), so the reference is created in vector mode and then switched to its own grid code path
with a correctly shaped space; the grid observations compared below are produced by lbforaging's code.
"""
import numpy as np
import pytest

lbf = pytest.importorskip("lbforaging.foraging.environment")
import gymnasium  # noqa: E402

from pgct_lbf.adapter import AdapterConfig, ForagingAdapter  # noqa: E402
from pgct_lbf.env.foraging import ForagingConfig  # noqa: E402

CASES = [(8, 2, 2, 2, False), (8, 2, 2, 8, True), (10, 3, 3, 2, False), (12, 4, 4, 3, False),
         (6, 3, 1, 1, True), (9, 4, 3, 2, True)]


@pytest.mark.parametrize("size,players,food,sight,coop", CASES)
def test_single_team_matches_lbforaging(size, players, food, sight, coop):
    ref = lbf.ForagingEnv(players=players, min_player_level=1, max_player_level=2, min_food_level=1,
                          max_food_level=None, field_size=(size, size), max_num_food=food, sight=sight,
                          max_episode_steps=50, force_coop=coop, grid_observation=False)
    k = 2 * sight + 1
    ref._grid_observation = True
    ref.observation_space = gymnasium.spaces.Tuple(
        [gymnasium.spaces.Box(0.0, np.inf, shape=(3, k, k), dtype=np.float32)] * players)
    ours = ForagingAdapter(ForagingConfig(field_size=size, team_sizes=(players,), food_per_team=food, sight=sight,
                                          horizon=50, force_coop=coop, team_assignment="fixed"),
                           AdapterConfig(probe_radius=min(2, sight)))
    rng = np.random.default_rng(size * 100 + players * 10 + food)
    collected = 0
    for episode in range(25):
        ref_obs, _ = ref.reset(seed=int(rng.integers(1 << 30)))
        env = ours.env
        env.positions[:] = [p.position for p in ref.players]
        env.levels[:] = [p.level for p in ref.players]
        env.field[:] = ref.field
        env.food_type[:] = np.where(ref.field > 0, 0, -1)
        env.food_spawned = np.array([float(ref._food_spawned)])
        env.t = 0
        assert np.array_equal(ours.observations().reshape(players, 3, 2 * sight + 1, 2 * sight + 1),
                              np.stack(ref_obs))
        done = False
        while not done:
            actions = np.where(rng.random(players) < .3, 5, rng.integers(0, 5, players))
            ref_obs, ref_rewards, ref_done, _, _ = ref.step(actions.tolist())
            obs, rewards, terminated, truncated, info = ours.step(actions)
            assert env.positions.tolist() == [list(p.position) for p in ref.players]
            assert np.array_equal(env.field, ref.field)
            assert np.allclose(rewards, ref_rewards)
            assert ref_done == (terminated or truncated)
            assert np.array_equal(obs.reshape(players, 3, 2 * sight + 1, 2 * sight + 1), np.stack(ref_obs))
            collected += int(info["events"]["collected_own"].sum() + info["events"]["load_failed"].sum())
            done = ref_done
    assert collected > 0  # the random actions really exercised loading (successful or failed)
