"""Multi-team LBF rules, observation/probe separation and the label boundary."""
import numpy as np
import pytest

from pgct_lbf.adapter import AdapterConfig, ForagingAdapter
from pgct_lbf.env.foraging import EAST, LOAD, NONE, NORTH, SOUTH, WEST, ForagingConfig, MultiTeamForaging


def blank(team_sizes=(2, 2), **kw):
    env = MultiTeamForaging(ForagingConfig(team_sizes=team_sizes, team_assignment="fixed", **kw))
    env.field[:] = 0
    env.food_type[:] = -1
    env.positions[:] = -1
    env.t = 0
    return env


def put_food(env, r, c, level, team):
    env.field[r, c], env.food_type[r, c] = level, team
    env.food_spawned = np.array([env.field[env.food_type == t].sum() for t in range(env.n_teams)], np.float64)


def test_spawn_follows_lbf_spacing_and_team_level_rule():
    env = MultiTeamForaging(ForagingConfig(team_seed=3))
    for seed in range(200):
        env.reset(seed=seed)
        foods = np.argwhere(env.field > 0)
        assert len(foods) == 6 and all(1 <= r <= 10 and 1 <= c <= 10 for r, c in foods)
        for a in range(len(foods)):
            for b in range(a + 1, len(foods)):
                d = np.abs(foods[a] - foods[b])
                assert d.max() > 1 and not (min(d) == 0 and max(d) <= 2)   # 3x3 and row/column distance-2 rules
        for t in range(2):
            levels = env.field[env.food_type == t]
            assert len(levels) == 3 and levels.min() >= 1
            assert levels.max() <= env.levels[env.oracle_team_ids() == t].sum()  # team LBF rule (2 players)
        assert len({tuple(p) for p in env.positions}) == 4
        assert all(env.field[r, c] == 0 for r, c in env.positions)


def test_coop_and_ind_levels():
    coop = MultiTeamForaging(ForagingConfig(force_coop=True, team_seed=1))
    ind = MultiTeamForaging(ForagingConfig(max_food_level=1, team_seed=1))
    for seed in range(50):
        coop.reset(seed=seed)
        ind.reset(seed=seed)
        for t in range(2):
            team_sum = coop.levels[coop.oracle_team_ids() == t].sum()
            assert np.all(coop.field[coop.food_type == t] == team_sum)
            assert team_sum > coop.levels[coop.oracle_team_ids() == t].max()     # needs both teammates
        assert set(np.unique(ind.field[ind.field > 0])) == {1}


def test_loading_pays_only_the_foods_team_with_lbf_shares():
    env = blank()
    put_food(env, 5, 5, 3, team=0)
    put_food(env, 2, 2, 1, team=1)
    env.positions[:] = [(4, 5), (5, 4), (6, 5), (9, 9)]   # agents 0, 1 (team 0) and 2 (team 1) touch the food
    env.levels[:] = [1, 1, 2, 1]
    rewards, term, trunc, info = env.step([LOAD, LOAD, LOAD, NONE])
    total = 1 + 1 + 2
    assert env.field[5, 5] == 0 and env.food_type[5, 5] == -1
    assert np.allclose(rewards, [1 * 3 / (total * 3), 1 * 3 / (total * 3), 0, 0])  # helper from team 1: no pay
    assert info["events"]["collected_own"].tolist() == [1, 1, 0, 0]
    assert info["events"]["collected_other"].tolist() == [0, 0, 1, 0]
    assert not term and not trunc


def test_failed_load_and_penalty():
    env = blank(penalty=.1)
    put_food(env, 5, 5, 3, team=0)
    env.positions[:] = [(4, 5), (5, 4), (0, 0), (9, 9)]
    env.levels[:] = [1, 1, 2, 2]
    rewards, *_ , info = env.step([LOAD, LOAD, NONE, NONE])
    assert env.field[5, 5] == 3 and np.allclose(rewards, [-.1, -.1, 0, 0])
    assert info["events"]["load_failed"].tolist() == [1, 1, 0, 0]


def test_movement_collisions_invalid_moves_and_lbf_swap():
    env = blank()
    put_food(env, 5, 5, 1, team=0)
    env.levels[:] = 1
    env.positions[:] = [(5, 4), (0, 0), (3, 3), (3, 5)]
    env.step([EAST, NORTH, EAST, WEST])   # into food -> NONE; off the field -> NONE; 2 and 3 aim at (3, 4)
    assert env.positions.tolist() == [[5, 4], [0, 0], [3, 3], [3, 5]]
    env.positions[:] = [(7, 7), (7, 8), (0, 0), (9, 9)]
    env.step([EAST, WEST, NONE, NONE])     # LBF lets two players swap cells
    assert env.positions[:2].tolist() == [[7, 8], [7, 7]]
    env.positions[:] = [(7, 7), (7, 8), (0, 0), (9, 9)]
    env.step([EAST, NONE, NONE, NONE])     # into a player who stays -> neither moves
    assert env.positions[:2].tolist() == [[7, 7], [7, 8]]
    env.step([LOAD, NONE, NONE, NONE])     # LOAD without adjacent food is NONE
    assert env.positions[0].tolist() == [7, 7]


def test_termination_when_all_food_is_gone_and_truncation_at_horizon():
    env = blank(horizon=3)
    put_food(env, 5, 5, 1, team=1)
    env.levels[:] = 1
    env.positions[:] = [(0, 0), (1, 1), (5, 4), (9, 9)]
    _, term, trunc, _ = env.step([NONE, NONE, LOAD, NONE])
    assert term and not trunc
    env = blank(horizon=2)
    put_food(env, 5, 5, 1, team=1)
    env.positions[:] = [(0, 0), (1, 1), (2, 2), (9, 9)]
    assert env.step([NONE] * 4)[1:3] == (False, False)
    assert env.step([NONE] * 4)[1:3] == (False, True)


def test_observation_layers_probe_view_and_no_labels():
    adapter = ForagingAdapter(ForagingConfig(team_seed=4), AdapterConfig(probe_radius=1))
    obs = adapter.reset(seed=11)
    env = adapter.env
    assert obs.shape == (4, 4 * 25) and obs.dtype == np.float32
    for i, (r, c) in enumerate(env.positions):
        grid = obs[i].reshape(4, 5, 5)
        assert grid[0, 2, 2] == env.levels[i]                 # self at the centre with its level
        for dr in range(-2, 3):
            for dc in range(-2, 3):
                rr, cc = r + dr, c + dc
                inside = 0 <= rr < 12 and 0 <= cc < 12
                food = env.field[rr, cc] if inside else 0
                kind = env.food_type[rr, cc] if inside else -1
                assert grid[1, dr + 2, dc + 2] == (food if kind == 0 else 0)
                assert grid[2, dr + 2, dc + 2] == (food if kind == 1 else 0)
                free = inside and food == 0 and not any((env.positions == (rr, cc)).all(1))
                assert grid[3, dr + 2, dc + 2] == float(free)
        view = adapter.projection.public_view(obs[i])
        assert view.dtype == np.uint8 and view.size == 4 * 9
        assert np.array_equal(view.reshape(4, 3, 3), grid[:, 1:4, 1:4].astype(np.uint8))
    _, _, _, _, info = adapter.step([NONE] * 4)
    assert set(info) == {"events"}                              # no labels in step info
    # Observations are a function of the physical state only: relabelling the hidden teams changes nothing.
    # (Teams do shape the episode itself: LBF's food-level rule is applied per team, so food levels depend on
    # the players' levels in each team; and only own-team food pays.)
    before = adapter.observations()
    env._team_ids = env._team_ids[::-1].copy()
    assert np.array_equal(adapter.observations(), before)


def test_contact_matrix_and_validation():
    adapter = ForagingAdapter(ForagingConfig(), AdapterConfig(comm_range=2))
    adapter.env.positions[:] = [(0, 0), (2, 2), (5, 5), (0, 3)]
    contact = adapter.contact_matrix()
    assert contact[0, 1] and contact[1, 3] and not contact[0, 2] and not contact[0, 3]
    assert np.array_equal(contact, contact.T) and contact.diagonal().all()
    with pytest.raises(ValueError):
        ForagingAdapter(ForagingConfig(sight=2), AdapterConfig(probe_radius=3))
    with pytest.raises(ValueError):
        ForagingConfig(min_food_level=3, max_food_level=2)
