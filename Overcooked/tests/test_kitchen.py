"""Kitchen mechanics, individual team-specific rewards and information separation."""
import numpy as np
import pytest

from pgct_overcooked.adapter import CH, EGO_SIZE, AdapterConfig, KitchenAdapter, obs_size
from pgct_overcooked.env.kitchen import EAST, INTERACT, NORTH, STAY, WEST, Kitchen, KitchenConfig

FIXED = dict(team_assignment="fixed")  # agents 0,1 -> team 0 (onion); 2,3 -> team 1 (tomato)


def kitchen(**kw):
    return Kitchen(KitchenConfig(**{**FIXED, **kw}))


def act(env, i, action):
    actions = [STAY] * env.n_agents
    actions[i] = action
    return env.step(actions)


def place(env, i, pos, facing, held=None):
    env.agents[i].pos, env.agents[i].facing, env.agents[i].held = pos, facing, held


def test_layout_and_hidden_teams():
    env = kitchen()
    assert env.n_agents == 4 and env.oracle_team_ids().tolist() == [0, 0, 1, 1]
    shuffled = Kitchen(KitchenConfig(team_seed=3))
    assert sorted(shuffled.oracle_team_ids().tolist()) == [0, 0, 1, 1]
    with pytest.raises(ValueError):
        KitchenConfig(team_recipes=("onion", "beef"))


def test_full_recipe_cycle_pays_only_the_acting_agent_of_the_matching_team():
    env = kitchen(cook_time=2)
    # Agent 0 (onion team) at (1,1) faces the onion dispenser at (1,0).
    place(env, 0, (1, 1), 3)
    place(env, 1, (4, 5), 2)
    place(env, 2, (5, 5), 2)
    place(env, 3, (4, 8), 2)
    rewards = []
    for _ in range(3):
        act(env, 0, INTERACT)                       # take onion
        assert env.agents[0].held == "onion"
        place(env, 0, (1, 3), 0, "onion")           # under the pot at (0,3), facing north
        r, *_ = act(env, 0, INTERACT)
        rewards.append(r.copy())
        place(env, 0, (1, 1), 3)
    assert all(r[0] == pytest.approx(.1) and r[1:].sum() == 0 for r in rewards)
    pot = env.pots[(0, 3)]
    assert pot.ingredients == ["onion"] * 3 and pot.cooking()
    act(env, 0, STAY)
    act(env, 0, STAY)                               # cook_time 2 elapsed
    place(env, 0, (2, 1), 3, "dish")                # dish in hand, back to the pot
    place(env, 0, (1, 3), 0, "dish")
    r, *_ = act(env, 0, INTERACT)
    assert env.agents[0].held == "soup_onion" and r[0] == pytest.approx(.2)
    place(env, 0, (3, 1), 3, "soup_onion")          # facing the serving window at (3,0)
    r, _, _, info = act(env, 0, INTERACT)
    assert r.tolist() == [1, 0, 0, 0] and info["events"]["deliveries_own"][0] == 1
    assert "team" not in str(info.keys())


def test_other_team_soup_and_mixed_soup_pay_nothing():
    env = kitchen()
    place(env, 2, (3, 1), 3, "soup_onion")          # tomato-team agent serves an onion soup
    r, _, _, info = act(env, 2, INTERACT)
    assert r.sum() == 0 and info["events"]["deliveries_other"][2] == 1
    pot = env.pots[(0, 3)]
    pot.ingredients, pot.timer = ["onion", "tomato", "onion"], env.config.cook_time
    assert pot.soup() == "soup_mixed"
    place(env, 0, (1, 3), 0, "dish")
    r, *_ = act(env, 0, INTERACT)
    assert env.agents[0].held == "soup_mixed" and r.sum() == 0


def test_counters_and_dispensers():
    env = kitchen()
    place(env, 0, (2, 1), 3)                         # dish dispenser at (2,0)
    act(env, 0, INTERACT)
    assert env.agents[0].held == "dish"
    place(env, 0, (4, 1), 3, "dish")                 # counter at (4,0)
    act(env, 0, INTERACT)
    assert env.agents[0].held is None and env.counter_objects[(4, 0)] == "dish"
    act(env, 0, INTERACT)
    assert env.agents[0].held == "dish" and (4, 0) not in env.counter_objects


def test_moves_turn_and_collisions():
    env = kitchen()
    place(env, 0, (1, 4), 2)
    place(env, 1, (1, 6), 2)
    env.step([EAST, WEST, STAY, STAY])               # both target (1,5): nobody moves
    assert env.agents[0].pos == (1, 4) and env.agents[1].pos == (1, 6)
    assert env.agents[0].facing == 1 and env.agents[1].facing == 3
    env.step([EAST, STAY, STAY, STAY])
    assert env.agents[0].pos == (1, 5)
    env.step([EAST, STAY, STAY, STAY])               # (1,6) occupied
    assert env.agents[0].pos == (1, 5)
    env.step([NORTH, STAY, STAY, STAY])              # counter above: turn only
    assert env.agents[0].pos == (1, 5) and env.agents[0].facing == 0


def test_horizon_truncates():
    env = kitchen(horizon=3)
    out = [env.step([STAY] * 4)[2] for _ in range(3)]
    assert out == [False, False, True]


def test_observations_and_probe_views_are_independent_of_hidden_teams():
    a = KitchenAdapter(KitchenConfig(team_seed=1))
    b = KitchenAdapter(KitchenConfig(team_seed=2))
    assert a.oracle_team_ids().tolist() != b.oracle_team_ids().tolist()
    rng = np.random.default_rng(0)
    oa, ob = a.reset(seed=4), b.reset(seed=4)
    for _ in range(100):
        actions = rng.integers(0, 6, size=4)
        oa, ra, *_ = a.step(actions)
        ob, rb, *_ = b.step(actions)
        assert np.array_equal(oa, ob)                # physics and views never depend on the team labels
    assert np.array_equal(a.projection.public_view(oa), b.projection.public_view(ob))


def test_observation_layout_and_public_view_subset():
    adapter = KitchenAdapter(KitchenConfig(**FIXED), AdapterConfig(obs_radius=3, probe_radius=1))
    obs = adapter.reset()
    assert obs.shape == (4, obs_size(3)) and set(np.unique(obs)) <= {0.0, 1.0}
    place(adapter.env, 0, (3, 1), 3, "soup_tomato")
    obs = adapter.observations()
    view = adapter.projection.public_view(obs[0])
    assert view.shape == (obs_size(1),) and view.dtype == np.uint8
    assert view[3] == 1 and view[4 + 5] == 1        # facing west, holding a tomato soup
    side = 3
    grid = view[EGO_SIZE:].reshape(-1, side, side)
    assert grid[CH["serve"], 1, 0] == 1               # serving window to the west
    assert grid[CH["agent"], 1, 1] == 0               # self is ego, not "another agent"
    with pytest.raises(ValueError):
        adapter.projection.mirror(view[None], np.zeros(1), view[None])


def test_contact_matrix_is_symmetric_chebyshev():
    adapter = KitchenAdapter(KitchenConfig(**FIXED), AdapterConfig(comm_range=2))
    adapter.reset()
    c = adapter.contact_matrix()
    pos = np.array([ag.pos for ag in adapter.env.agents])
    assert np.array_equal(c, c.T) and c.diagonal().all()
    assert np.array_equal(c, np.abs(pos[:, None] - pos[None]).max(-1) <= 2)
