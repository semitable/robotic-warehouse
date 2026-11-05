"""v2 facing-frame probe projection (D134): geometry, relative actions, and the opposite-window case.

The decisive check mirrors what the 500k pilot showed (E89): teammates that serve only at opposite serving
windows. With the north-up v1 view the gate cannot pair them; with the facing frame it must pair exactly the
hidden teammates. Scripted experience, no policy learning; labels only in assertions.
"""
import numpy as np
import pytest

from pgct_core.metrics import same_team_matrix
from pgct_core.reward_gate import RewardGateConfig, RewardProbeGate
from pgct_overcooked.adapter import (CH, EGO_SIZE, N_CHANNELS, AdapterConfig, FacingProbeProjection, KitchenAdapter,
                                     KitchenProbeProjection, relative_action)
from pgct_overcooked.env.kitchen import EAST, INTERACT, NORTH, SOUTH, STAY, WEST, KitchenConfig

FIXED = KitchenConfig(team_assignment="fixed")  # agents 0,1 onion team; 2,3 tomato team


def grid(view):
    return np.asarray(view)[EGO_SIZE:].reshape(N_CHANNELS, 3, 3)


def place(adapter, i, pos, facing, held=None):
    a = adapter.env.agents[i]
    a.pos, a.facing, a.held = pos, facing, held


def test_default_stays_v1_and_option_selects_v2():
    assert isinstance(KitchenAdapter(FIXED).projection, KitchenProbeProjection)
    assert not hasattr(KitchenAdapter(FIXED).projection, "public_transition")
    adapter = KitchenAdapter(FIXED, AdapterConfig(probe_frame="facing"))
    assert isinstance(adapter.projection, FacingProbeProjection)
    assert adapter.projection.view_size == KitchenAdapter(FIXED).projection.view_size
    with pytest.raises(ValueError):
        AdapterConfig(probe_frame="mirror")


@pytest.mark.parametrize("heading", range(4))
def test_relative_actions(heading):
    absolute = {0: NORTH, 1: EAST, 2: SOUTH, 3: WEST}
    assert relative_action(absolute[heading], heading) == NORTH            # forward
    assert relative_action(absolute[(heading + 1) % 4], heading) == EAST   # right
    assert relative_action(absolute[(heading + 2) % 4], heading) == SOUTH  # back
    assert relative_action(absolute[(heading + 3) % 4], heading) == WEST   # left
    assert relative_action(STAY, heading) == STAY and relative_action(INTERACT, heading) == INTERACT


def test_west_and_east_window_deliveries_share_the_front_cell_in_the_facing_frame():
    v1 = KitchenAdapter(FIXED)
    v2 = KitchenAdapter(FIXED, AdapterConfig(probe_frame="facing"))
    views = {}
    for name, adapter in (("v1", v1), ("v2", v2)):
        adapter.reset()
        place(adapter, 0, (3, 1), 3, "soup_tomato")   # facing the west window
        place(adapter, 1, (3, 9), 1, "soup_tomato")   # facing the east window
        place(adapter, 2, (1, 5), 2)
        place(adapter, 3, (5, 5), 2)
        obs = adapter.observations()
        views[name] = [adapter.projection.public_view(obs[i]) for i in (0, 1)]
    west, east = views["v1"]
    assert grid(west)[CH["serve"], 1, 0] == 1 and grid(east)[CH["serve"], 1, 2] == 1  # v1: window left vs right
    west, east = views["v2"]
    for v in (west, east):
        assert grid(v)[CH["serve"], 0, 1] == 1                       # v2: window straight ahead
        assert v[:4].tolist() == [1, 0, 0, 0] and v[4 + 5] == 1     # heading 'up', holding tomato soup
    # Rotation keeps handedness: dish dispenser front-right at the west window, front-left at the east one.
    assert grid(west)[CH["dish_dispenser"], 0, 2] == 1 and grid(east)[CH["dish_dispenser"], 0, 0] == 1
    diff = grid(west) != grid(east)
    assert set(map(tuple, np.argwhere(diff.any(0)))) == {(0, 0), (0, 2)}  # only the two front corners differ


def test_transition_after_view_uses_the_pre_step_heading():
    adapter = KitchenAdapter(FIXED, AdapterConfig(probe_frame="facing"))
    adapter.reset()
    place(adapter, 0, (1, 5), 1)                                     # facing east, open floor
    before = adapter.observations()[0]
    after, *_ = adapter.step([SOUTH, STAY, STAY, STAY])                # turn right and move south
    vb, a, va = adapter.projection.public_transition(before, SOUTH, after[0])
    assert a == EAST                                                  # 'right' in the pre-step frame
    assert vb[:4].tolist() == [1, 0, 0, 0] and va[:4].tolist() == [0, 1, 0, 0]


def opposite_window_rollout(adapter, gate, rng, steps):
    """Agents 0 and 2 serve only at the west window, 1 and 3 only at the east window; soup types random."""
    west, east = [0, 2], [1, 3]
    for _ in range(steps):
        servers = {rng.choice(west): ((3, 1), 3), rng.choice(east): ((3, 9), 1)}
        idle = iter([(1, 5), (5, 5)])
        for i in range(4):
            if i in servers:
                pos, facing = servers[i]
                place(adapter, i, pos, facing, "soup_onion" if rng.random() < .5 else "soup_tomato")
            else:
                place(adapter, i, next(idle), int(rng.integers(0, 4)))
        before = adapter.observations()
        actions = np.array([INTERACT if i in servers else rng.integers(0, 6) for i in range(4)])
        after, rewards, *_ = adapter.step(actions)
        gate.observe(list(before), actions, list(after), rewards)


def run_gate(frame, updates=8):
    adapter = KitchenAdapter(FIXED, AdapterConfig(probe_frame=frame))
    adapter.reset(seed=0)
    cfg = RewardGateConfig(radius=1, interval=1, distance_mode="action_max", mirror=False, contact_snapshot=True)
    gate = RewardProbeGate(4, cfg, adapter.projection)
    rng = np.random.default_rng(0)
    for update in range(1, updates + 1):
        gate.snapshot()
        opposite_window_rollout(adapter, gate, rng, 150)
        gate(update, np.ones((4, 4), bool), active=True)
    return gate, adapter.oracle_team_ids()


def test_opposite_window_teammates_need_the_facing_frame():
    gate, teams = run_gate("absolute")
    same = same_team_matrix(teams)
    v1 = gate.ema + gate.ema.T
    assert gate.maturity().all() and not gate.approved()[same].any()  # v1: no teammate pair approved
    assert min(v1[0, 1], v1[2, 3]) > .5                                # (measured .53/.54)
    gate, teams = run_gate("facing")
    assert gate.maturity().all() and np.array_equal(gate.approved(), same_team_matrix(teams))
    v2 = gate.ema + gate.ema.T
    # Measured .32/.27 (same) vs >= .92 (cross): the two front corners (dish dispenser vs counter) still differ
    # at mirrored windows, so the same-team margin below the .5 threshold is real but not large.
    assert max(v2[0, 1], v2[2, 3]) < .4 and min(v2[0, 2], v2[0, 3], v2[1, 2], v2[1, 3]) > .8
    assert v2[0, 1] < v1[0, 1] and v2[2, 3] < v1[2, 3]


def test_v2_config_changes_only_the_gate(tmp_path):
    """ippo is bit-identical under v1 and v2 configs; pgct_d2 runs with the facing projection."""
    import torch
    from pgct_core.trainer import run_training
    from pgct_overcooked import train as TR
    from pgct_overcooked.experiment import apply_overrides, build, load_config
    tiny = TR.SMOKE + ["trainer.total_timesteps=128", "trainer.eval_interval=0", "trainer.log_interval=0"]
    final = {}
    for name in ("main_4p2t.json", "main_4p2t_probe_v2.json"):
        for method in ("ippo", "pgct_d2"):
            make_adapter, cfg, ppo, gate, mg, resolved = build(apply_overrides(load_config(name), tiny), method, 1)
            out = tmp_path / f"{name}_{method}"
            run_training(make_adapter, out, cfg, ppo, gate, mg)
            final[name, method] = torch.load(out / "final.pt", weights_only=False)
    a, b = final["main_4p2t.json", "ippo"], final["main_4p2t_probe_v2.json", "ippo"]
    assert all(torch.equal(x[k], y[k]) for x, y in zip(a["agents"], b["agents"]) for k in x)
    assert final["main_4p2t_probe_v2.json", "pgct_d2"]["reward_gate"]["predictors"][0]["version"] >= 1


def test_v3_cross_makes_west_and_east_window_deliveries_identical():
    adapter = KitchenAdapter(FIXED, AdapterConfig(probe_frame="facing_cross"))
    adapter.reset()
    place(adapter, 0, (3, 1), 3, "soup_tomato")
    place(adapter, 1, (3, 9), 1, "soup_tomato")
    place(adapter, 2, (1, 5), 2)
    place(adapter, 3, (5, 5), 2)
    obs = adapter.observations()
    west, east = (adapter.projection.public_view(obs[i]) for i in (0, 1))
    assert len(west) == adapter.projection.view_size == EGO_SIZE + N_CHANNELS * 5
    assert np.array_equal(west, east)
    cells = west[EGO_SIZE:].reshape(N_CHANNELS, 5)
    assert cells[CH["serve"], 0] == 1 and cells[CH["serve"], 1:].sum() == 0  # the window is the front cell
    with pytest.raises(ValueError):
        KitchenAdapter(FIXED, AdapterConfig(obs_radius=3, probe_radius=2, probe_frame="facing_cross"))


def test_v3_pairs_opposite_window_teammates_with_a_wide_margin():
    gate, teams = run_gate("facing_cross")
    assert gate.maturity().all() and np.array_equal(gate.approved(), same_team_matrix(teams))
    v3 = gate.ema + gate.ema.T
    assert max(v3[0, 1], v3[2, 3]) < .1 and min(v3[0, 2], v3[0, 3], v3[1, 2], v3[1, 3]) > .8
