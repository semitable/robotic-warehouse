"""Does the shared reward-probe gate separate the hidden LBF teams from individual rewards?

Scripted experience, no policy learning: every step two random agents stand next to a food of a random type and
LOAD (as exploring agents do), the others move randomly. Each agent is paid only for its own team's food. The
gate (pgct_core with the LBF main-config values: action_max, no mirror, rollout-start snapshot, relative reward
events) must approve exactly the true teammate pairs. Labels are used only in the assertions.
"""
import numpy as np

from pgct_core.metrics import identification_summary, same_team_matrix
from pgct_core.reward_gate import RewardGateConfig, RewardProbeGate
from pgct_lbf.adapter import ForagingAdapter
from pgct_lbf.env.foraging import LOAD, ForagingConfig
from pgct_lbf.experiment import load_config

FOOD_CELLS = [(3, 3), (8, 8)]
SIDES = [(-1, 0), (1, 0), (0, -1), (0, 1)]
IDLE_CELLS = [(0, 11), (11, 0)]


def scripted_rollout(adapter, gate, rng, steps):
    env = adapter.env
    for _ in range(steps):
        env.field[:] = 0
        env.food_type[:] = -1
        for cell in FOOD_CELLS:
            env.field[cell] = 1
            env.food_type[cell] = rng.integers(0, 2)
        env.food_spawned = np.ones(2)
        env.levels[:] = 1
        env.t = 0
        loaders = rng.permutation(4)[:2]
        for slot, i in enumerate(loaders):
            dr, dc = SIDES[rng.integers(0, 4)]
            env.positions[i] = (FOOD_CELLS[slot][0] + dr, FOOD_CELLS[slot][1] + dc)
        for slot, i in enumerate(k for k in range(4) if k not in loaders):
            env.positions[i] = IDLE_CELLS[slot]
        before = adapter.observations()
        actions = np.array([LOAD if i in loaders else rng.integers(0, 5) for i in range(4)])
        after, rewards, *_ = adapter.step(actions)
        gate.observe(list(before), actions, list(after), rewards)


def test_gate_approves_exactly_the_hidden_teammates():
    adapter = ForagingAdapter(ForagingConfig(team_seed=5))
    adapter.reset(seed=0)
    teams = adapter.oracle_team_ids()
    gate_values = dict(load_config("main_4p2t.json")["gate"], interval=1, contact_snapshot=True)
    gate = RewardProbeGate(4, RewardGateConfig(radius=2, **gate_values), adapter.projection)
    rng = np.random.default_rng(0)
    contact = np.ones((4, 4), bool)
    records = []
    for update in range(1, 9):
        gate.snapshot()
        scripted_rollout(adapter, gate, rng, 150)
        gate(update, contact, active=True)
        records.append(dict(env_steps=update * 150, approved=gate.approved(), pairs=gate.last_log["ac_pairs"]))
    assert gate.maturity().all()
    assert np.array_equal(gate.approved(), same_team_matrix(teams))
    summary = identification_summary(records, teams)
    assert summary["auroc"] == 1.0 and summary["final"]["f1"] == 1.0
    assert summary["distance_same_mean"] < .2 and summary["distance_cross_mean"] > .8
    assert summary["first_wrong_approval_step"] is None
