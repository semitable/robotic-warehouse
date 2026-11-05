"""Does the shared reward-probe gate separate the hidden kitchen teams from individual rewards?

Scripted experience, no policy learning: every agent sometimes serves an onion soup and sometimes a tomato
soup (as exploring agents do), and otherwise moves randomly. Each agent is paid only for its own team's soup.
The gate (pgct_core, RWARE PGCT-D2 values: action_max, no mirror, rollout-start snapshot) must approve
exactly the true teammate pairs. Labels are used only in the assertions.
"""
import numpy as np

from pgct_core.metrics import identification_summary, same_team_matrix
from pgct_core.reward_gate import RewardGateConfig, RewardProbeGate
from pgct_overcooked.adapter import KitchenAdapter
from pgct_overcooked.env.kitchen import INTERACT, STAY, KitchenConfig

SERVE_CELLS = [((3, 1), 3), ((3, 9), 1)]   # floor cell next to a serving window, facing it
IDLE_CELLS = [(1, 5), (5, 5)]


def scripted_rollout(adapter, gate, rng, steps):
    env = adapter.env
    for _ in range(steps):
        servers = rng.permutation(4)[:2]
        for slot, i in enumerate(servers):
            pos, facing = SERVE_CELLS[slot]
            env.agents[i].pos, env.agents[i].facing = pos, facing
            env.agents[i].held = "soup_onion" if rng.random() < .5 else "soup_tomato"
        for slot, i in enumerate(k for k in range(4) if k not in servers):
            env.agents[i].pos, env.agents[i].held = IDLE_CELLS[slot], None
        before = adapter.observations()
        actions = np.array([INTERACT if i in servers else rng.integers(0, 6) for i in range(4)])
        after, rewards, *_ = adapter.step(actions)
        gate.observe(list(before), actions, list(after), rewards)


def test_gate_approves_exactly_the_hidden_teammates():
    adapter = KitchenAdapter(KitchenConfig(team_seed=5))
    adapter.reset(seed=0)
    teams = adapter.oracle_team_ids()
    cfg = RewardGateConfig(radius=1, interval=1, distance_mode="action_max", mirror=False, contact_snapshot=True)
    gate = RewardProbeGate(4, cfg, adapter.projection)
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
