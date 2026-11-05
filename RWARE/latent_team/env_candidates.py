"""Candidate environments to replace E1-F (D62).  Only existing environment arguments are used.

All candidates keep E1-F's separated team goals (one goal per team at the two bottom corners, so a
team-specific direction exists whenever no goal is in view) and switch to TEAM reward (delivery /
return rewards go to every member of the delivering team; shaping stays individual).  The goal side is
swapped on even seeds: a robustness check across seeds only -- within one seed direction and team still
coincide, so this does not separate team discovery from movement direction (D63).

Current candidates (D63, priority B -> A -> C):
  a_e1f4_team        E1 map (11 x 10), 4 agents (2+2), sensor 5, communication 5   minimal change from E1-F
  b_e1f6_comm3       E1 map (11 x 10), 6 agents (3+3), sensor 5, communication 3   main candidate
  c_e1f6_wide_comm3  wider map (11 x 16, shelf_columns 5), 6 agents (3+3),         spatial extension; 13 x 12 is
                     sensor 5, communication 3                                     not buildable (odd columns only)
  6-agent candidates use 3 requests per team.

Pickup-penalty ablation (D69 / E45): the same environments with the two shaping penalties that make
TOGGLE the worst action under early exploration set to 0 (unrequested_shelf_pickup_penalty,
premature_drop_penalty; everything else unchanged):
  a_e1f4_team_nopen, b_e1f6_comm3_nopen
Setting them to 0 made TOGGLE free and agents toggled 100-300 times per episode (E46).  The middle value
(D70 / E47): both penalties at .005, just below the point (~.0056-.0059) where TOGGLE stops being the worst
action under a uniform random policy:
  b_e1f6_comm3_pen005

Identifiability 2 x 2 (D74): one base (B: 6 agents 3+3, sensor 5, communication 3, pickup penalties .005),
only the goal layout and the reward mode change.  b_e1f6_comm3_pen005 is the corner / team cell.
  b6_corner_indiv_pen005    goals at the two bottom corners, individual reward   (E1-F type)
  b6_adjacent_team_pen005   goals side by side at the bottom centre, team reward  (E1 type + team reward)
  b6_adjacent_indiv_pen005  goals side by side at the bottom centre, individual   (E1 type)
Adjacent goals are (width/2 - 1, bottom) and (width/2, bottom), i.e. E1's (4, 10) / (5, 10); like the
corners they swap sides on even seeds.

Contact range 5 (D110 / T178): the paper's default r_c equals the sensor range (5); range 3 above was an
arbitrary choice.  Only communication_range changes (contact graph for the A+C gate and Max-GAE peers);
observations, rewards and dynamics are those of b6_corner_indiv_pen005.
  b6_corner_indiv_comm5_pen005   b6_corner_indiv_pen005 with communication range 5
Slightly harder variants of it (T179/T180, user: raise the difficulty a little; one change each):
  b6_wide_corner_indiv_comm5_pen005   shelf_columns 5 -> 11 x 16 map, goals at the two far bottom corners
  b6_corner_indiv_comm5_req2_pen005   2 requested shelves per team instead of 3 (request_queue_size 4)

Earlier candidates (E40, kept for the record; superseded by D63):
  c1_e1f_team      = a_e1f4_team
  c2_e1f8_team     wider map (11 x 16), 8 agents / 2 teams, 4 requests/team
  c3_wide_s3_team  larger map (20 x 16), 4 agents / 2 teams, sensor 3, communication range 3

Used through train_from_scratch.py `--maps e1 --env-variant <candidate>`; the candidate's reward mode
(LAYOUT, team unless stated) overrides --team-reward.
"""
from __future__ import annotations

from examples import train_recurrent_ippo_consensus as T

# The trainer overrides env_kwargs["sensor_range"] / n_agents with its own config fields
# (cfg.sensor_range, cfg.agent_count), so those are set on the training config as well.
CONFIG = {
    "a_e1f4_team": dict(),
    "b_e1f6_comm3": dict(agent_count=6),
    "c_e1f6_wide_comm3": dict(agent_count=6),
    "a_e1f4_team_nopen": dict(),
    "b_e1f6_comm3_nopen": dict(agent_count=6),
    "b_e1f6_comm3_pen005": dict(agent_count=6),
    "b6_corner_indiv_pen005": dict(agent_count=6),
    "b6_adjacent_team_pen005": dict(agent_count=6),
    "b6_adjacent_indiv_pen005": dict(agent_count=6),
    "b6_corner_indiv_comm5_pen005": dict(agent_count=6),
    "b6_wide_corner_indiv_comm5_pen005": dict(agent_count=6),
    # the trainer raises requests to >= agents per team unless auto-scaling is off (2 < 3 is intended here)
    "b6_corner_indiv_comm5_req2_pen005": dict(agent_count=6, auto_scale_request_queue=False),
    "c1_e1f_team": dict(),
    "c2_e1f8_team": dict(agent_count=8),
    "c3_wide_s3_team": dict(sensor_range=3),
}

NO_PICKUP_PENALTY = dict(unrequested_shelf_pickup_penalty=0.0, premature_drop_penalty=0.0)
SMALL_PICKUP_PENALTY = dict(unrequested_shelf_pickup_penalty=0.005, premature_drop_penalty=0.005)
SIX = dict(n_agents=6, request_queue_size=6, request_queue_size_per_team=[3, 3])
CANDIDATES = {
    "a_e1f4_team": dict(),
    "b_e1f6_comm3": dict(SIX, communication_range=3),
    "c_e1f6_wide_comm3": dict(SIX, shelf_columns=5, communication_range=3),
    "a_e1f4_team_nopen": dict(NO_PICKUP_PENALTY),
    "b_e1f6_comm3_nopen": dict(SIX, communication_range=3, **NO_PICKUP_PENALTY),
    "b_e1f6_comm3_pen005": dict(SIX, communication_range=3, **SMALL_PICKUP_PENALTY),
    "b6_corner_indiv_pen005": dict(SIX, communication_range=3, **SMALL_PICKUP_PENALTY),
    "b6_adjacent_team_pen005": dict(SIX, communication_range=3, **SMALL_PICKUP_PENALTY),
    "b6_adjacent_indiv_pen005": dict(SIX, communication_range=3, **SMALL_PICKUP_PENALTY),
    "b6_corner_indiv_comm5_pen005": dict(SIX, communication_range=5, **SMALL_PICKUP_PENALTY),
    "b6_wide_corner_indiv_comm5_pen005": dict(SIX, shelf_columns=5, communication_range=5, **SMALL_PICKUP_PENALTY),
    "b6_corner_indiv_comm5_req2_pen005": dict(SIX, communication_range=5, **SMALL_PICKUP_PENALTY,
                                              request_queue_size=4, request_queue_size_per_team=[2, 2]),
    "c1_e1f_team": dict(),
    "c2_e1f8_team": dict(n_agents=8, shelf_columns=5, request_queue_size=8, request_queue_size_per_team=[4, 4]),
    "c3_wide_s3_team": dict(shelf_columns=5, shelf_rows=2, sensor_range=3, communication_range=3),
}
# Goal layout and reward mode; candidates not listed use corner goals and team reward.
LAYOUT = {
    "b6_corner_indiv_pen005": dict(goals="corner", reward="individual"),
    "b6_adjacent_team_pen005": dict(goals="adjacent", reward="team"),
    "b6_adjacent_indiv_pen005": dict(goals="adjacent", reward="individual"),
    "b6_corner_indiv_comm5_pen005": dict(goals="corner", reward="individual"),
    "b6_wide_corner_indiv_comm5_pen005": dict(goals="corner", reward="individual"),
    "b6_corner_indiv_comm5_req2_pen005": dict(goals="corner", reward="individual"),
}


def grid_size(env_id, env_kwargs):
    kwargs = dict(env_kwargs)
    kwargs.pop("team_goal_positions", None)
    with T.make_env(env_id, kwargs, "flat") as env:
        env.reset(seed=0)
        return tuple(int(v) for v in env.unwrapped.grid_size)  # (height, width)


def env_overrides(variant, seed, env_id, base_env_kwargs):
    """Environment kwargs for a candidate: geometry, reward mode and seed-dependent goal sides."""
    layout = LAYOUT.get(variant, dict(goals="corner", reward="team"))
    over = dict(CANDIDATES[variant], team_reward_mode=layout["reward"])
    height, width = grid_size(env_id, {**base_env_kwargs, **over})
    if layout["goals"] == "corner":
        left, right = [[0, height - 1]], [[width - 1, height - 1]]
    else:
        left, right = [[width // 2 - 1, height - 1]], [[width // 2, height - 1]]
    over["team_goal_positions"] = [left, right] if int(seed) % 2 else [right, left]  # team 0 = red
    return over


# --- more than four agents -------------------------------------------------------------------------
# examples/train_pgct_spatial_experiments.SpatialMonitor (training-time spatial log) allocates 4 x 4
# arrays.  The file is left unchanged; for runs with another agent count the monitor class is swapped
# for one sized from the environment.  Its mean_physical_neighbors field still divides by 4 (computed
# inside experiment_logging), so for n agents it reads n/4 times the true mean; the raw
# geometric_candidate_step_counts in spatial.jsonl are exact.
import contextlib  # noqa: E402

import numpy as np  # noqa: E402

from examples import train_pgct_spatial_experiments as SPX  # noqa: E402


class NAgentSpatialMonitor(SPX.SpatialMonitor):
    def reset(self):
        self.steps = 0
        self.occupancy = None
        self.candidates = None
        self.phase_counts = None

    def record(self, env):
        base = env.unwrapped
        n = len(base.agents)
        if self.candidates is None:
            self.candidates = np.zeros((n, n), dtype=np.int64)
            self.phase_counts = np.zeros((n, 3), dtype=np.int64)
        if self.occupancy is None:
            self.occupancy = np.zeros((n, *base.grid_size), dtype=np.int64)
        self.candidates += SPX.geometric_adjacency(env, self.radius)
        for i, agent in enumerate(base.agents):
            self.occupancy[i, agent.y, agent.x] += 1
            carrying = agent.carrying_shelf
            phase = (2 if carrying.id in base._shelves_awaiting_return else 1) if carrying else 0
            self.phase_counts[i, phase] += 1
        self.steps += 1


@contextlib.contextmanager
def agent_count_support(agent_count):
    """Swap in the n-agent spatial monitor while a run with agent_count != 4 trains."""
    if not agent_count or int(agent_count) == 4:
        yield
        return
    original = SPX.SpatialMonitor
    SPX.SpatialMonitor = NAgentSpatialMonitor
    try:
        yield
    finally:
        SPX.SpatialMonitor = original
