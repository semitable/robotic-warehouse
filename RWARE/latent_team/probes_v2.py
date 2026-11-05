"""Probe bank v2: realised from the training environment itself, de-duplicated and validated.

Differences from the canonical bank (rware/utils/probe_maps.py, left unchanged):
  * every probe is produced by placing objects in an environment built with the *training* env kwargs and
    calling the real semantic observation builder, so window clipping and team-relative channels are exact;
  * every scenario declares what must be visible (carrying / goal visible / requested visible / return phase)
    and the bank is rejected if a realised observation contradicts it or two probes are identical;
  * two roles are separated:
      gate      states where the optimal action depends on the agent's own team even under team-relative
                observations: carrying an own-team requested shelf while no goal is visible (meaningful only
                when team goals are spatially separated, e.g. env variant E1-F);
      transfer  states covering every task phase evenly, used as the critic-distillation set.
All agents can generate the same bank from the shared deterministic rule; no observations are exchanged.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch

try:
    from . import common as C  # noqa: F401  (repository root on sys.path)
except ImportError:
    import common as C  # noqa: F401

from examples import train_recurrent_ippo_consensus as T
from rware.utils.semantic_observation import (SEMANTIC_EGO_FEATURES, SEMANTIC_SPATIAL_CHANNELS,
                                              build_semantic_observation)
from rware.warehouse import Direction

CH = {n: k for k, n in enumerate(SEMANTIC_SPATIAL_CHANNELS)}
EGO = {n: k for k, n in enumerate(SEMANTIC_EGO_FEATURES)}
DIRS = [Direction.UP, Direction.DOWN, Direction.LEFT, Direction.RIGHT]

# E1-F: same E1 map, team goals moved to the two bottom corners (Red left, Blue right).
ENV_VARIANTS = {"e1": {}, "e1f": {"team_goal_positions": [[[0, 10]], [[9, 10]]]}}


@dataclass
class Scenario:
    role: str            # gate | transfer
    phase: str           # search | carry_goal_visible | carry_goal_hidden | at_goal | return
    team: int
    x: int
    y: int
    direction: int       # index into DIRS
    carrying: bool = False
    return_phase: bool = False
    expect_goal_visible: bool | None = None
    expect_requested_visible: bool | None = None


class ProbeRealiser:
    """Places one focal agent (agent 0) in a fresh copy of the training environment."""

    def __init__(self, env_id, env_kwargs, observation_format="semantic", seed=1):
        self.env_cm = T.make_env(env_id, env_kwargs, observation_format)
        self.env = self.env_cm.__enter__()
        self.seed = seed
        self.env.reset(seed=seed)
        self.u = self.env.unwrapped
        self.H, self.W = map(int, self.u.grid_size)
        self.r = int(self.u.sensor_range)
        self.homes = {int(s.id): (int(s.x), int(s.y)) for s in self.u.shelfs}

    def close(self):
        self.env_cm.__exit__(None, None, None)

    def highway_cells(self):
        return [(x, y) for y in range(self.H) for x in range(self.W) if self.u._is_highway(x, y)]

    def goal_visible(self, x, y, team):
        return any(abs(gx - x) <= self.r and abs(gy - y) <= self.r for gx, gy in self.u.goals_by_team[team])

    def realise(self, sc: Scenario):
        u = self.u
        self.env.reset(seed=self.seed)
        for s in u.shelfs:
            s.x, s.y = self.homes[int(s.id)]
        for a in u.agents:
            a.carrying_shelf = None
        u.team_request_queues = [[] for _ in range(u.n_teams)]
        u._shelves_awaiting_return.clear()
        # Two requested shelves per team, chosen deterministically from each team's own shelves.
        for team in range(u.n_teams):
            owned = sorted((s for s in u.shelfs if u.shelf_team_ids.get(int(s.id)) == team), key=lambda s: s.id)
            u.team_request_queues[team] = owned[:: max(1, len(owned) // 2)][:2]
        u._sync_global_request_queue()
        focal = u.agents[0]
        u.agent_team_ids = np.asarray(u.agent_team_ids, dtype=np.int32)
        u.agent_team_ids[0] = sc.team
        focal.x, focal.y, focal.dir = sc.x, sc.y, DIRS[sc.direction]
        if sc.carrying:
            shelf = u.team_request_queues[sc.team][0]
            focal.carrying_shelf = shelf
            shelf.x, shelf.y = sc.x, sc.y
            if sc.return_phase:
                u._shelves_awaiting_return.add(int(shelf.id))
        # Other agents parked on the highway cells farthest from the focal agent (normally out of view).
        far = sorted(self.highway_cells(), key=lambda c: -(abs(c[0] - sc.x) + abs(c[1] - sc.y)))
        for agent, (x, y) in zip(u.agents[1:], far):
            agent.x, agent.y, agent.dir, agent.carrying_shelf = x, y, Direction.UP, None
        u._recalc_grid()
        return build_semantic_observation(u, focal)


def scenarios(realiser: ProbeRealiser):
    """Deterministic scenario list for both teams; positions come from the actual highway layout."""
    out = []
    cells = realiser.highway_cells()
    H, W = realiser.H, realiser.W
    for team in range(len(realiser.u.goals_by_team)):
        # Gate: carrying an own-team requested shelf with no goal in view, every highway cell and heading.
        for (x, y) in cells:
            # No goal of ANY team in view: the same input is then a valid state for either team's actor.
            if any(realiser.goal_visible(x, y, t) for t in range(len(realiser.u.goals_by_team))):
                continue
            for d in range(4):
                out.append(Scenario("gate", "carry_goal_hidden", team, x, y, d, carrying=True, expect_goal_visible=False))
        # Transfer: every phase, a coarse grid of positions so that no phase dominates.
        for (x, y) in cells[::3]:
            d = (x + y) % 4
            gv = realiser.goal_visible(x, y, team)
            out.append(Scenario("transfer", "search", team, x, y, d))
            out.append(Scenario("transfer", "carry_goal_visible" if gv else "carry_goal_hidden", team, x, y, d,
                                carrying=True, expect_goal_visible=gv))
            out.append(Scenario("transfer", "return", team, x, y, d, carrying=True, return_phase=True))
        for gx, gy in realiser.u.goals_by_team[team]:
            out.append(Scenario("transfer", "at_goal", team, gx, gy, 0, carrying=True, expect_goal_visible=True))
    return out


def validate(obs, sc: Scenario, spec):
    ego, grid = obs[:spec.ego_dim], obs[spec.ego_dim:].reshape(spec.spatial_channels, spec.spatial_size, spec.spatial_size)
    problems = []
    if bool(ego[EGO["self_carrying_shelf"]]) != sc.carrying:
        problems.append("carrying flag")
    if sc.carrying and not sc.return_phase and not bool(ego[EGO["self_carrying_requested_shelf"]]):
        problems.append("carried shelf not requested")
    if bool(ego[EGO["self_return_phase"]]) != sc.return_phase:
        problems.append("return phase flag")
    if sc.expect_goal_visible is not None and bool(grid[CH["relevant_goal"]].any()) != sc.expect_goal_visible:
        problems.append("relevant goal visibility")
    if sc.role == "gate" and bool(grid[CH["goal"]].any()):
        problems.append("a goal is visible in a gate probe")
    return problems


def build_bank(env_id, env_kwargs, *, seed=1, max_gate=48, max_transfer=64):
    realiser = ProbeRealiser(env_id, env_kwargs, seed=seed)
    try:
        spec = T.get_semantic_observation_spec(realiser.env)
        realised, rejected, seen = {"gate": [], "transfer": []}, [], {}
        for sc in scenarios(realiser):
            obs = realiser.realise(sc).astype(np.float32)
            problems = validate(obs, sc, spec)
            key = (sc.role, obs.tobytes())  # unique within each role; roles may share a state
            if problems:
                rejected.append(dict(asdict(sc), reason=problems))
            elif key in seen:
                rejected.append(dict(asdict(sc), reason=[f"duplicate of {seen[key]}"]))
            else:
                seen[key] = f"{sc.role}:{sc.phase}:t{sc.team}:({sc.x},{sc.y}):d{sc.direction}"
                realised[sc.role].append((sc, obs))
    finally:
        realiser.close()
    rng = np.random.default_rng(seed + 9173)  # same deterministic rule for every agent
    bank = {}
    for role, cap in (("gate", max_gate), ("transfer", max_transfer)):
        items = realised[role]
        if len(items) > cap:
            # Keep phases and teams balanced: round-robin over (phase, team) groups.
            groups = {}
            for sc, obs in items:
                groups.setdefault((sc.phase, sc.team), []).append((sc, obs))
            for g in groups.values():
                rng.shuffle(g)
            picked, keys = [], sorted(groups)
            while len(picked) < cap:
                for k in keys:
                    if groups[k] and len(picked) < cap:
                        picked.append(groups[k].pop())
            items = picked
        bank[role] = dict(sequences=torch.as_tensor(np.stack([o for _, o in items]))[:, None, :],
                          scenarios=[asdict(sc) for sc, _ in items])
    summary = dict(env_id=env_id, goals_by_team=[list(map(list, g)) for g in json.loads(json.dumps(
        env_kwargs.get("team_goal_positions") or []))], seed=seed,
        counts={r: len(bank[r]["scenarios"]) for r in bank},
        phases={r: {p: sum(s["phase"] == p for s in bank[r]["scenarios"]) for p in
                    sorted({s["phase"] for s in bank[r]["scenarios"]})} for r in bank},
        rejected=len(rejected), rejected_examples=rejected[:10],
        all_unique=True, note="v2 bank: realised from the training env, unique and validated")
    return bank, summary


def save_bank(bank, summary, path):
    torch.save(dict(sequences=bank["gate"]["sequences"], metadata=dict(probe_sequence_count=float(len(bank["gate"]["scenarios"])),
                                                                         probe_sequence_length=1.0, probe_team_conditioned=0.0,
                                                                         objective_probe_count=float(len(bank["gate"]["scenarios"])),
                                                                         rollout_probe_count=0.0),
                    transfer_sequences=bank["transfer"]["sequences"], gate_scenarios=bank["gate"]["scenarios"],
                    transfer_scenarios=bank["transfer"]["scenarios"], summary=summary, version="v2"), path)


def render(bank, spec_size, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rows = []
    for role in ("gate", "transfer"):
        for sc, seq in zip(bank[role]["scenarios"], bank[role]["sequences"][:, 0].numpy()):
            rows.append((role, sc, seq))
    cols = 12
    fig, axes = plt.subplots(int(np.ceil(len(rows) / cols)), cols, figsize=(cols * 1.7, np.ceil(len(rows) / cols) * 1.9), dpi=100)
    ego_dim = len(SEMANTIC_EGO_FEATURES)
    for ax in axes.flat:
        ax.axis("off")
    for ax, (role, sc, obs) in zip(axes.flat, rows):
        g = obs[ego_dim:].reshape(len(SEMANTIC_SPATIAL_CHANNELS), spec_size, spec_size)
        img = np.ones((spec_size, spec_size, 3)) * .95
        img[g[CH["highway"]] > 0] = (.98, .98, .98)
        img[g[CH["storage_cell"]] > 0] = (.86, .86, .86)
        img[g[CH["shelf"]] > 0] = (.55, .55, .55)
        img[g[CH["requested_shelf"]] > 0] = (1.0, .65, 0.0)
        img[g[CH["goal"]] > 0] = (.75, .75, 1.0)
        img[g[CH["relevant_goal"]] > 0] = (.1, .7, .2)
        img[g[CH["out_of_bounds"]] > 0] = (.15, .15, .15)
        ax.imshow(img, interpolation="nearest")
        c = spec_size // 2
        ax.add_patch(plt.Rectangle((c - .5, c - .5), 1, 1, fill=False, ec="red", lw=1.5))
        ax.set_title(f"{role[:4]} {sc['phase'][:14]}\nT{sc['team']} ({sc['x']},{sc['y']}) d{sc['direction']}", fontsize=6)
    fig.suptitle("Probe bank v2 (gate = team-discriminative, transfer = critic distillation); green = MY goal, "
                 "orange = MY requested shelf, red box = self", fontsize=10)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--env-variant", choices=list(ENV_VARIANTS), default="e1f")
    parser.add_argument("--plan", type=Path, help="Use env kwargs of an existing from-scratch plan (pgct job).")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()
    plan = args.plan or C.REFERENCE_EXPERIMENT
    job = next(j for j in json.loads((plan / "experiment.json").read_text(encoding="utf-8"))["jobs"]
               if j["mode"] == "pgct" and j["map"] == "e1")
    env_kwargs = json.loads(job["config"]["env_kwargs_json"])
    env_kwargs.update(ENV_VARIANTS[args.env_variant])
    bank, summary = build_bank(job["config"]["env_id"], env_kwargs, seed=args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    save_bank(bank, summary, args.output_dir / f"probe_bank_v2_{args.env_variant}_s{args.seed}.pt")
    (args.output_dir / f"probe_bank_v2_{args.env_variant}_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    render(bank, 2 * int(env_kwargs.get("sensor_range", 5)) + 1, args.output_dir / f"probe_bank_v2_{args.env_variant}.png")
    print(json.dumps({k: v for k, v in summary.items() if k != "rejected_examples"}, indent=1))


if __name__ == "__main__":
    main()
