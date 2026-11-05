"""N-agent, multi-team kitchen with Overcooked-style mechanics (original implementation for PGCT).

Mechanics follow the common Overcooked research benchmark rules: ingredient and dish dispensers, counters,
pots that cook three ingredients, serving windows; six actions (stay, four moves that also turn, interact
with the faced cell). Extensions for latent multi-team research:

- any number of agents; every agent belongs to a hidden team; each team has its own recipe (soup type);
- individual rewards: an agent is paid only for its OWN actions that advance its OWN team's recipe
  (serving that soup, optional shaping for potting its ingredient / picking up its soup). Nothing is
  summed or shared between agents, and serving another team's soup pays nothing;
- observations never contain team identity or recipes; the step info has no labels.
  ``oracle_team_ids()`` exists for the Oracle baseline and evaluation only.

Grid characters: 'X' counter, 'O' onion dispenser, 'T' tomato dispenser, 'D' dish dispenser, 'P' pot,
'S' serving window, ' ' floor, '1'-'9' agent start cells (floor).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

STAY, NORTH, SOUTH, EAST, WEST, INTERACT = range(6)
N_ACTIONS = 6
ACTION_NAMES = ("stay", "north", "south", "east", "west", "interact")
# Facing index: 0 north, 1 east, 2 south, 3 west (row, col deltas)
DIRS = ((-1, 0), (0, 1), (1, 0), (0, -1))
MOVE_TO_DIR = {NORTH: 0, EAST: 1, SOUTH: 2, WEST: 3}

INGREDIENTS = ("onion", "tomato")
HELD = (None, "onion", "tomato", "dish", "soup_onion", "soup_tomato", "soup_mixed")
OBJECTS = HELD[1:]
STATIC = {"X": "counter", "O": "onion_dispenser", "T": "tomato_dispenser", "D": "dish_dispenser",
          "P": "pot", "S": "serve"}
WALKABLE = set(" 123456789")


@dataclass
class KitchenConfig:
    layout: str = "two_team_kitchen"
    horizon: int = 400
    cook_time: int = 10
    team_recipes: Tuple[str, ...] = ("onion", "tomato")   # recipe (ingredient) of team k
    team_sizes: Tuple[int, ...] = (2, 2)
    team_assignment: str = "shuffle"   # "shuffle" (permutation drawn from team_seed) or "fixed" (in order)
    team_seed: int = 0
    delivery_reward: float = 1.0       # serving a soup of one's own recipe
    wrong_delivery_reward: float = 0.0  # serving any other soup
    pot_reward: float = .1             # shaping: putting one's own ingredient into a pot
    soup_pickup_reward: float = .2     # shaping: taking one's own soup out of a pot

    def __post_init__(self):
        if len(self.team_recipes) != len(self.team_sizes):
            raise ValueError("one recipe per team")
        if any(r not in INGREDIENTS for r in self.team_recipes):
            raise ValueError(f"recipes must be in {INGREDIENTS}")
        if self.team_assignment not in ("shuffle", "fixed"):
            raise ValueError("team_assignment must be 'shuffle' or 'fixed'")
        if self.horizon < 1 or self.cook_time < 0:
            raise ValueError("horizon must be positive and cook_time nonnegative")


@dataclass
class Agent:
    pos: Tuple[int, int]
    facing: int = 2
    held: Optional[str] = None


@dataclass
class Pot:
    ingredients: List[str] = field(default_factory=list)
    timer: int = -1  # -1: not cooking; counts up once three ingredients are in

    def cooking(self):
        return 0 <= self.timer

    def soup(self):
        if all(x == "onion" for x in self.ingredients):
            return "soup_onion"
        if all(x == "tomato" for x in self.ingredients):
            return "soup_tomato"
        return "soup_mixed"


class Kitchen:
    def __init__(self, config: Optional[KitchenConfig] = None, layout_rows: Optional[Sequence[str]] = None):
        from .layouts import LAYOUTS
        self.config = config or KitchenConfig()
        rows = list(layout_rows if layout_rows is not None else LAYOUTS[self.config.layout])
        if len({len(r) for r in rows}) != 1:
            raise ValueError("layout rows must have equal length")
        self.height, self.width = len(rows), len(rows[0])
        self.static = np.full((self.height, self.width), "", dtype=object)
        starts = {}
        for r, line in enumerate(rows):
            for c, ch in enumerate(line):
                if ch in STATIC:
                    self.static[r, c] = STATIC[ch]
                elif ch in WALKABLE:
                    if ch != " ":
                        starts[int(ch)] = (r, c)
                else:
                    raise ValueError(f"unknown layout character {ch!r}")
        self.starts = [starts[k] for k in sorted(starts)]
        self.n_agents = len(self.starts)
        if self.n_agents != sum(self.config.team_sizes):
            raise ValueError(f"layout has {self.n_agents} start cells but teams hold {sum(self.config.team_sizes)}")
        teams = np.repeat(np.arange(len(self.config.team_sizes)), self.config.team_sizes)
        if self.config.team_assignment == "shuffle":
            teams = np.random.default_rng(self.config.team_seed).permutation(teams)
        self._team_ids = teams.astype(np.int64)
        self._recipes = [self.config.team_recipes[t] for t in self._team_ids]
        self.pot_cells = [tuple(x) for x in np.argwhere(self.static == "pot")]
        self.rng = np.random.default_rng()
        self.reset()

    # ----------------------------------------------------------------- evaluation-only information
    def oracle_team_ids(self) -> np.ndarray:
        """Ground-truth teams (Oracle baseline / evaluation only)."""
        return self._team_ids.copy()

    def oracle_recipes(self) -> List[str]:
        return list(self._recipes)

    # ----------------------------------------------------------------- dynamics
    def reset(self, seed: Optional[int] = None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.agents = [Agent(pos=p, facing=2) for p in self.starts]
        self.counter_objects: Dict[Tuple[int, int], str] = {}
        self.pots = {cell: Pot() for cell in self.pot_cells}
        self.t = 0
        return self

    def walkable(self, cell):
        r, c = cell
        return 0 <= r < self.height and 0 <= c < self.width and self.static[r, c] == ""

    def facing_cell(self, agent: Agent):
        dr, dc = DIRS[agent.facing]
        return agent.pos[0] + dr, agent.pos[1] + dc

    def step(self, actions):
        actions = [int(a) for a in actions]
        if len(actions) != self.n_agents or any(not 0 <= a < N_ACTIONS for a in actions):
            raise ValueError(f"need {self.n_agents} actions in [0, {N_ACTIONS})")
        rewards = np.zeros(self.n_agents, dtype=np.float32)
        events = {k: np.zeros(self.n_agents) for k in
                  ("deliveries_own", "deliveries_other", "pots_own", "pots_other", "soup_pickups_own",
                   "soup_pickups_other", "dish_pickups", "moves_blocked")}
        # Movement: turn towards the move direction; move if the target floor cell is free now and no other
        # agent targets it in this step (no swaps, no chains; deterministic).
        targets = {}
        for i, a in enumerate(actions):
            if a in MOVE_TO_DIR:
                agent = self.agents[i]
                agent.facing = MOVE_TO_DIR[a]
                dr, dc = DIRS[agent.facing]
                target = (agent.pos[0] + dr, agent.pos[1] + dc)
                if self.walkable(target):
                    targets[i] = target
                else:
                    events["moves_blocked"][i] += 1
        occupied = {a.pos for a in self.agents}
        claims: Dict[Tuple[int, int], int] = {}
        for target in targets.values():
            claims[target] = claims.get(target, 0) + 1
        for i, target in targets.items():
            if target in occupied or claims[target] > 1:
                events["moves_blocked"][i] += 1
                continue
            self.agents[i].pos = target
        # Interactions in agent order (sequential; an agent sees changes made earlier in the same step).
        for i, a in enumerate(actions):
            if a == INTERACT:
                self._interact(i, rewards, events)
        for pot in self.pots.values():
            if pot.cooking() and pot.timer < self.config.cook_time:
                pot.timer += 1
        self.t += 1
        truncated = self.t >= self.config.horizon
        return rewards, False, truncated, dict(events=events)

    def _interact(self, i, rewards, events):
        agent, cfg, recipe = self.agents[i], self.config, self._recipes[i]
        cell = self.facing_cell(agent)
        r, c = cell
        if not (0 <= r < self.height and 0 <= c < self.width):
            return
        kind = self.static[r, c]
        held = agent.held
        if kind in ("onion_dispenser", "tomato_dispenser") and held is None:
            agent.held = kind.split("_")[0]
        elif kind == "dish_dispenser" and held is None:
            agent.held = "dish"
            events["dish_pickups"][i] += 1
        elif kind == "counter":
            if held is not None and cell not in self.counter_objects:
                self.counter_objects[cell] = held
                agent.held = None
            elif held is None and cell in self.counter_objects:
                agent.held = self.counter_objects.pop(cell)
        elif kind == "pot":
            pot = self.pots[cell]
            if held in INGREDIENTS and len(pot.ingredients) < 3 and not pot.cooking():
                pot.ingredients.append(held)
                agent.held = None
                own = held == recipe
                events["pots_own" if own else "pots_other"][i] += 1
                if own:
                    rewards[i] += cfg.pot_reward
                if len(pot.ingredients) == 3:
                    pot.timer = 0
            elif held == "dish" and pot.cooking() and pot.timer >= cfg.cook_time:
                soup = pot.soup()
                agent.held = soup
                self.pots[cell] = Pot()
                own = soup == f"soup_{recipe}"
                events["soup_pickups_own" if own else "soup_pickups_other"][i] += 1
                if own:
                    rewards[i] += cfg.soup_pickup_reward
        elif kind == "serve" and held is not None and held.startswith("soup"):
            agent.held = None
            own = held == f"soup_{recipe}"
            events["deliveries_own" if own else "deliveries_other"][i] += 1
            rewards[i] += cfg.delivery_reward if own else cfg.wrong_delivery_reward

    # ----------------------------------------------------------------- encoding
    def grid_channels(self) -> np.ndarray:
        """Full-map channels [C, H, W] (no self/other distinction; see adapter.CHANNELS)."""
        from .. import adapter as A
        g = np.zeros((A.N_CHANNELS, self.height, self.width), dtype=np.float32)
        for name, ch in (("counter", A.CH["counter"]), ("onion_dispenser", A.CH["onion_dispenser"]),
                         ("tomato_dispenser", A.CH["tomato_dispenser"]),
                         ("dish_dispenser", A.CH["dish_dispenser"]), ("pot", A.CH["pot"]), ("serve", A.CH["serve"])):
            g[ch] = self.static == name
        for (r, c), obj in self.counter_objects.items():
            g[A.CH["obj_" + obj], r, c] = 1
        for (r, c), pot in self.pots.items():
            n_on = sum(x == "onion" for x in pot.ingredients)
            n_to = sum(x == "tomato" for x in pot.ingredients)
            if n_on:
                g[A.CH[f"pot_onion_{n_on}"], r, c] = 1
            if n_to:
                g[A.CH[f"pot_tomato_{n_to}"], r, c] = 1
            if pot.cooking():
                g[A.CH["pot_ready" if pot.timer >= self.config.cook_time else "pot_cooking"], r, c] = 1
        for agent in self.agents:
            r, c = agent.pos
            g[A.CH["agent"], r, c] = 1
            g[A.CH["agent_facing"] + agent.facing, r, c] = 1
            if agent.held is not None:
                g[A.CH["agent_held_" + agent.held], r, c] = 1
        return g

    def render_text(self) -> str:
        chars = {v: k for k, v in STATIC.items()}
        out = []
        for r in range(self.height):
            line = []
            for c in range(self.width):
                ch = chars.get(self.static[r, c], " ")
                if (r, c) in self.counter_objects:
                    ch = self.counter_objects[(r, c)][0].lower()
                for i, a in enumerate(self.agents):
                    if a.pos == (r, c):
                        ch = str(i)
                line.append(ch)
            out.append("".join(line))
        return "\n".join(out)
