"""PGCT reward-response gate (A+C gate v2), environment independent.

Extracted without algorithmic change from RWARE ``latent_team/reward_probe_ac_v2.py`` (commit 122615f).
Only the RWARE-specific parts moved out: the public observation projection, the action count and the
left-right mirror now come from a :class:`pgct_core.interfaces.ProbeProjection`.

- Predictor: each agent fits a small MLP r_hat_i(public before, action, public after) to its OWN scalar
  reward, from a bounded buffer of its own transitions. Its reward events (r >= event_reward) are
  oversampled.
- Probes: Q_ij = E_i u E_j, the public views of both agents' own recent reward events. Only views are
  exchanged, never a reward. An optional symmetry augmentation (``mirror``) is off by default here.
- Abstention: an agent votes only when its predictor is mature (enough own events, and it predicts them).
- Distance: sum|a-b| / sum(|a|+|b|) over the probes (0 = same responses, 1 = disjoint); EMA over rounds;
  same team if below the threshold after ``min_rounds`` rounds. ``distance_mode="action_max"`` (D108)
  takes the maximum over public-action strata in which both agents have >= min_probes own events.
- ``contact_snapshot`` (D103): compare what each agent held at the rollout start, then refit.
Team labels never enter a decision; :func:`audit` is for logging only. Private RNGs leave the caller's
random streams untouched.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import torch
import torch.nn as nn

DISTANCE_MODES = ("pooled", "action_max")


@dataclass(frozen=True)
class RewardGateConfig:
    radius: int = 2                 # radius of the public view (interpreted by the projection)
    buffer_size: int = 20000
    event_buffer_size: int = 512
    event_reward: float = .5
    event_fraction: float = .25
    hidden: int = 128
    learning_rate: float = 1e-3
    train_steps: int = 50
    batch_size: int = 256
    min_events: int = 20
    min_event_response: float = .3
    probes_per_agent: int = 32
    mirror: bool = False            # Q_ij = E_i u E_j; no augmentation unless explicitly requested
    min_probes: int = 8
    distance_threshold: float = .5
    ema_beta: float = .35
    min_rounds: int = 3
    interval: int = 5
    seed: int = 0
    contact_snapshot: bool = False
    distance_mode: str = "pooled"

    def __post_init__(self):
        if self.distance_mode not in DISTANCE_MODES:
            raise ValueError(f"distance_mode must be one of {DISTANCE_MODES}")
        counts = (self.radius, self.buffer_size, self.event_buffer_size, self.hidden, self.train_steps,
                  self.batch_size, self.min_events, self.probes_per_agent, self.min_probes, self.min_rounds,
                  self.interval)
        if min(counts) < 1:
            raise ValueError("A+C v2 counts and radius must be positive")
        if not 0 < self.ema_beta <= 1 or not 0 <= self.event_fraction <= 1:
            raise ValueError("Invalid EMA beta or event fraction")
        if min(self.event_reward, self.learning_rate, self.distance_threshold) <= 0:
            raise ValueError("Event reward, learning rate and distance threshold must be positive")


def features(before, actions, after, n_actions):
    before = np.asarray(before, dtype=np.float32)
    onehot = np.eye(n_actions, dtype=np.float32)[np.asarray(actions, dtype=np.int64)]
    return np.concatenate([before, onehot, np.asarray(after, dtype=np.float32)], axis=-1)


class Ring:
    """Fixed-size FIFO of (before, action, after, reward) public transitions."""
    def __init__(self, capacity, size, dtype=np.uint8):
        self.before = np.zeros((capacity, size), dtype)
        self.after = np.zeros((capacity, size), dtype)
        self.action = np.zeros(capacity, np.int8)
        self.reward = np.zeros(capacity, np.float32)
        self.capacity, self.count, self.head = capacity, 0, 0

    def add(self, before, action, after, reward):
        self.before[self.head], self.after[self.head] = before, after
        self.action[self.head], self.reward[self.head] = action, reward
        self.head = (self.head + 1) % self.capacity
        self.count = min(self.count + 1, self.capacity)

    def recent(self, n):
        n = min(n, self.count)
        idx = (self.head - 1 - np.arange(n)) % self.capacity
        return self.before[idx], self.action[idx], self.after[idx], self.reward[idx]

    def sample(self, n, rng):
        idx = rng.integers(0, self.count, size=n)
        return self.before[idx], self.action[idx], self.after[idx], self.reward[idx]


class RewardPredictor:
    """r_hat_i fitted only on agent i's own (public transition, own scalar reward) pairs."""
    def __init__(self, config, seed, view_size, n_actions, view_dtype=np.uint8):
        self.config = config
        self.n_actions = int(n_actions)
        size = int(view_size)
        self.buffer = Ring(config.buffer_size, size, view_dtype)
        self.events = Ring(config.event_buffer_size, size, view_dtype)
        self.events_seen = 0
        self.version = 0
        self.last_loss = None
        self.rng = np.random.default_rng(seed)
        with torch.random.fork_rng(devices=[]):  # private init: the trainer's RNG stream is untouched
            torch.manual_seed(seed)
            self.net = nn.Sequential(nn.Linear(2 * size + self.n_actions, config.hidden), nn.ReLU(),
                                     nn.Linear(config.hidden, config.hidden // 2), nn.ReLU(),
                                     nn.Linear(config.hidden // 2, 1))
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=config.learning_rate)

    def features(self, before, actions, after):
        return features(before, actions, after, self.n_actions)

    def learn(self, before, action, after, reward):
        if not np.isfinite(reward):
            raise ValueError("Nonfinite individual reward")
        self.buffer.add(before, action, after, reward)
        if reward >= self.config.event_reward:
            self.events.add(before, action, after, reward)
            self.events_seen += 1

    def train(self):
        if self.buffer.count < self.config.batch_size:
            return None
        n_event = int(round(self.config.batch_size * self.config.event_fraction)) if self.events.count else 0
        losses = []
        for _ in range(self.config.train_steps):
            parts = [self.buffer.sample(self.config.batch_size - n_event, self.rng)]
            if n_event:
                parts.append(self.events.sample(n_event, self.rng))
            b, a, f, r = (np.concatenate(x) for x in zip(*parts))
            x = torch.as_tensor(self.features(b, a, f))
            loss = ((self.net(x).squeeze(-1) - torch.as_tensor(r)) ** 2).mean()
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            self.optimizer.step()
            losses.append(float(loss.detach()))
        self.version += 1
        self.last_loss = float(np.mean(losses))
        return self.last_loss

    @torch.inference_mode()
    def predict(self, before, actions, after):
        return self.net(torch.as_tensor(self.features(before, actions, after))).squeeze(-1).numpy().astype(np.float64)

    def event_probes(self):
        b, a, f, _ = self.events.recent(self.config.probes_per_agent)
        return b, a, f

    def mature(self):
        if self.events_seen < self.config.min_events or self.version == 0:
            return False
        b, a, f = self.event_probes()
        return bool(self.predict(b, a, f).mean() >= self.config.min_event_response)

    def state_dict(self):
        b, a, f, r = self.events.recent(self.events.count)
        return dict(net=self.net.state_dict(), events_seen=self.events_seen, version=self.version,
                    last_loss=self.last_loss, event_before=b, event_action=a, event_after=f, event_reward=r)


def response_distance(a, b):
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    mass = float(np.sum(np.abs(a) + np.abs(b)))
    return float(np.sum(np.abs(a - b)) / mass) if mass > 1e-8 else None


def action_max_distance(p, q, actions, own_i, own_j, min_count):
    """D108: largest response distance among the public actions of the probes in which BOTH agents have at
    least min_count own probe events (own_i/own_j: actions of each agent's own, unmirrored probe events).

    The pooled distance equals sum_g w_g d_g with w_g the response mass of action g, so it never exceeds this
    maximum. Returns (distance, action) or (None, None) when no action qualifies."""
    p, q, actions = np.asarray(p, dtype=np.float64), np.asarray(q, dtype=np.float64), np.asarray(actions)
    own_i, own_j = np.asarray(own_i), np.asarray(own_j)
    best = (None, None)
    for g in np.unique(actions):
        if min(np.count_nonzero(own_i == g), np.count_nonzero(own_j == g)) < min_count:
            continue
        d = response_distance(p[actions == g], q[actions == g])
        if d is not None and (best[0] is None or d > best[0]):
            best = (d, int(g))
    return best


class RewardProbeGate:
    """Pairwise probe exchange across the supplied physical links (the v2 controller protocol).

    ``__call__(update, neighbors, active)`` refits/compares every ``interval`` updates and returns a
    row-normalised legacy allocation (gate & neighbors); :meth:`approved` is the contact-independent
    symmetric approval used by Max-GAE.
    """
    model_name = "mlp_reward_event_probe_v2"

    def __init__(self, n_agents, config=None, projection=None):
        if projection is None:
            raise ValueError("RewardProbeGate needs a ProbeProjection (view size, actions, public view)")
        self.config = config or RewardGateConfig()
        self.projection = projection
        dtype = getattr(projection, "view_dtype", np.uint8)
        self.predictors = [RewardPredictor(self.config, self.config.seed * 1000 + i, projection.view_size,
                                           projection.n_actions, dtype) for i in range(n_agents)]
        self.ema = np.zeros((n_agents, n_agents), dtype=np.float64)
        self.rounds = np.zeros((n_agents, n_agents), dtype=np.int64)
        self.gate = np.zeros((n_agents, n_agents), dtype=bool)
        self.last_versions = {}
        self.last_probe_update = None
        self.active = False
        self.episode = 0
        self.last_log = {}
        self.last_metrics = {}
        self.frozen = None  # contact_snapshot: state at the start of the current rollout

    def snapshot(self):
        """contact_snapshot: freeze what each agent can send on a contact during the coming rollout.

        Predictors are refitted only inside __call__ (after a rollout), so the frozen versions are the
        weights used throughout the rollout; the probe views are copies of the recent own events."""
        self.frozen = dict(probes=[p.event_probes() for p in self.predictors],
                           mature=[p.mature() for p in self.predictors],
                           versions=[p.version for p in self.predictors])

    def observe(self, before, actions, after, rewards):
        n = len(self.predictors)
        if np.asarray(rewards).shape != (n,) or np.asarray(actions).shape != (n,):
            raise ValueError("A+C v2 consumes one separate scalar reward/action per agent")
        view = self.projection.public_view
        self.observe_views([view(o) for o in before], actions, [view(o) for o in after], rewards)

    def observe_views(self, before, actions, after, rewards):
        for i, predictor in enumerate(self.predictors):
            predictor.learn(before[i], int(actions[i]), after[i], float(rewards[i]))

    def own_probes(self, k):
        """Agent k's own recent event views (frozen at the rollout start with contact_snapshot)."""
        return self.frozen["probes"][k] if self.config.contact_snapshot else self.predictors[k].event_probes()

    def probes(self, i, j):
        parts = [self.own_probes(i), self.own_probes(j)]
        b, a, f = (np.concatenate(x) for x in zip(*parts))
        if self.config.mirror and len(b):
            mb, ma, mf = self.projection.mirror(b, a, f)
            b, a, f = (np.concatenate([b, mb]), np.concatenate([a, ma]), np.concatenate([f, mf]))
        return b, a, f

    def __call__(self, update, neighbors, active):
        neighbors = np.asarray(neighbors, dtype=bool).copy()
        np.fill_diagonal(neighbors, False)
        if neighbors.shape != self.gate.shape:
            raise ValueError("Physical neighbor graph has the wrong shape")
        self.gate &= neighbors  # lost contact never keeps a stale transfer
        if update % self.config.interval == 0 and update != self.last_probe_update:
            self.last_probe_update = update
            if self.config.contact_snapshot:
                # Compare with the rollout-start state first (sendable on contact), then refit.
                mature = self.maturity().tolist()
                details = self.compare_neighbors(neighbors, mature) if active else []
                losses = [p.train() for p in self.predictors]
            else:
                losses = [p.train() for p in self.predictors]  # also during warm-up, so votes are ready
                mature = [p.mature() for p in self.predictors]
                details = self.compare_neighbors(neighbors, mature) if active else []
            self.last_log = dict(ac_pairs=details, ac_mature=mature)
            done = [x for x in losses if x is not None]
            self.last_metrics = dict(
                ac_v2_mature_agents=float(sum(mature)),
                ac_v2_pairs_compared=float(sum(d["distance"] is not None for d in details)),
                ac_v2_pairs_accepted=float(sum(d["accepted"] for d in details)),
                ac_v2_own_events_mean=float(np.mean([p.events_seen for p in self.predictors])),
                **(dict(ac_v2_pairs_no_common_action=float(sum(d["reason"] == "no_common_action" for d in details)))
                   if self.config.distance_mode == "action_max" else {}),
                **(dict(ac_v2_predictor_loss=float(np.mean(done))) if done else {}),
                **({"ac_v2_distance_mean": float(np.mean([d["distance_ema"] for d in details
                                                          if d["distance_ema"] is not None]))}
                   if any(d["distance_ema"] is not None for d in details) else {}))
        allocation = self.gate.astype(np.float32) if active else np.zeros_like(self.ema, dtype=np.float32)
        allocation *= neighbors
        self.last_metrics["ac_v2_gate_arcs"] = float(np.count_nonzero(allocation))
        denom = allocation.sum(1, keepdims=True)
        allocation = np.divide(allocation, denom, out=np.zeros_like(allocation), where=denom > 0)
        return allocation, dict(**self.last_log, ac_model=self.model_name,
                                ac_probe_update=self.last_probe_update,
                                ac_own_events=[p.events_seen for p in self.predictors])

    def compare_neighbors(self, neighbors, mature):
        details = []
        for i in range(len(self.predictors)):
            for j in range(i + 1, len(self.predictors)):
                if neighbors[i, j] and neighbors[j, i]:
                    details.append(self.compare(i, j, mature))
        return details

    def compare(self, i, j, mature):
        row = dict(receiver=i, donor=j, probes=0, distance=None, raw_distance=None, distance_ema=None,
                   rounds=int(self.rounds[i, j]), accepted=False, reason="immature")
        if not (mature[i] and mature[j]):
            self.gate[i, j] = self.gate[j, i] = False
            return row
        if self.config.contact_snapshot and any(self.predictors[k].version != self.frozen["versions"][k]
                                                for k in (i, j)):
            raise RuntimeError("A+C v2 predictor refitted after the rollout snapshot")
        b, a, f = self.probes(i, j)
        row["probes"] = int(len(b))
        if len(b) < self.config.min_probes:
            row["reason"] = "insufficient_probes"
            self.gate[i, j] = self.gate[j, i] = False
            return row
        p, q = self.predictors[i].predict(b, a, f), self.predictors[j].predict(b, a, f)
        distance = response_distance(p, q)
        row.update(pooled_distance=distance, stratum=None)
        if self.config.distance_mode == "action_max":
            distance, row["stratum"] = action_max_distance(p, q, a, self.own_probes(i)[1], self.own_probes(j)[1],
                                                           self.config.min_probes)
            if distance is None:
                row["reason"] = "no_common_action"
                self.gate[i, j] = self.gate[j, i] = False
                return row
        versions = (self.predictors[i].version, self.predictors[j].version)
        if distance is not None and versions != self.last_versions.get((i, j)):
            # A round counts only after both predictors were refitted on new own experience.
            self.ema[i, j] = distance if not self.rounds[i, j] else (
                (1 - self.config.ema_beta) * self.ema[i, j] + self.config.ema_beta * distance)
            self.rounds[i, j] += 1
            self.last_versions[i, j] = versions
        accepted = (distance is not None and self.rounds[i, j] >= self.config.min_rounds
                    and self.ema[i, j] < self.config.distance_threshold)
        self.gate[i, j] = self.gate[j, i] = accepted
        row.update(distance=distance, raw_distance=float(np.mean(np.abs(p - q))),
                   distance_ema=float(self.ema[i, j]) if self.rounds[i, j] else None,
                   rounds=int(self.rounds[i, j]), accepted=bool(accepted),
                   reason="accepted" if accepted else "distance_or_rounds")
        return row

    def state_dict(self):
        return dict(config=asdict(self.config), predictors=[p.state_dict() for p in self.predictors],
                    ema=self.ema, rounds=self.rounds, gate=self.gate,
                    last_versions={f"{i}-{j}": v for (i, j), v in self.last_versions.items()},
                    last_probe_update=self.last_probe_update, episode=self.episode)

    def maturity(self):
        """Reward-predictor maturity, independent of contact and the last logged round: current, or with
        contact_snapshot the flags frozen at the rollout start (all False before the first snapshot)."""
        if self.config.contact_snapshot:
            flags = self.frozen["mature"] if self.frozen is not None else [False] * len(self.predictors)
            return np.asarray(flags, dtype=bool)
        return np.asarray([p.mature() for p in self.predictors], dtype=bool)

    def approved(self):
        """Symmetric reward approval, without a physical-contact mask or team labels.

        Unlike the legacy distillation gate (cleared on lost contact), this is derived from the retained
        pair evidence. Max-GAE intersects it with this rollout's contacts. Maturity comes from maturity()
        (current, or frozen at the rollout start with contact_snapshot); callers receive a fresh array.
        """
        mature = self.maturity()
        upper = np.triu((self.rounds >= self.config.min_rounds)
                        & (self.ema < self.config.distance_threshold), k=1)
        return (upper | upper.T) & mature[:, None] & mature[None, :]


def audit(controller, team_ids):
    """Same/cross accounting of the last decisions for logging. Never fed back into the gate."""
    team_ids = np.asarray(team_ids)
    rows = controller.last_log.get("ac_pairs", [])
    same = [team_ids[r["receiver"]] == team_ids[r["donor"]] for r in rows]
    out = dict(ac_v2_audit_same_accepted=float(sum(r["accepted"] and s for r, s in zip(rows, same))),
               ac_v2_audit_cross_accepted=float(sum(r["accepted"] and not s for r, s in zip(rows, same))))
    for label, flag in (("same", True), ("cross", False)):
        d = [r["distance_ema"] for r, s in zip(rows, same) if s == flag and r["distance_ema"] is not None]
        if d:
            out[f"ac_v2_audit_distance_{label}"] = float(np.mean(d))
    return out
