"""A+C gate v2: generalising personal reward predictors queried on shared reward-event probes.

v1 (reward_probe_ac.py) answers only exactly repeated public transitions and abstains when the reward
of a transition varies. Offline (output/ac_offline_20261026) that removed every delivery, the only
team-discriminating event. v2 keeps the v1 information rules and changes the model and the probes:

- Predictor: each agent fits a small MLP r_hat_i(public before, action, public after) to its OWN
  scalar reward, from a bounded buffer of its own transitions. Its reward events are oversampled.
- Probes (C): the public views of each neighbour's own recent reward events (|r| >= event_reward).
  Only the views are exchanged, never a reward. (A): the left-right mirror of those events.
- Abstention: an agent votes only when its predictor is mature (enough own events, and it predicts
  them). An agent that never earned a reward has no reward response to compare and abstains.
- Distance: sum|a-b| / sum(|a|+|b|) over the probes (0 = same responses, 1 = disjoint); EMA over
  rounds; same team if below .5, i.e. most of the response mass is shared. Scale-free; not fitted
  to any run.
Team labels never enter a decision. The install() logger audits same/cross decisions afterwards
from the environment info, for W&B only. Private RNGs leave the trainer's random stream untouched.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict, dataclass
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

try:
    from . import reward_probe_ac as AC
except ImportError:
    import reward_probe_ac as AC

EGO = 6          # public ego: carrying, up, down, left, right, highway
N_ACTIONS = 5    # NOOP, FORWARD, LEFT, RIGHT, TOGGLE_LOAD
LEFT, RIGHT = 2, 3
DIR_LEFT, DIR_RIGHT = 7, 8  # public channel positions of the visible agents' left/right heading


@dataclass(frozen=True)
class ACV2Config:
    radius: int = 2
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
    mirror: bool = True
    min_probes: int = 8
    distance_threshold: float = .5
    ema_beta: float = .35
    min_rounds: int = 3
    interval: int = 5
    seed: int = 0
    # D102 (ii): compare only what each agent held at the rollout start (predictor, recent own event views,
    # maturity), i.e. what it could send on a contact during that rollout; refit after the comparison.
    contact_snapshot: bool = False

    def __post_init__(self):
        counts = (self.radius, self.buffer_size, self.event_buffer_size, self.hidden, self.train_steps,
                  self.batch_size, self.min_events, self.probes_per_agent, self.min_probes, self.min_rounds,
                  self.interval)
        if min(counts) < 1:
            raise ValueError("A+C v2 counts and radius must be positive")
        if not 0 < self.ema_beta <= 1 or not 0 <= self.event_fraction <= 1:
            raise ValueError("Invalid EMA beta or event fraction")
        if min(self.event_reward, self.learning_rate, self.distance_threshold) <= 0:
            raise ValueError("Event reward, learning rate and distance threshold must be positive")


def view_size(radius):
    return EGO + len(AC.PUBLIC_CHANNELS) * (2 * radius + 1) ** 2


def mirror_view(views, radius):
    """Left-right mirror of public views [..., V]: a feasible variant of the same local event."""
    views = np.array(views, copy=True)
    side = 2 * radius + 1
    ego = views[..., :EGO].copy()
    ego[..., [3, 4]] = ego[..., [4, 3]]
    grid = views[..., EGO:].reshape(views.shape[:-1] + (len(AC.PUBLIC_CHANNELS), side, side))[..., ::-1].copy()
    grid[..., [DIR_LEFT, DIR_RIGHT], :, :] = grid[..., [DIR_RIGHT, DIR_LEFT], :, :]
    views[..., :EGO] = ego
    views[..., EGO:] = grid.reshape(views.shape[:-1] + (-1,))
    return views


def mirror_action(actions):
    actions = np.asarray(actions)
    return np.where(actions == LEFT, RIGHT, np.where(actions == RIGHT, LEFT, actions))


def features(before, actions, after):
    before = np.asarray(before, dtype=np.float32)
    onehot = np.eye(N_ACTIONS, dtype=np.float32)[np.asarray(actions, dtype=np.int64)]
    return np.concatenate([before, onehot, np.asarray(after, dtype=np.float32)], axis=-1)


class Ring:
    """Fixed-size FIFO of (before, action, after, reward) public transitions."""
    def __init__(self, capacity, size):
        self.before = np.zeros((capacity, size), np.uint8)
        self.after = np.zeros((capacity, size), np.uint8)
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
    def __init__(self, config, seed):
        self.config = config
        size = view_size(config.radius)
        self.buffer = Ring(config.buffer_size, size)
        self.events = Ring(config.event_buffer_size, size)
        self.events_seen = 0
        self.version = 0
        self.last_loss = None
        self.rng = np.random.default_rng(seed)
        with torch.random.fork_rng(devices=[]):  # private init: the trainer's RNG stream is untouched
            torch.manual_seed(seed)
            self.net = nn.Sequential(nn.Linear(2 * size + N_ACTIONS, config.hidden), nn.ReLU(),
                                     nn.Linear(config.hidden, config.hidden // 2), nn.ReLU(),
                                     nn.Linear(config.hidden // 2, 1))
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=config.learning_rate)

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
            x = torch.as_tensor(features(b, a, f))
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
        return self.net(torch.as_tensor(features(before, actions, after))).squeeze(-1).numpy().astype(np.float64)

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


class ACV2Controller:
    """Pairwise probe exchange across the supplied physical links; same call protocol as v1."""
    def __init__(self, n_agents, config=None):
        self.config = config or ACV2Config()
        self.predictors = [RewardPredictor(self.config, self.config.seed * 1000 + i) for i in range(n_agents)]
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
        r = self.config.radius
        self.observe_views([AC.public_view(o, r) for o in before], actions,
                           [AC.public_view(o, r) for o in after], rewards)

    def observe_views(self, before, actions, after, rewards):
        for i, predictor in enumerate(self.predictors):
            predictor.learn(before[i], int(actions[i]), after[i], float(rewards[i]))

    def probes(self, i, j):
        if self.config.contact_snapshot:
            parts = [self.frozen["probes"][i], self.frozen["probes"][j]]
        else:
            parts = [self.predictors[i].event_probes(), self.predictors[j].event_probes()]
        b, a, f = (np.concatenate(x) for x in zip(*parts))
        if self.config.mirror and len(b):
            r = self.config.radius
            b, a, f = (np.concatenate([b, mirror_view(b, r)]), np.concatenate([a, mirror_action(a)]),
                       np.concatenate([f, mirror_view(f, r)]))
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
                **(dict(ac_v2_predictor_loss=float(np.mean(done))) if done else {}),
                **({"ac_v2_distance_mean": float(np.mean([d["distance_ema"] for d in details
                                                          if d["distance_ema"] is not None]))}
                   if any(d["distance_ema"] is not None for d in details) else {}))
        allocation = self.gate.astype(np.float32) if active else np.zeros_like(self.ema, dtype=np.float32)
        allocation *= neighbors
        self.last_metrics["ac_v2_gate_arcs"] = float(np.count_nonzero(allocation))
        denom = allocation.sum(1, keepdims=True)
        allocation = np.divide(allocation, denom, out=np.zeros_like(allocation), where=denom > 0)
        return allocation, dict(**self.last_log, ac_model="mlp_reward_event_probe_v2",
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

        Unlike the legacy distillation gate (cleared on lost contact), this is derived from
        the retained pair evidence. Max-GAE intersects it with this rollout's contacts.
        Maturity comes from maturity() (current, or frozen at the rollout start with contact_snapshot), also
        between probe rounds; callers receive a fresh array.
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


class AuditedObservationEnv(AC.LocalObservationEnv):
    """v1's coordinate-free wrapper; also keeps the latest team ids from info for the audit log only."""
    def __init__(self, env, controller=None):
        super().__init__(env, controller)
        self.audit_team_ids = None

    def step(self, actions):
        obs, rewards, terminated, truncated, info = self.env.step(actions)
        if isinstance(info, dict) and info.get("agent_team_ids") is not None:
            self.audit_team_ids = np.asarray(info["agent_team_ids"]).copy()
        after = AC.strip_coordinates(obs)
        if self.controller is not None and self.controller.active:
            self.controller.observe(self.previous, actions, after, rewards)
        self.previous = after
        return tuple(after.copy()), rewards, terminated, truncated, info


@contextmanager
def install(cfg, run_dir=None, config=None, *, learn=True):
    """Scoped adapter like reward_probe_ac.install; adds update/ac_v2_* metrics for W&B."""
    from examples import train_recurrent_ippo_consensus as T

    config = config or ACV2Config()
    env_kwargs = json.loads(cfg.env_kwargs_json)
    if env_kwargs.get("team_reward_mode") != "individual":
        raise ValueError("A+C requires explicit individual rewards; team/ALL is forbidden")
    if cfg.observation_format != "semantic" or cfg.sensor_range < config.radius:
        raise ValueError("A+C v2 needs semantic observations covering the probe radius")
    originals = {name: getattr(T, name) for name in
                 ("make_env", "collect_rollout", "sample_policy_probe_sequences", "save_checkpoint", "update_ippo")}
    controller = ACV2Controller(cfg.agent_count or env_kwargs["n_agents"], config) if learn else None
    envs = []

    def make_env(*args, **kwargs):
        env = AuditedObservationEnv(originals["make_env"](*args, **kwargs), controller)
        envs.append(env)
        return env

    def collect(*args, **kwargs):
        if controller is not None:
            if controller.config.contact_snapshot:
                controller.snapshot()  # before this rollout's first step (and before any contact in it)
            controller.active = True
        try:
            return originals["collect_rollout"](*args, **kwargs)
        finally:
            if controller is not None:
                controller.active = False

    def sample(*args, **kwargs):
        probes, metadata = originals["sample_policy_probe_sequences"](*args, **kwargs)
        probes = probes.clone()
        probes[..., :2] = 0
        return probes, metadata

    def update(*args, **kwargs):
        metrics = originals["update_ippo"](*args, **kwargs)
        if controller is not None and isinstance(metrics, dict):
            metrics.update(controller.last_metrics)
            teams = next((e.audit_team_ids for e in reversed(envs) if e.audit_team_ids is not None), None)
            if teams is not None:
                metrics.update(audit(controller, teams))
        return metrics

    def save(path, *args, **kwargs):
        originals["save_checkpoint"](path, *args, **kwargs)
        payload = torch.load(path, map_location="cpu", weights_only=False)
        payload["ac_observation"] = "self_x/self_y masked to zero; input dimension retained"
        if controller is not None:
            payload["reward_probe_ac_v2"] = controller.state_dict()
        torch.save(payload, path)

    T.make_env, T.collect_rollout, T.sample_policy_probe_sequences = make_env, collect, sample
    T.save_checkpoint, T.update_ippo = save, update
    try:
        if run_dir is not None:
            Path(run_dir).mkdir(parents=True, exist_ok=True)
            (Path(run_dir) / "reward_probe_ac_v2.json").write_text(json.dumps(dict(
                config=asdict(config), learning=learn, reward="individual", coordinates="masked",
                predictor="mlp_own_reward", probes="own reward events + mirror",
                comparison=("rollout-start snapshot, compared before refit" if config.contact_snapshot
                            else "refit, then compare"),
                distance="sum|a-b|/sum(|a|+|b|)", unsupported="abstain"), indent=2), encoding="utf-8")
        yield controller
    finally:
        for name, original in originals.items():
            setattr(T, name, original)
