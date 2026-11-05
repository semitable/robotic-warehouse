"""Experience-grounded counterfactual reward probes, without team IDs or coordinates.

C supplies local transition templates. A also branches locally feasible actions from
observed scenes. Every receiver asks what *its own* reward would be. A conservative
empirical conditional-mean predictor only answers templates supported by its own
experience; there is no relabelling with the environment's hidden ownership maps.
Unknown/aliased scenes abstain. This is a nonparametric reward predictor, not a CNN.
"""
from __future__ import annotations

from collections import OrderedDict
from contextlib import contextmanager
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch


# Public, local physical features only. Channels 4/12 and ego[15:] depend
# on personal request/goal/return information and never enter a shared query.
PUBLIC_CHANNELS = (0, 1, 2, 3, 5, 6, 7, 8, 9, 10, 11)
DELTAS = ((0, -1), (0, 1), (-1, 0), (1, 0))
TURN_LEFT = (2, 3, 1, 0)
TURN_RIGHT = (3, 2, 0, 1)


@dataclass(frozen=True)
class ACConfig:
    radius: int = 2
    max_entries: int = 4096
    max_queries: int = 64
    min_samples: int = 8
    min_episodes: int = 3
    min_queries: int = 4
    max_std: float = .10
    max_stderr: float = .04
    distance_threshold: float = .12
    min_signal: float = .20
    ema_beta: float = .35
    min_rounds: int = 8
    interval: int = 5

    def __post_init__(self):
        if min(self.radius, self.max_entries, self.max_queries, self.min_samples,
               self.min_episodes, self.min_queries, self.min_rounds, self.interval) < 1:
            raise ValueError("AC counts and radius must be positive")
        if self.min_queries > self.max_queries or not 0 < self.ema_beta <= 1:
            raise ValueError("Invalid AC query count or EMA beta")
        if min(self.max_std, self.max_stderr, self.distance_threshold, self.min_signal) <= 0:
            raise ValueError("AC uncertainty, distance and signal bounds must be positive")


def strip_coordinates(obs):
    result = np.asarray(obs, dtype=np.float32).copy()
    result[..., :2] = 0
    return result


def split_observation(obs):
    obs = np.asarray(obs, dtype=np.float32)
    side = int(round(np.sqrt((obs.size - 20) / 13)))
    if obs.ndim != 1 or side * side * 13 + 20 != obs.size or side % 2 != 1:
        raise ValueError("AC requires the 20-ego/13-channel semantic observation")
    return obs[:20], obs[20:].reshape(13, side, side)


def public_view(obs, radius):
    ego, grid = split_observation(obs)
    center = grid.shape[-1] // 2
    if center < radius:
        raise ValueError("AC radius exceeds the local sensor window")
    patch = grid[list(PUBLIC_CHANNELS), center-radius:center+radius+1,
                 center-radius:center+radius+1]
    # Loaded, direction, highway: interpreted as the hypothetical performing self.
    return np.concatenate((ego[2:8], patch.reshape(-1))).astype(np.float32)


@dataclass
class Query:
    before: np.ndarray
    action: int
    after: np.ndarray
    source: str  # experience / virtual_local; never includes rewards or team labels

    @property
    def key(self):
        data = self.before.astype('<f4').tobytes() + bytes([self.action]) + self.after.astype('<f4').tobytes()
        return hashlib.sha256(data).hexdigest()


def experience_query(before, action, after, radius):
    if int(action) not in range(5):
        raise ValueError("Invalid action")
    return Query(public_view(before, radius), int(action), public_view(after, radius), "experience")


def virtual_queries(obs, radius):
    """Feasible local one-step branches with all other agents staying still.

    Forward requires one extra visible border, a free target and no loaded-shelf
    collision. We do not invent pickups, request status, reward or hidden returns.
    Delivery is crossing onto an observed generic goal while carrying a shelf.
    Terminal transitions, hidden map edges and dock/wall variants are not simulated.
    """
    ego, grid = split_observation(obs)
    center = grid.shape[-1] // 2
    if center < radius + 1:
        return []
    heading = int(np.argmax(ego[3:7]))
    result = []
    for action in range(4):
        new_ego, new_grid = ego.copy(), grid.copy()
        dx = dy = 0
        if action == 1:
            dx, dy = DELTAS[heading]
            y, x = center + dy, center + dx
            if grid[0, y, x] or grid[5, y, x] or (ego[2] and grid[3, y, x]):
                continue
        direction = TURN_LEFT[heading] if action == 2 else TURN_RIGHT[heading] if action == 3 else heading
        # Erase the old self, then center the hypothetical self at the new pose.
        new_grid[5:11, center, center] = 0
        if ego[2] and action == 1:
            new_grid[3, center, center] = 0
            new_grid[3, center+dy, center+dx] = 1
        # Crop from the original larger window. No unobserved padding is generated.
        patch = new_grid[:, center+dy-radius:center+dy+radius+1,
                         center+dx-radius:center+dx+radius+1].copy()
        patch[5, radius, radius] = 1
        patch[6+direction, radius, radius] = 1
        patch[10, radius, radius] = ego[2]
        new_ego[3:7] = 0
        new_ego[3+direction] = 1
        new_ego[7] = patch[1, radius, radius]
        after = np.concatenate((new_ego[2:8], patch[list(PUBLIC_CHANNELS)].reshape(-1))).astype(np.float32)
        result.append(Query(public_view(obs, radius), action, after, "virtual_local"))
    return result


@dataclass
class RewardStats:
    count: int = 0
    mean: float = 0.
    m2: float = 0.
    episodes: set = None

    def add(self, reward, episode, episode_cap):
        self.count += 1
        delta = reward - self.mean
        self.mean += delta / self.count
        self.m2 += delta * (reward - self.mean)
        if self.episodes is None:
            self.episodes = set()
        if len(self.episodes) < episode_cap:
            self.episodes.add(int(episode))

    def response(self, config):
        if self.count < config.min_samples or len(self.episodes or ()) < config.min_episodes:
            return None
        std = np.sqrt(max(self.m2 / max(self.count - 1, 1), 0.))
        stderr = std / np.sqrt(self.count)
        if std > config.max_std or stderr > config.max_stderr:
            return None
        return self.mean, float(stderr)


class LocalRewardPredictor:
    """phi_i(q): exact public-scene lookup in agent i's OWN local experience.

    Conditional means estimate personal rewards. Private request flags are neither
    copied from a sender nor supplied by the environment. Conflicting hidden
    meanings increase variance and cause abstention, rather than oracle relabelling.
    """
    def __init__(self, config):
        self.config = config
        self.memory = OrderedDict()
        self.bank = OrderedDict()
        self.observed_bank = OrderedDict()
        self.version = 0

    def learn(self, query, own_reward, episode):
        if not np.isfinite(own_reward):
            raise ValueError("Nonfinite individual reward")
        key = query.key
        stats = self.memory.setdefault(key, RewardStats())
        stats.add(float(own_reward), episode, self.config.min_episodes)
        self.memory.move_to_end(key)
        self.observed_bank[key] = query
        self.observed_bank.move_to_end(key)
        if len(self.memory) > self.config.max_entries:
            evicted, _ = self.memory.popitem(last=False)
            self.observed_bank.pop(evicted, None)
        self.offer(query)
        self.version += 1

    def offer(self, query):
        self.bank[query.key] = query
        self.bank.move_to_end(query.key)
        if len(self.bank) > self.config.max_entries:
            self.bank.popitem(last=False)

    def predict(self, query):
        stats = self.memory.get(query.key)
        return None if stats is None else stats.response(self.config)

    def public_queries(self):
        # A's branches must not evict C's rare, genuinely experienced deliveries.
        return {**self.bank, **self.observed_bank}

    def state_dict(self):
        # Primitive records keep checkpoint loading independent of whether this
        # module was imported as a package or by a direct-script runner.
        def queries(bank):
            return [dict(before=q.before, action=q.action, after=q.after, source=q.source)
                    for q in bank.values()]
        return dict(version=self.version,
                    memory={key: dict(count=s.count, mean=s.mean, m2=s.m2, episodes=sorted(s.episodes or ()))
                            for key, s in self.memory.items()},
                    bank=queries(self.bank), observed_bank=queries(self.observed_bank))


class ACController:
    """Pairwise query/response exchange only across the supplied physical links."""
    def __init__(self, n_agents, config=None):
        self.config = config or ACConfig()
        self.predictors = [LocalRewardPredictor(self.config) for _ in range(n_agents)]
        self.ema = np.zeros((n_agents, n_agents), dtype=np.float64)
        self.rounds = np.zeros((n_agents, n_agents), dtype=np.int64)
        self.gate = np.zeros((n_agents, n_agents), dtype=bool)
        self.last_versions = {}
        self.last_probe_update = None
        self.active = False
        self.episode = 0
        self.last_log = {}

    def observe(self, before, actions, after, rewards):
        n = len(self.predictors)
        if np.asarray(rewards).shape != (n,) or np.asarray(actions).shape != (n,):
            raise ValueError("AC consumes one separate scalar reward/action per agent")
        for i, predictor in enumerate(self.predictors):
            query = experience_query(before[i], actions[i], after[i], self.config.radius)
            predictor.learn(query, rewards[i], self.episode)
            for virtual in virtual_queries(before[i], self.config.radius):
                predictor.offer(virtual)  # virtual rewards are never used as labels

    def __call__(self, update, neighbors, active):
        neighbors = np.asarray(neighbors, dtype=bool).copy()
        np.fill_diagonal(neighbors, False)
        if neighbors.shape != self.gate.shape:
            raise ValueError("Physical neighbor graph has the wrong shape")
        self.gate &= neighbors  # lost contact never keeps a stale transfer
        details = []
        if active and update % self.config.interval == 0 and update != self.last_probe_update:
            self.last_probe_update = update
            for i in range(len(self.predictors)):
                for j in range(i+1, len(self.predictors)):
                    if not (neighbors[i, j] and neighbors[j, i]):
                        continue
                    p, q = self.predictors[i], self.predictors[j]
                    # Pool only this pair's public queries, never training labels.
                    bank = {**p.public_queries(), **q.public_queries()}
                    supported = []
                    for key in sorted(bank):
                        query = bank[key]
                        a, b = p.predict(query), q.predict(query)
                        if a is not None and b is not None:
                            supported.append((key, a, b, query.source))
                    # Deterministic sample after support filtering, so unsupported
                    # clutter cannot displace the rare mutually supported events.
                    if len(supported) > self.config.max_queries:
                        indices = np.linspace(0, len(supported)-1, self.config.max_queries, dtype=int)
                        supported = [supported[k] for k in indices]
                    row = dict(receiver=i, donor=j, offered=len(bank), supported=len(supported),
                               virtual_supported=sum(s[3] == "virtual_local" for s in supported),
                               reason="insufficient_support", raw_distance=None, distance_ema=None)
                    accepted = False
                    if len(supported) >= self.config.min_queries:
                        a = np.array([s[1] for s in supported])
                        b = np.array([s[2] for s in supported])
                        distance = float(np.mean(np.abs(a[:, 0]-b[:, 0])))
                        uncertainty = float(2*np.mean(a[:, 1]+b[:, 1]))
                        signal = min(float(np.ptp(a[:, 0])), float(np.ptp(b[:, 0])))
                        # Only new evidence for the actually compared templates
                        # advances confidence; unrelated rollouts do not count.
                        versions = tuple((s[0], p.memory[s[0]].count, q.memory[s[0]].count)
                                         for s in supported)
                        fresh = versions != self.last_versions.get((i, j))
                        if fresh:
                            self.ema[i, j] = distance if not self.rounds[i, j] else (
                                (1-self.config.ema_beta)*self.ema[i, j] + self.config.ema_beta*distance)
                            self.rounds[i, j] += 1
                            self.last_versions[i, j] = versions
                        accepted = (signal >= self.config.min_signal and
                                    self.rounds[i, j] >= self.config.min_rounds and
                                    max(distance, self.ema[i, j]) + uncertainty < self.config.distance_threshold)
                        row.update(raw_distance=distance, distance_ema=float(self.ema[i, j]),
                                   uncertainty=uncertainty, signal=signal, rounds=int(self.rounds[i, j]),
                                   reason="accepted" if accepted else "distance_signal_or_evidence")
                    self.gate[i, j] = self.gate[j, i] = accepted
                    details.append(row)
            self.last_log = dict(ac_pairs=details)
        allocation = self.gate.astype(np.float32) if active else np.zeros_like(self.ema, dtype=np.float32)
        allocation *= neighbors
        denom = allocation.sum(1, keepdims=True)
        allocation = np.divide(allocation, denom, out=np.zeros_like(allocation), where=denom > 0)
        return allocation, dict(**self.last_log, ac_model="local_empirical_reward",
                                ac_probe_update=self.last_probe_update,
                                ac_own_samples=[p.version for p in self.predictors])

    def state_dict(self):
        return dict(config=asdict(self.config), predictors=[p.state_dict() for p in self.predictors], ema=self.ema,
                    rounds=self.rounds, gate=self.gate, last_versions=self.last_versions,
                    last_probe_update=self.last_probe_update, episode=self.episode)


class LocalObservationEnv(gym.Wrapper):
    """Coordinate-free policy/evaluation observations; capture pre-reset transitions."""
    def __init__(self, env, controller=None):
        super().__init__(env)
        self.controller = controller
        self.previous = None

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.previous = strip_coordinates(obs)
        if self.controller is not None and self.controller.active:
            self.controller.episode += 1
        return tuple(self.previous.copy()), info

    def step(self, actions):
        obs, rewards, terminated, truncated, info = self.env.step(actions)
        after = strip_coordinates(obs)
        if self.controller is not None and self.controller.active:
            self.controller.observe(self.previous, actions, after, rewards)
        self.previous = after
        return tuple(after.copy()), rewards, terminated, truncated, info


@contextmanager
def install(cfg, run_dir=None, config=None, *, learn=True):
    """Scoped adapter: original trainer/environment source and old modes unchanged."""
    from examples import train_recurrent_ippo_consensus as T

    env_kwargs = json.loads(cfg.env_kwargs_json)
    if env_kwargs.get("team_reward_mode") != "individual":
        raise ValueError("A+C requires explicit individual rewards; team/ALL is forbidden")
    if cfg.observation_format != "semantic" or cfg.sensor_range < (config or ACConfig()).radius + 1:
        raise ValueError("A+C needs semantic observations and a visible simulation border")
    if any(env_kwargs.get(k) for k in ("wall_cells", "dock_cooldown_steps")) or "constrained" in cfg.env_id:
        raise ValueError("Local A simulation currently supports the standard warehouse only")
    originals = {name: getattr(T, name) for name in
                 ("make_env", "collect_rollout", "sample_policy_probe_sequences", "save_checkpoint")}
    controller = ACController(cfg.agent_count or env_kwargs["n_agents"], config) if learn else None

    def make_env(*args, **kwargs):
        return LocalObservationEnv(originals["make_env"](*args, **kwargs), controller)

    def collect(*args, **kwargs):
        if controller is not None:
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

    def save(path, *args, **kwargs):
        originals["save_checkpoint"](path, *args, **kwargs)
        payload = torch.load(path, map_location="cpu", weights_only=False)
        payload["ac_observation"] = "self_x/self_y masked to zero; input dimension retained"
        if controller is not None:
            payload["reward_probe_ac"] = controller.state_dict()
        torch.save(payload, path)

    T.make_env, T.collect_rollout, T.sample_policy_probe_sequences, T.save_checkpoint = make_env, collect, sample, save
    try:
        if run_dir is not None:
            Path(run_dir).mkdir(parents=True, exist_ok=True)
            (Path(run_dir)/"reward_probe_ac.json").write_text(json.dumps(dict(
                config=asdict(config or ACConfig()), learning=learn,
                reward="individual", coordinates="masked", predictor="empirical_conditional_mean",
                unsupported="abstain"), indent=2), encoding="utf-8")
        yield controller
    finally:
        for name, original in originals.items():
            setattr(T, name, original)
