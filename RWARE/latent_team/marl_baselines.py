"""MAPPO / SePS / HetDPS baselines that reuse the existing RWARE trainer loop unchanged.

The original ``T.train`` still creates the environment, collects rollouts with
``collect_rollout``, evaluates with ``evaluate``, logs to W&B/metrics.jsonl and
saves checkpoints. Inside one process this module only substitutes
(1) the per-agent network factory with views onto one learner and
(2) ``update_ippo`` with the algorithm's own update. Environment, rewards,
observations, action masks, evaluation seeds and budgets are therefore identical
to the IPPO/PGCT runs. Differences from the original papers are listed in
``latent_team/BASELINES.md``.
"""
from __future__ import annotations

import json
import math
import time
from contextlib import contextmanager
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical

try:
    from . import common as C  # noqa: F401  (puts the repository root on sys.path)
except ImportError:
    import common as C  # noqa: F401

from examples import train_recurrent_ippo_consensus as T

ORIGINAL_NET = T.RecurrentActorCritic
ALGORITHMS = ["mappo", "seps", "hetdps"]


def _init(module, head_gain):
    """Same initialisation as the trainer's RecurrentActorCritic."""
    for m in module.modules():
        if isinstance(m, (nn.Linear, nn.Conv2d)):
            nn.init.orthogonal_(m.weight, gain=math.sqrt(2))
            nn.init.constant_(m.bias, 0.0)
    nn.init.orthogonal_(module.head.weight, gain=head_gain)


def _encoder(spec, cfg):
    return T.SemanticCNNEncoder(obs_dim=spec.obs_dim, spatial_channels=spec.spatial_channels,
                                spatial_size=spec.spatial_size, ego_dim=spec.ego_dim,
                                hidden_dim=cfg.mlp_hidden_dim, output_dim=cfg.recurrent_hidden_dim)


class RecurrentNet(nn.Module):
    """Same CNN encoder + GRUCell + linear head as the trainer; optional agent-ID input."""

    def __init__(self, spec, cfg, out_dim, id_dim=0, head_gain=0.01):
        super().__init__()
        hidden = cfg.recurrent_hidden_dim
        self.encoder = _encoder(spec, cfg)
        self.id_proj = nn.Linear(hidden + id_dim, hidden) if id_dim else None
        self.rnn = nn.GRUCell(hidden, hidden)
        self.head = nn.Linear(hidden, out_dim)
        _init(self, head_gain)

    def step(self, obs, h, ids=None):
        x = self.encoder(obs)
        if self.id_proj is not None:
            x = torch.relu(self.id_proj(torch.cat([x, ids], dim=-1)))
        h = self.rnn(x, h)
        return self.head(h), h


class CentralValue(nn.Module):
    """MAPPO centralised V: all agents' local observations (shared encoder) + evaluated agent's ID."""

    def __init__(self, spec, cfg, n_agents):
        super().__init__()
        hidden = cfg.recurrent_hidden_dim
        self.n_agents = n_agents
        self.encoder = _encoder(spec, cfg)
        self.state_proj = nn.Linear(n_agents * hidden + n_agents, hidden)
        self.rnn = nn.GRUCell(hidden, hidden)
        self.head = nn.Linear(hidden, 1)
        _init(self, 1.0)

    def step(self, joint_obs, h, ids):
        """joint_obs [B, N, D] (every agent's observation), ids [B, N] one-hot of the valued agent."""
        b, n, d = joint_obs.shape
        z = self.encoder(joint_obs.reshape(b * n, d)).reshape(b, -1)
        h = self.rnn(torch.relu(self.state_proj(torch.cat([z, ids], dim=-1))), h)
        return self.head(h).squeeze(-1), h


class ValueNorm(nn.Module):
    """MAPPO (marlbenchmark/on-policy) ValueNorm: debiased running mean/var, beta=0.99999."""

    def __init__(self, beta=0.99999, epsilon=1e-5):
        super().__init__()
        self.beta, self.epsilon = beta, epsilon
        for name in ("running_mean", "running_mean_sq", "debiasing_term"):
            self.register_buffer(name, torch.zeros(1))

    def stats(self):
        debias = self.debiasing_term.clamp(min=self.epsilon)
        mean, mean_sq = self.running_mean / debias, self.running_mean_sq / debias
        return mean, (mean_sq - mean ** 2).clamp(min=1e-2)

    @torch.no_grad()
    def update(self, x):
        x = x.reshape(-1).float()
        self.running_mean.mul_(self.beta).add_(x.mean() * (1 - self.beta))
        self.running_mean_sq.mul_(self.beta).add_((x ** 2).mean() * (1 - self.beta))
        self.debiasing_term.mul_(self.beta).add_(1 - self.beta)

    def normalize(self, x):
        mean, var = self.stats()
        return (x - mean) / var.sqrt()

    def denormalize(self, x):
        mean, var = self.stats()
        return x * var.sqrt() + mean


def masked(logits, mask):
    return ORIGINAL_NET._masked_logits(None, logits, mask)


class AgentView(nn.Module):
    """Duck-typed RecurrentActorCritic for one agent, backed by the shared learner.

    View 0 registers the learner's modules so .to(device), state_dict() and the
    trainer's checkpoints include every network; other views hold a frozen placeholder.
    """

    initial_hidden = ORIGINAL_NET.initial_hidden
    _masked_logits = ORIGINAL_NET._masked_logits
    act = ORIGINAL_NET.act
    action_probabilities = ORIGINAL_NET.action_probabilities
    max_action_probability = ORIGINAL_NET.max_action_probability

    def __init__(self, learner, agent_id, obs_dim, action_dim, hidden):
        super().__init__()
        object.__setattr__(self, "learner", learner)
        self.agent_id, self.obs_dim, self.action_dim, self.recurrent_hidden_dim = agent_id, obs_dim, action_dim, hidden
        if agent_id == 0:
            self.learner_modules = learner.modules
        else:
            self.placeholder = nn.Parameter(torch.zeros(1), requires_grad=False)

    def _actor_step(self, obs, h):
        return self.learner.actor_step(self.agent_id, obs, h)

    def _critic_step(self, obs, h):
        return self.learner.local_value_step(self.agent_id, obs, h)


class Learner:
    """Grouped recurrent PPO with an agent -> network assignment (MAPPO: one shared actor)."""

    def __init__(self, algo, cfg, spec, n_agents, action_dim, run_dir):
        self.algo, self.cfg, self.n, self.action_dim, self.run_dir = algo, cfg, n_agents, action_dim, Path(run_dir)
        mods = nn.ModuleDict()
        if algo == "mappo":
            mods["actors"] = nn.ModuleList([RecurrentNet(spec, cfg, action_dim, id_dim=n_agents)])
            mods["central_critic"] = CentralValue(spec, cfg, n_agents)
            mods["value_norm"] = ValueNorm()
            self.assign = [0] * n_agents
        else:
            # SePS / HetDPS: one actor and one local critic per possible network (laac_size = N).
            mods["actors"] = nn.ModuleList([RecurrentNet(spec, cfg, action_dim) for _ in range(n_agents)])
            mods["critics"] = nn.ModuleList([RecurrentNet(spec, cfg, 1, head_gain=1.0) for _ in range(n_agents)])
            self.assign = [0] * n_agents
        self.modules = mods
        self.optimizers = None
        self.central_h = None
        self.recorder = dict(timeouts={}, steps=0, last_obs=None)  # replaced by install() with the env's
        self.update_count = 0
        self.env_steps = 0
        self.log = None

    # ---------------- acting (used by collect_rollout / evaluate through AgentView) ----------------
    def device(self):
        return next(self.modules.parameters()).device

    def ids(self, agent_ids, batch, device):
        return torch.eye(self.n, device=device)[agent_ids].expand(batch, self.n) if isinstance(agent_ids, int) \
            else torch.eye(self.n, device=device)[agent_ids]

    def actor_step(self, agent_id, obs, h):
        net = self.modules["actors"][self.assign[agent_id]]
        ids = self.ids(agent_id, obs.shape[0], obs.device) if self.algo == "mappo" else None
        return net.step(obs, h, ids)

    def local_value_step(self, agent_id, obs, h):
        if self.algo == "mappo":
            # Placeholder only: centralised values are recomputed from all observations before the update.
            return torch.zeros(obs.shape[0], device=obs.device), h
        value, h = self.modules["critics"][self.assign[agent_id]].step(obs, h)
        return value.squeeze(-1), h

    def build_optimizers(self):
        lr, eps = self.cfg.learning_rate, 1e-5
        if self.algo == "mappo":
            # MAPPO keeps separate actor and critic optimisers.
            self.optimizers = dict(actor=torch.optim.Adam(self.modules["actors"].parameters(), lr=lr, eps=eps),
                                   critic=torch.optim.Adam(self.modules["central_critic"].parameters(), lr=lr, eps=eps))
        else:
            self.optimizers = dict(all=torch.optim.Adam(
                list(self.modules["actors"].parameters()) + list(self.modules["critics"].parameters()), lr=lr, eps=eps))

    # ---------------- MAPPO centralised value recomputation ----------------
    @torch.no_grad()
    def central_values(self, rollout, device):
        critic, norm = self.modules["central_critic"], self.modules["value_norm"]
        obs = rollout.obs.to(device)
        steps = obs.shape[0]
        hidden = self.cfg.recurrent_hidden_dim
        ids = torch.eye(self.n, device=device)
        h = self.central_h if self.central_h is not None else torch.zeros(self.n, hidden, device=device)
        values = torch.zeros(steps, self.n, device=device)
        timeout = torch.zeros(steps, self.n, device=device)
        hs = torch.zeros(steps, self.n, hidden, device=device)
        first = self.recorder["steps"] - steps
        for t in range(steps):
            hs[t] = h
            joint = obs[t].unsqueeze(0).expand(self.n, *obs[t].shape)
            v, h_next = critic.step(joint, h, ids)
            values[t] = v
            next_obs = self.recorder["timeouts"].get(first + t)
            if next_obs is not None:
                joint_next = torch.as_tensor(next_obs, device=device).unsqueeze(0).expand(self.n, -1, -1)
                tv, _ = critic.step(joint_next, h_next, ids)
                timeout[t] = norm.denormalize(tv)
            h = h_next * (1.0 - float(rollout.dones[t]))
        last = torch.as_tensor(self.recorder["last_obs"], device=device).unsqueeze(0).expand(self.n, -1, -1)
        last_v, _ = critic.step(last, h, ids)
        self.central_h = h
        self.recorder["timeouts"] = {k: v for k, v in self.recorder["timeouts"].items() if k >= self.recorder["steps"]}
        return values, hs, timeout, norm.denormalize(last_v)

    # ---------------- PPO update ----------------
    def update(self, rollout, device, rng):
        cfg = self.cfg
        if self.optimizers is None:
            self.build_optimizers()
        self.update_count += 1
        self.env_steps += rollout.obs.shape[0]
        self.before_update(rollout, device)
        if self.algo == "mappo":
            old_values, critic_h, timeout_values, last_values = self.central_values(rollout, device)
            norm = self.modules["value_norm"]
            gae_values = norm.denormalize(old_values).cpu()
            advantages, returns = T.compute_gae(rollout.rewards, rollout.dones, gae_values, last_values.cpu(),
                                                cfg.gamma, cfg.gae_lambda, timeout_values.cpu())
            old_values, critic_h = old_values.cpu(), critic_h.cpu()
            value_estimate = gae_values
        else:
            value_estimate = rollout.values
            old_values, critic_h = rollout.values, rollout.critic_h
            advantages, returns = T.compute_gae(rollout.rewards, rollout.dones, rollout.values, rollout.last_values,
                                                cfg.gamma, cfg.gae_lambda, rollout.timeout_values)
        if cfg.normalize_advantages:
            # MAPPO / HetDPS normalise over the whole batch (all agents together).
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-5)
        stats = {k: [] for k in ("policy_loss", "value_loss", "entropy", "approx_kl", "clip_fraction")}
        first_gap = None
        steps, length = rollout.obs.shape[0], cfg.sequence_length
        for _ in range(cfg.ppo_epochs):
            starts = np.arange(0, steps, length)
            rng.shuffle(starts)
            for offset in range(0, len(starts), cfg.minibatch_chunks):
                chunk = starts[offset:offset + cfg.minibatch_chunks]
                self._value_gap = 0.0
                result = self.minibatch_loss(rollout, chunk, advantages, returns, old_values, critic_h, device)
                if first_gap is None:
                    # Before any parameter step the replayed values must equal the rollout values.
                    first_gap = result["value_replay_gap"]
                for k in stats:
                    stats[k].append(result[k])
        metrics = {k: float(np.mean(v)) for k, v in stats.items()}
        rewards = rollout.rewards
        metrics.update(reward_density=float((rewards.abs() > cfg.reward_epsilon).float().mean()),
                       positive_reward_density=float((rewards > cfg.reward_epsilon).float().mean()),
                       actor_update_skipped=0.0, networks_in_use=float(len(set(self.assign))),
                       value_replay_gap_first_minibatch=float(first_gap),
                       mean_abs_td_error=float((returns - value_estimate).abs().mean()))
        for k, net in enumerate(self.assign):
            metrics[f"agent_{k}/network"] = float(net)
        metrics.update(self.after_update(rollout, device))
        return metrics

    def sequences(self, tensor, chunk, length, agents=None):
        """[T, N, ...] -> [L, len(chunk) * len(agents), ...] (chunk-major, then agent)."""
        agents = range(self.n) if agents is None else agents
        return torch.stack([tensor[s:s + length, a] for s in chunk for a in agents], dim=1)

    def minibatch_loss(self, rollout, chunk, advantages, returns, old_values, critic_h, device):
        cfg, length = self.cfg, self.cfg.sequence_length
        groups = {}
        for agent, net in enumerate(self.assign):
            groups.setdefault(net, []).append(agent)
        policy_terms, entropy_terms, kl_terms, clip_terms, value_terms = [], [], [], [], []
        for net_index, agents in groups.items():
            obs = self.sequences(rollout.obs, chunk, length, agents).to(device)
            actions = self.sequences(rollout.actions, chunk, length, agents).to(device)
            masks = self.sequences(rollout.action_masks, chunk, length, agents).to(device)
            old_logp = self.sequences(rollout.old_log_probs, chunk, length, agents).to(device)
            adv = self.sequences(advantages, chunk, length, agents).to(device)
            done = torch.stack([rollout.dones[s:s + length] for s in chunk for _ in agents], dim=1).to(device)
            h = torch.stack([rollout.actor_h[s, a] for s in chunk for a in agents]).to(device)
            ids = torch.eye(self.n, device=device)[[a for _ in chunk for a in agents]] if self.algo == "mappo" else None
            net = self.modules["actors"][net_index]
            logps, ents = [], []
            for t in range(obs.shape[0]):
                logits, h = net.step(obs[t], h, ids)
                dist = Categorical(logits=masked(logits, masks[t]))
                logps.append(dist.log_prob(actions[t]))
                ents.append(dist.entropy())
                h = h * (1.0 - done[t]).unsqueeze(-1)
            logp, ent = torch.stack(logps), torch.stack(ents)
            ratio = torch.exp(logp - old_logp)
            policy_terms.append(-torch.min(ratio * adv, ratio.clamp(1 - cfg.clip_coef, 1 + cfg.clip_coef) * adv))
            entropy_terms.append(ent)
            kl_terms.append(old_logp - logp)
            clip_terms.append(((ratio - 1).abs() > cfg.clip_coef).float())
            if self.algo != "mappo":
                value_terms.append(self.local_value_loss(net_index, rollout, chunk, agents, returns, old_values,
                                                         critic_h, device))
        policy_loss = torch.cat([x.reshape(-1) for x in policy_terms]).mean()
        entropy = torch.cat([x.reshape(-1) for x in entropy_terms]).mean()
        if self.algo == "mappo":
            actor_loss = policy_loss - cfg.entropy_coef * entropy
            self.optimizers["actor"].zero_grad()
            actor_loss.backward()
            nn.utils.clip_grad_norm_(self.modules["actors"].parameters(), cfg.max_grad_norm)
            self.optimizers["actor"].step()
            value_loss = self.central_value_loss(rollout, chunk, returns, old_values, critic_h, device)
            self.optimizers["critic"].zero_grad()
            (value_loss * cfg.value_loss_coef).backward()
            nn.utils.clip_grad_norm_(self.modules["central_critic"].parameters(), cfg.max_grad_norm)
            self.optimizers["critic"].step()
        else:
            value_loss = torch.cat([x.reshape(-1) for x in value_terms]).mean()
            loss = policy_loss + cfg.value_loss_coef * value_loss - cfg.entropy_coef * entropy
            self.optimizers["all"].zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(list(self.modules["actors"].parameters()) +
                                     list(self.modules["critics"].parameters()), cfg.max_grad_norm)
            self.optimizers["all"].step()
        return dict(policy_loss=policy_loss.item(), value_loss=value_loss.item(), entropy=entropy.item(),
                    approx_kl=torch.cat([x.reshape(-1) for x in kl_terms]).mean().item(),
                    clip_fraction=torch.cat([x.reshape(-1) for x in clip_terms]).mean().item(),
                    value_replay_gap=self._value_gap)

    def central_value_loss(self, rollout, chunk, returns, old_values, critic_h, device):
        """MAPPO value loss: ValueNorm-normalised targets, value clipping, Huber (delta 10), max of both."""
        cfg, length, norm = self.cfg, self.cfg.sequence_length, self.modules["value_norm"]
        critic = self.modules["central_critic"]
        agents = list(range(self.n))
        joint = torch.stack([rollout.obs[s:s + length] for s in chunk for _ in agents], dim=1).to(device)
        ids = torch.eye(self.n, device=device)[[a for _ in chunk for a in agents]]
        done = torch.stack([rollout.dones[s:s + length] for s in chunk for _ in agents], dim=1).to(device)
        h = torch.stack([critic_h[s, a] for s in chunk for a in agents]).to(device)
        values = []
        for t in range(joint.shape[0]):
            v, h = critic.step(joint[t], h, ids)
            values.append(v)
            h = h * (1.0 - done[t]).unsqueeze(-1)
        values = torch.stack(values)
        old = self.sequences(old_values, chunk, length).to(device)
        self._value_gap = (values - old).abs().max().item()
        target = self.sequences(returns, chunk, length).to(device)
        norm.update(target)
        target = norm.normalize(target)
        clipped = old + (values - old).clamp(-cfg.value_clip_coef, cfg.value_clip_coef)
        huber = nn.functional.huber_loss
        return torch.max(huber(clipped, target, reduction="none", delta=10.0),
                         huber(values, target, reduction="none", delta=10.0)).mean()

    def local_value_loss(self, net_index, rollout, chunk, agents, returns, old_values, critic_h, device):
        """Per-network local critic with PPO value clipping (as in HetDPS / the existing IPPO trainer)."""
        cfg, length = self.cfg, self.cfg.sequence_length
        critic = self.modules["critics"][net_index]
        obs = self.sequences(rollout.obs, chunk, length, agents).to(device)
        done = torch.stack([rollout.dones[s:s + length] for s in chunk for _ in agents], dim=1).to(device)
        h = torch.stack([critic_h[s, a] for s in chunk for a in agents]).to(device)
        values = []
        for t in range(obs.shape[0]):
            v, h = critic.step(obs[t], h)
            values.append(v.squeeze(-1))
            h = h * (1.0 - done[t]).unsqueeze(-1)
        values = torch.stack(values)
        old = self.sequences(old_values, chunk, length, agents).to(device)
        self._value_gap = max(self._value_gap, (values - old).abs().max().item())
        target = self.sequences(returns, chunk, length, agents).to(device)
        clipped = old + (values - old).clamp(-cfg.value_clip_coef, cfg.value_clip_coef)
        return 0.5 * torch.max((values - target) ** 2, (clipped - target) ** 2)

    # ---------------- algorithm hooks ----------------
    def before_update(self, rollout, device):
        pass

    def after_update(self, rollout, device):
        return {}

    def extra_state(self):
        return dict(algorithm=self.algo, assign=list(self.assign), update_count=self.update_count,
                    env_steps=self.env_steps)


class ObsRecorder(gym.Wrapper):
    """Training-env wrapper: keeps the joint observation needed by a centralised critic."""

    def __init__(self, env, recorder):
        super().__init__(env)
        self.recorder = recorder

    def _flat(self, obs):
        return np.asarray(T.flatten_multi_agent_obs(self.env, obs), dtype=np.float32)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.recorder["last_obs"] = self._flat(obs)
        return obs, info

    def step(self, actions):
        obs, rewards, terminated, truncated, info = self.env.step(actions)
        rec = self.recorder
        if bool(np.any(truncated)) and not bool(np.any(terminated)):
            rec["timeouts"][rec["steps"]] = self._flat(obs)
        rec["steps"] += 1
        rec["last_obs"] = self._flat(obs)
        return obs, rewards, terminated, truncated, info


def make_learner(algo, cfg, spec, n_agents, action_dim, run_dir):
    if algo == "mappo":
        return Learner(algo, cfg, spec, n_agents, action_dim, run_dir)
    if algo == "seps":
        return SePSLearner(cfg, spec, n_agents, action_dim, run_dir)
    if algo == "hetdps":
        return HetDPSLearner(cfg, spec, n_agents, action_dim, run_dir)
    raise ValueError(f"Unknown baseline {algo}")


@contextmanager
def install(cfg, algo, run_dir):
    """Swap the trainer's network factory/update for one baseline inside this process only."""
    if algo not in ALGORITHMS:
        raise ValueError(algo)
    if cfg.communication_enabled or cfg.peer_transfer_mode != "none" or cfg.init_checkpoint:
        raise ValueError("Baselines run without PGCT communication, peer transfer or initial checkpoints.")
    run_dir = Path(run_dir)
    state = dict(learner=None, views=0, train_env=None, recorder=dict(timeouts={}, steps=0, last_obs=None))
    originals = {name: getattr(T, name) for name in ("RecurrentActorCritic", "update_ippo", "make_env", "save_checkpoint")}
    spec_holder = {}

    def factory(obs_dim, action_dim, mlp_hidden_dim=128, recurrent_hidden_dim=128, encoder_type="mlp",
                spatial_channels=0, spatial_size=0, ego_dim=0):
        if encoder_type != "cnn":
            raise ValueError("Baselines use the semantic CNN encoder of the main experiments.")
        if state["learner"] is None:
            spec = T.get_semantic_observation_spec(state["train_env"])
            state["learner"] = make_learner(algo, cfg, spec, cfg.agent_count, action_dim, run_dir)
            state["learner"].recorder = state["recorder"]
        if state["views"] >= cfg.agent_count:
            raise RuntimeError("Unexpected extra network construction inside the trainer.")
        view = AgentView(state["learner"], state["views"], obs_dim, action_dim, recurrent_hidden_dim)
        state["views"] += 1
        return view

    def make_env(*args, **kwargs):
        env = originals["make_env"](*args, **kwargs)
        if state["train_env"] is None:  # the first environment train() creates is the training env
            env = ObsRecorder(env, state["recorder"])
            state["train_env"] = env
        return env

    def update(agents, optimizers, rollout, config, graph_estimator, device, rng, peer_allocation=None,
               peer_probe_sequences=None):
        learner = state["learner"]
        if learner.update_count == 0:
            from examples.diagnose_pgct_rollout_compatibility import model_hash
            (run_dir / "initial_weights_sha256.txt").write_text(model_hash(agents) + "\n", encoding="utf-8")
        start = time.perf_counter()
        metrics = learner.update(rollout, device, rng)
        metrics["baseline_update_seconds"] = time.perf_counter() - start
        return metrics

    def save_checkpoint(path, *args, **kwargs):
        originals["save_checkpoint"](path, *args, **kwargs)
        if state["learner"] is not None:
            torch.save(state["learner"].extra_state(), Path(path).with_suffix(".baseline.pt"))

    T.RecurrentActorCritic, T.update_ippo, T.make_env, T.save_checkpoint = factory, update, make_env, save_checkpoint
    try:
        yield state
    finally:
        for name, func in originals.items():
            setattr(T, name, func)
        if state["learner"] is not None:
            (run_dir / "baseline_state.json").write_text(json.dumps(state["learner"].extra_state(), indent=2,
                                                                     default=str), encoding="utf-8")


# ============ clustering (numpy re-implementations of the scikit-learn defaults used upstream) ============
def kmeans(x, k, *, n_init=10, max_iter=300, tol=1e-4, seed=0):
    """KMeans(init='k-means++', n_init=10): best of n_init Lloyd runs by inertia."""
    x = np.asarray(x, dtype=np.float64)
    rng = np.random.default_rng(seed)
    tol = tol * np.mean(np.var(x, axis=0))  # scikit-learn scales tol by the mean feature variance
    best = None
    for _ in range(n_init):
        centers = [x[rng.integers(len(x))]]
        for _ in range(1, k):  # k-means++ seeding
            d2 = np.min(((x[:, None] - np.asarray(centers)[None]) ** 2).sum(-1), axis=1)
            probs = d2 / d2.sum() if d2.sum() > 0 else np.full(len(x), 1 / len(x))
            centers.append(x[rng.choice(len(x), p=probs)])
        centers = np.asarray(centers)
        for _ in range(max_iter):
            labels = np.argmin(((x[:, None] - centers[None]) ** 2).sum(-1), axis=1)
            new = np.asarray([x[labels == c].mean(0) if np.any(labels == c) else centers[c] for c in range(k)])
            shift = ((new - centers) ** 2).sum()
            centers = new
            if shift <= tol:
                break
        labels = np.argmin(((x[:, None] - centers[None]) ** 2).sum(-1), axis=1)
        inertia = ((x - centers[labels]) ** 2).sum()
        if best is None or inertia < best[0]:
            best = (inertia, labels)
    return best[1]


def davies_bouldin(x, labels):
    x, labels = np.asarray(x, dtype=np.float64), np.asarray(labels)
    ids = np.unique(labels)
    centers = np.asarray([x[labels == c].mean(0) for c in ids])
    spread = np.asarray([np.linalg.norm(x[labels == c] - centers[i], axis=1).mean() for i, c in enumerate(ids)])
    dist = np.linalg.norm(centers[:, None] - centers[None], axis=-1)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = (spread[:, None] + spread[None]) / dist
    np.fill_diagonal(ratio, -np.inf)
    ratio[~np.isfinite(ratio)] = 0.0 if len(ids) == 1 else ratio[~np.isfinite(ratio)]
    return float(np.max(ratio, axis=1).mean())


def affinity_propagation(similarity, *, damping=0.5, max_iter=200, convergence_iter=15, seed=0):
    """AffinityPropagation(affinity='precomputed') defaults: preference = median similarity."""
    s = np.array(similarity, dtype=np.float64)
    n = len(s)
    if n == 1:
        return np.zeros(1, dtype=int)
    rng = np.random.RandomState(seed)
    s.flat[::n + 1] = np.median(s)
    # scikit-learn removes degeneracies with tiny noise
    s += (np.finfo(float).eps * s + np.finfo(float).tiny * 100) * rng.standard_normal((n, n))
    a, r = np.zeros((n, n)), np.zeros((n, n))
    history = np.zeros((n, convergence_iter))
    ind = np.arange(n)
    for it in range(max_iter):
        tmp = a + s
        i_max = np.argmax(tmp, axis=1)
        y_max = tmp[ind, i_max]
        tmp[ind, i_max] = -np.inf
        y2 = np.max(tmp, axis=1)
        tmp = s - y_max[:, None]
        tmp[ind, i_max] = s[ind, i_max] - y2
        r = damping * r + (1 - damping) * tmp
        tmp = np.maximum(r, 0)
        tmp.flat[::n + 1] = r.flat[::n + 1]
        tmp -= tmp.sum(axis=0)
        dA = np.diag(tmp).copy()
        tmp = np.clip(tmp, 0, np.inf)
        tmp.flat[::n + 1] = dA
        a = damping * a - (1 - damping) * tmp
        exemplar = (np.diag(a) + np.diag(r)) > 0
        history[:, it % convergence_iter] = exemplar
        k = exemplar.sum()
        if it >= convergence_iter:
            se = history.sum(axis=1)
            if (not np.sum((se == convergence_iter) + (se == 0)) != n) and k > 0:
                break
    centers = np.flatnonzero(exemplar)
    if len(centers) == 0:  # scikit-learn: no convergence -> every sample its own cluster label -1; use singletons
        return np.arange(n)
    labels = np.argmax(s[:, centers], axis=1)
    labels[centers] = np.arange(len(centers))
    for k in range(len(centers)):  # refine exemplars as in scikit-learn
        members = np.flatnonzero(labels == k)
        centers[k] = members[np.argmax(s[np.ix_(members, members)].sum(axis=0))]
    labels = np.argmax(s[:, centers], axis=1)
    labels[centers] = np.arange(len(centers))
    return np.unique(labels, return_inverse=True)[1]


# ===================================== SePS =====================================
class LinearVAE(nn.Module):
    """SePS (uoe-agents/seps model.py) LinearVAE: encoder(agent one-hot) -> (mu, logvar); decoder([z, obs, act])."""

    def __init__(self, features, input_size, extra_decoder_input, reconstruct_size, hidden=64):
        super().__init__()
        self.features = features
        self.encoder = nn.Sequential(nn.Linear(input_size, hidden), nn.ReLU(), nn.Linear(hidden, 2 * features))
        self.decoder = nn.Sequential(nn.Linear(features + extra_decoder_input, hidden), nn.ReLU(),
                                     nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, reconstruct_size))

    def encode(self, x):
        return self.encoder(x)[:, :self.features]

    def forward(self, x, xp):
        out = self.encoder(x)
        mu, log_var = out[:, :self.features], out[:, self.features:]
        z = mu + torch.randn_like(mu) * torch.exp(0.5 * log_var)
        return self.decoder(torch.cat([z, xp], dim=-1)), mu, log_var


class TransitionBuffer:
    """Uniform reservoir of per-agent (agent, obs, act, rew, next_obs) transitions; obs stored as float16."""

    def __init__(self, capacity, seed):
        self.capacity, self.rng, self.seen, self.items = capacity, np.random.default_rng(seed), 0, []

    def add_rollout(self, rollout, n_agents):
        obs, acts, rews, dones = rollout.obs, rollout.actions, rollout.rewards, rollout.dones
        for t in range(obs.shape[0] - 1):
            if float(dones[t]) > 0:  # the next stored observation belongs to a new episode
                continue
            for k in range(n_agents):
                item = (k, obs[t, k].half(), int(acts[t, k]), float(rews[t, k]), obs[t + 1, k].half())
                self.seen += 1
                if len(self.items) < self.capacity:
                    self.items.append(item)
                else:
                    j = int(self.rng.integers(0, self.seen))
                    if j < self.capacity:
                        self.items[j] = item

    def tensors(self, n_agents):
        agent = torch.eye(n_agents)[[it[0] for it in self.items]]
        obs = torch.stack([it[1] for it in self.items]).float()
        act = torch.tensor([[float(it[2])] for it in self.items])
        rew = torch.tensor([[it[3]] for it in self.items])
        nxt = torch.stack([it[4] for it in self.items]).float()
        return agent, obs, act, rew, nxt


def train_vae(vae, inputs, conditions, targets, *, epochs, batch_size, lr, kl_weight, seed, device, optimizer=None):
    """Official SePS/HetDPS VAE objective: MSE(sum) reconstruction + kl_weight * KLD(sum)."""
    optimizer = optimizer or torch.optim.Adam(vae.parameters(), lr=lr)
    gen = torch.Generator().manual_seed(seed)
    losses = []
    for _ in range(epochs):
        for idx in torch.randperm(len(inputs), generator=gen).split(batch_size):
            recon, mu, log_var = vae(inputs[idx].to(device), conditions[idx].to(device))
            kld = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
            loss = nn.functional.mse_loss(recon, targets[idx].to(device), reduction="sum") + kl_weight * kld
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
    return losses, optimizer


class SePSLearner(Learner):
    """Selective Parameter Sharing (Christianos et al., ICML 2021) on the common recurrent PPO learner.

    Official defaults (ops_utils.py / ac.py): pretraining_steps=5000 with 8 parallel envs (= 40,000 env
    steps here), z_features=10, kl_weight=1e-4, epochs=10, batch_size=128, lr=3e-4, encoder_in=[agent],
    decoder_in=[obs, act], reconstruct=[next_obs, rew], clusters=None -> Davies-Bouldin over K=2..N-1,
    KMeans(k-means++, n_init=10). RL training continues during pretraining (delay_training=False) and the
    pretraining steps are part of the budget. Before clustering agents are randomly mapped to networks
    (agent 0 -> network 0); after clustering the networks are not re-initialised.
    """

    PRETRAIN_ENV_STEPS = 5000 * 8
    BUFFER_CAPACITY = 50000

    def __init__(self, cfg, spec, n_agents, action_dim, run_dir, clusters=None):
        super().__init__("seps", cfg, spec, n_agents, action_dim, run_dir)
        self.pretrain_env_steps = self.PRETRAIN_ENV_STEPS
        if cfg.total_timesteps <= 64:  # smoke test: exercise the clustering path within the tiny budget
            self.pretrain_env_steps = cfg.rollout_steps
        self.clusters, self.clustered, self.cluster_record = clusters, False, None
        self.assign_rng = np.random.default_rng(cfg.seed + 70001)
        self.buffer = TransitionBuffer(self.BUFFER_CAPACITY, cfg.seed + 70002)
        self.sample_assignment()

    def sample_assignment(self):
        # model.sample_laac: Categorical(logits=ones(N-1, N)); agent 0 is fixed to network 0.
        self.assign = [0] + [int(x) for x in self.assign_rng.integers(0, self.n, self.n - 1)]

    def before_update(self, rollout, device):
        if not self.clustered:
            self.buffer.add_rollout(rollout, self.n)

    def after_update(self, rollout, device):
        metrics = dict(seps_clustered=float(self.clustered))
        if not self.clustered and self.env_steps >= self.pretrain_env_steps:
            metrics.update(self.cluster(device))
        elif not self.clustered:
            # One environment instead of 8 parallel ones: resample the pre-clustering mapping per rollout.
            self.sample_assignment()
        return metrics

    def cluster(self, device):
        start = time.perf_counter()
        agent, obs, act, rew, nxt = self.buffer.tensors(self.n)
        vae = LinearVAE(10, self.n, obs.shape[1] + 1, nxt.shape[1] + 1).to(device)
        losses, _ = train_vae(vae, agent, torch.cat([obs, act], -1), torch.cat([nxt, rew], -1), epochs=10,
                              batch_size=128, lr=3e-4, kl_weight=1e-4, seed=self.cfg.seed + 70003, device=device)
        with torch.no_grad():
            z = vae.encode(torch.eye(self.n, device=device)).cpu().numpy()
        scores = {}
        if self.clusters is None:
            for k in range(2, self.n):
                labels = kmeans(z, k, seed=self.cfg.seed)
                scores[k] = davies_bouldin(z, labels) if len(set(labels)) > 1 else float("inf")
            k_best = min(scores, key=scores.get)
        else:
            k_best = self.clusters
        labels = kmeans(z, k_best, seed=self.cfg.seed)
        self.assign = [int(x) for x in labels]
        self.clustered = True
        seconds = time.perf_counter() - start
        self.cluster_record = dict(env_steps=self.env_steps, transitions=len(agent), transitions_seen=self.buffer.seen,
                                   vae_final_loss=losses[-1], vae_parameters=sum(p.numel() for p in vae.parameters()),
                                   davies_bouldin=scores, clusters=k_best, assign=self.assign,
                                   latent_means=z.tolist(), seconds=seconds)
        (self.run_dir / "seps_clustering.json").write_text(json.dumps(self.cluster_record, indent=2), encoding="utf-8")
        self.buffer = None
        return dict(seps_clusters=float(k_best), seps_cluster_seconds=seconds, seps_vae_loss=losses[-1])

    def extra_state(self):
        return dict(super().extra_state(), pretrain_env_steps=self.pretrain_env_steps, clustered=self.clustered,
                    cluster_record=self.cluster_record)


# ===================================== HetDPS =====================================
class BiHierarchicalVAE(nn.Module):
    """HetDPS (Harry67Hu/HetDPS model_Het.py) BiHierarchicalVAE, HIDDEN=32, verbatim structure."""

    def __init__(self, features, id_size, condition_size, reconstruct_size, hidden=32):
        super().__init__()
        self.features = features
        self.id_encoder = nn.Sequential(nn.Linear(id_size, hidden), nn.ReLU(), nn.Linear(hidden, hidden))
        self.deep_encoder = nn.Sequential(nn.Linear(condition_size + hidden, hidden * 2), nn.ReLU(),
                                          nn.Linear(hidden * 2, hidden * 2), nn.ReLU(),
                                          nn.Linear(hidden * 2, 2 * features))
        self.id_decoder = nn.Sequential(nn.Linear(id_size, hidden), nn.ReLU(), nn.Linear(hidden, hidden))
        self.deep_decoder = nn.Sequential(nn.Linear(condition_size + hidden + features, hidden * 2), nn.ReLU(),
                                          nn.Linear(hidden * 2, hidden * 2), nn.ReLU(),
                                          nn.Linear(hidden * 2, reconstruct_size))

    def encode(self, id_input, condition):
        out = self.deep_encoder(torch.cat([self.id_encoder(id_input), condition], dim=-1))
        mu, log_var = out[..., :self.features], out[..., self.features:]
        return mu + torch.randn_like(mu) * torch.exp(0.5 * log_var), mu, log_var

    def forward(self, id_input, condition):
        z, mu, log_var = self.encode(id_input, condition)
        # decoder_in = condition (local_state, act), as in the official configuration
        return self.deep_decoder(torch.cat([self.id_decoder(id_input), z, condition], dim=-1)), mu, log_var


def gaussian_w2(mus, log_vars):
    """calculate_N_Gaussians_WD: diagonal-Gaussian 2-Wasserstein, [B, N, D] -> mean over B of [N, N]."""
    std = torch.exp(log_vars) ** 0.5
    term1 = ((mus.unsqueeze(1) - mus.unsqueeze(2)) ** 2).sum(-1)
    term2 = ((std.unsqueeze(1) - std.unsqueeze(2)) ** 2).sum(-1)
    return (term1 + term2) ** 0.5


def max_overlap_assignment(overlap):
    """linear_sum_assignment(-overlap) for the small (<= N x N) overlap matrix: exact permutation search."""
    import itertools
    rows, cols = overlap.shape
    best, best_pairs = -np.inf, []
    if rows <= cols:
        for perm in itertools.permutations(range(cols), rows):
            score = sum(overlap[i, perm[i]] for i in range(rows))
            if score > best:
                best, best_pairs = score, [(i, perm[i]) for i in range(rows)]
    else:
        for perm in itertools.permutations(range(rows), cols):
            score = sum(overlap[perm[j], j] for j in range(cols))
            if score > best:
                best, best_pairs = score, [(perm[j], j) for j in range(cols)]
    return best_pairs


def dynamic_parameter_sharing(wd, previous_clustering, previous_networks, copy, n_networks, seed):
    """Port of HetDPS compute_dynamic_parameter_sharing (merge_mode='majority').

    ``copy(src, dst)`` copies actor and critic parameters like Policy.copy_parameters.
    """
    from collections import Counter
    new = affinity_propagation(-np.asarray(wd), seed=seed)
    unique_new = np.unique(new)
    if previous_clustering is None:
        return new, new.copy()
    old, net_old = np.asarray(previous_clustering), np.asarray(previous_networks)
    unique_old = np.unique(old)
    overlap = np.zeros((len(unique_new), len(unique_old)))
    for i in range(len(new)):
        overlap[np.flatnonzero(unique_new == new[i])[0], np.flatnonzero(unique_old == old[i])[0]] += 1
    old_to_net = {c: Counter(net_old[old == c]).most_common(1)[0][0] for c in unique_old}
    net_new = np.zeros_like(new)
    if len(unique_old) == len(unique_new):
        mapping = {unique_new[i]: unique_old[j] for i, j in max_overlap_assignment(overlap)}
        for i in range(len(new)):
            net_new[i] = old_to_net.get(mapping.get(new[i], new[i]), new[i])
    elif len(unique_old) < len(unique_new):  # split: unmatched clusters get a new network copied from a source
        mapping = {unique_new[i]: unique_old[j] for i, j in max_overlap_assignment(overlap) if overlap[i, j] > 0}
        to_net, next_id = {}, int(max(net_old)) + 1
        for c in unique_new:
            if c in mapping:
                to_net[c] = old_to_net[mapping[c]]
                continue
            target = next_id if next_id < n_networks else min(set(range(n_networks)) - set(to_net.values())
                                                              - set(old_to_net.values()) or {next_id % n_networks})
            row = overlap[np.flatnonzero(unique_new == c)[0]]
            if row.max() > 0:
                source = old_to_net[unique_old[int(np.argmax(row))]]
            else:
                center = np.flatnonzero(new == c)[0]
                source = min(((float(wd[center, np.flatnonzero(old == oc)[0]]), old_to_net[oc]) for oc in unique_old))[1]
            copy(source, target)
            to_net[c] = target
            next_id += 1
        net_new = np.asarray([to_net[c] for c in new])
    else:  # merge: renumber networks 0..K-1, majority network's parameters survive
        pairs = max_overlap_assignment(overlap.T)  # (old index, new index)
        old_to_new = {unique_old[i]: unique_new[j] for i, j in pairs if overlap[j, i] > 0}
        to_net = {}
        for target, c in enumerate(unique_new):
            to_net[c] = target
            sources = [old_to_net[oc] for oc, nc in old_to_new.items() if nc == c]
            if len(sources) > 1:
                copy(max(sources, key=lambda n: int(np.sum(net_old == n))), target)
            elif len(sources) == 1 and sources[0] != target:
                copy(sources[0], target)
        net_new = np.asarray([to_net[c] for c in new])
    return new, net_new


class HetDPSLearner(Learner):
    """Heterogeneity-based Dynamic Parameter Sharing (HetDPS, AAMAS 2026) on the common recurrent PPO learner.

    Official configuration (main_Het.py / utils_Het.py): all agents start on network 0; a FIFO replay buffer
    keeps max_rb_steps(20) x parallel_envs(32) x n_steps(5) = 3,200 env steps of per-agent transitions;
    every reparameter_interval(20) updates (= 3,200 env steps) the BiHierarchicalVAE is trained further
    (z=40, lr=5e-4, 10 epochs, batch 128, kl=1e-4; encoder id=agent, condition=[local_state, act one-hot],
    reconstruct=[next_local_state, next_obs, rew_pad]); the meta-heterogeneity Wasserstein matrix is the mean
    over all buffered local states x all discrete actions; AffinityPropagation(-WD) and the dynamic
    parameter-sharing matching (merge_mode='majority') re-assign networks.
    """

    INTERVAL_ENV_STEPS = 20 * 5 * 32
    BUFFER_ENV_STEPS = 20 * 32 * 5

    def __init__(self, cfg, spec, n_agents, action_dim, run_dir):
        super().__init__("hetdps", cfg, spec, n_agents, action_dim, run_dir)
        self.interval = self.INTERVAL_ENV_STEPS if cfg.total_timesteps > 64 else cfg.rollout_steps  # smoke
        self.buffer_steps = self.BUFFER_ENV_STEPS
        self.items = []  # (agent, obs16, act, rew, next_obs16), FIFO of buffer_steps * N entries
        self.vae, self.previous, self.measurements, self.next_measure = None, None, [], self.interval
        self.assign = [0] * n_agents

    def before_update(self, rollout, device):
        obs, acts, rews, dones = rollout.obs, rollout.actions, rollout.rewards, rollout.dones
        for t in range(obs.shape[0] - 1):
            if float(dones[t]) > 0:
                continue
            for k in range(self.n):
                self.items.append((k, obs[t, k].half(), int(acts[t, k]), float(rews[t, k]), obs[t + 1, k].half()))
        self.items = self.items[-self.buffer_steps * self.n:]

    def copy(self, src, dst):
        if src == dst:
            return
        for group in ("actors", "critics"):
            self.modules[group][dst].load_state_dict(self.modules[group][src].state_dict())

    def after_update(self, rollout, device):
        if self.env_steps < self.next_measure or not self.items:
            return {}
        self.next_measure += self.interval
        start = time.perf_counter()
        n, a = self.n, self.action_dim
        agent = torch.eye(n)[[it[0] for it in self.items]]
        obs = torch.stack([it[1] for it in self.items]).float()
        act = torch.eye(a)[[it[2] for it in self.items]]
        rew = torch.tensor([[it[3]] for it in self.items])
        nxt = torch.stack([it[4] for it in self.items]).float()
        condition = torch.cat([obs, act], -1)
        target = torch.cat([nxt, nxt, rew.expand(-1, obs.shape[1])], -1)  # next_local_state, next_obs, rew_pad
        if self.vae is None:
            self.vae = BiHierarchicalVAE(40, n, condition.shape[1], target.shape[1]).to(device)
        losses, _ = train_vae(self.vae, agent, condition, target, epochs=10, batch_size=128, lr=5e-4,
                              kl_weight=1e-4, seed=self.cfg.seed + 80000 + len(self.measurements), device=device)
        wd_sum, count = torch.zeros(n, n, device=device), 0
        with torch.no_grad():  # "fixed_obs_all_acts": every buffered local state with every discrete action
            states = obs.to(device)
            for chunk in states.split(max(1, 10000 // a)):
                cond = torch.cat([chunk.unsqueeze(1).expand(-1, a, -1),
                                  torch.eye(a, device=device).unsqueeze(0).expand(len(chunk), -1, -1)], -1)
                cond = cond.reshape(-1, cond.shape[-1])
                ids = torch.eye(n, device=device).unsqueeze(1).expand(-1, len(cond), -1)
                _, mus, log_vars = self.vae.encode(ids, cond.unsqueeze(0).expand(n, -1, -1))
                wd_sum += gaussian_w2(mus.transpose(0, 1), log_vars.transpose(0, 1)).sum(0)
                count += len(cond)
        wd = (wd_sum / count).cpu().numpy()
        old_assign = list(self.assign)
        clusters, networks = dynamic_parameter_sharing(
            wd, None if self.previous is None else self.previous[0], None if self.previous is None else self.previous[1],
            self.copy, n, seed=self.cfg.seed)
        self.previous = (clusters, networks)
        self.assign = [int(x) for x in networks]
        seconds = time.perf_counter() - start
        record = dict(env_steps=self.env_steps, transitions=len(self.items), conditions=count,
                      vae_final_loss=losses[-1], wd=wd.tolist(), clusters=clusters.tolist(),
                      previous_assign=old_assign, assign=self.assign, seconds=seconds)
        self.measurements.append(record)
        with (self.run_dir / "hetdps_measurements.jsonl").open("a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")
        off = wd[~np.eye(n, dtype=bool)]
        return dict(hetdps_measure_seconds=seconds, hetdps_clusters=float(len(set(clusters.tolist()))),
                    hetdps_vae_loss=losses[-1], hetdps_mean_wd=float(off.mean()),
                    hetdps_vae_parameters=float(sum(p.numel() for p in self.vae.parameters())))

    def extra_state(self):
        return dict(super().extra_state(), measurements=len(self.measurements), interval_env_steps=self.interval,
                    last_measurement=self.measurements[-1] if self.measurements else None)
