from __future__ import annotations

import argparse
import json
import math
import random
import re
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

import gymnasium as gym
import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import rware  # noqa: F401 - registers Gymnasium environments
from rware.utils.probe_maps import (
    build_rware_agent_conditioned_probe_bank as _build_rware_agent_probe_map_bank,
    build_rware_objective_probe_bank as _build_rware_probe_map_bank,
)
from rware.utils.semantic_observation import (
    RwareSemanticObservationWrapper,
    get_semantic_observation_spec,
)
from rware.warehouse import Action, Direction, _LAYER_AGENTS, _LAYER_SHELFS


TensorPair = Tuple[torch.Tensor, torch.Tensor]


def mlp(input_dim: int, hidden_dim: int, output_dim: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.Tanh(),
        nn.Linear(hidden_dim, hidden_dim),
        nn.Tanh(),
        nn.Linear(hidden_dim, output_dim),
        nn.Tanh(),
    )


class SemanticCNNEncoder(nn.Module):
    """Small CNN encoder for ego || CxSxS semantic RWARE observations."""

    def __init__(
        self,
        obs_dim: int,
        spatial_channels: int,
        spatial_size: int,
        ego_dim: int,
        hidden_dim: int,
        output_dim: int,
    ) -> None:
        super().__init__()
        self.obs_dim = int(obs_dim)
        self.spatial_channels = int(spatial_channels)
        self.spatial_size = int(spatial_size)
        self.ego_dim = int(ego_dim)
        spatial_dim = self.spatial_channels * self.spatial_size * self.spatial_size
        if self.ego_dim + spatial_dim != self.obs_dim:
            raise ValueError(
                "semantic CNN layout does not match obs_dim: "
                f"ego={self.ego_dim}, spatial={spatial_dim}, obs_dim={self.obs_dim}"
            )

        self.spatial_cnn = nn.Sequential(
            nn.Conv2d(self.spatial_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Flatten(),
        )
        self.spatial_fc = nn.Sequential(
            nn.Linear(32 * self.spatial_size * self.spatial_size, hidden_dim),
            nn.Tanh(),
        )
        ego_hidden = max(hidden_dim // 2, 16)
        self.ego_encoder = nn.Sequential(
            nn.Linear(self.ego_dim, ego_hidden),
            nn.Tanh(),
        )
        self.output = nn.Sequential(
            nn.Linear(hidden_dim + ego_hidden, output_dim),
            nn.Tanh(),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        if obs.ndim == 1:
            obs = obs.unsqueeze(0)
        ego = obs[..., : self.ego_dim]
        spatial = obs[..., self.ego_dim :].reshape(
            -1,
            self.spatial_channels,
            self.spatial_size,
            self.spatial_size,
        )
        spatial_features = self.spatial_fc(self.spatial_cnn(spatial))
        ego_features = self.ego_encoder(ego)
        return self.output(torch.cat([spatial_features, ego_features], dim=-1))


class RecurrentActorCritic(nn.Module):
    """One local recurrent actor-critic used by one agent.

    The actor and critic only receive that agent's local observation. They do not
    receive global state, other agents' actions, or oracle team IDs.
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        mlp_hidden_dim: int = 128,
        recurrent_hidden_dim: int = 128,
        encoder_type: str = "mlp",
        spatial_channels: int = 0,
        spatial_size: int = 0,
        ego_dim: int = 0,
    ):
        super().__init__()
        self.obs_dim = int(obs_dim)
        self.action_dim = int(action_dim)
        self.recurrent_hidden_dim = int(recurrent_hidden_dim)
        self.encoder_type = str(encoder_type)

        if self.encoder_type == "mlp":
            self.actor_encoder = mlp(obs_dim, mlp_hidden_dim, recurrent_hidden_dim)
            self.critic_encoder = mlp(obs_dim, mlp_hidden_dim, recurrent_hidden_dim)
        elif self.encoder_type == "cnn":
            self.actor_encoder = SemanticCNNEncoder(
                obs_dim=obs_dim,
                spatial_channels=spatial_channels,
                spatial_size=spatial_size,
                ego_dim=ego_dim,
                hidden_dim=mlp_hidden_dim,
                output_dim=recurrent_hidden_dim,
            )
            self.critic_encoder = SemanticCNNEncoder(
                obs_dim=obs_dim,
                spatial_channels=spatial_channels,
                spatial_size=spatial_size,
                ego_dim=ego_dim,
                hidden_dim=mlp_hidden_dim,
                output_dim=recurrent_hidden_dim,
            )
        else:
            raise ValueError("--obs-encoder must be 'mlp' or 'cnn'")

        self.actor_rnn = nn.GRUCell(recurrent_hidden_dim, recurrent_hidden_dim)
        self.actor_head = nn.Linear(recurrent_hidden_dim, action_dim)

        self.critic_rnn = nn.GRUCell(recurrent_hidden_dim, recurrent_hidden_dim)
        self.critic_head = nn.Linear(recurrent_hidden_dim, 1)

        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=math.sqrt(2))
                nn.init.constant_(module.bias, 0.0)
            elif isinstance(module, nn.Conv2d):
                nn.init.orthogonal_(module.weight, gain=math.sqrt(2))
                nn.init.constant_(module.bias, 0.0)
        nn.init.orthogonal_(self.actor_head.weight, gain=0.01)
        nn.init.orthogonal_(self.critic_head.weight, gain=1.0)

    def initial_hidden(self, batch_size: int, device: torch.device) -> TensorPair:
        shape = (batch_size, self.recurrent_hidden_dim)
        return torch.zeros(shape, device=device), torch.zeros(shape, device=device)

    def actor_parameters(self) -> Iterable[nn.Parameter]:
        yield from self.actor_encoder.parameters()
        yield from self.actor_rnn.parameters()
        yield from self.actor_head.parameters()

    def critic_parameters(self) -> Iterable[nn.Parameter]:
        yield from self.critic_encoder.parameters()
        yield from self.critic_rnn.parameters()
        yield from self.critic_head.parameters()

    def _actor_step(
        self,
        obs: torch.Tensor,
        actor_h: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        features = self.actor_encoder(obs)
        next_h = self.actor_rnn(features, actor_h)
        logits = self.actor_head(next_h)
        return logits, next_h

    def _critic_step(
        self,
        obs: torch.Tensor,
        critic_h: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        features = self.critic_encoder(obs)
        next_h = self.critic_rnn(features, critic_h)
        value = self.critic_head(next_h).squeeze(-1)
        return value, next_h

    def _masked_logits(
        self,
        logits: torch.Tensor,
        action_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if action_mask is None:
            return logits
        if action_mask.ndim == 1:
            action_mask = action_mask.unsqueeze(0)
        action_mask = action_mask.to(device=logits.device, dtype=torch.bool)
        if action_mask.shape != logits.shape:
            raise ValueError(
                f"action_mask shape {tuple(action_mask.shape)} does not match "
                f"logits shape {tuple(logits.shape)}"
            )
        has_valid = action_mask.any(dim=-1, keepdim=True)
        if not bool(has_valid.all()):
            fallback = torch.zeros_like(action_mask)
            fallback[..., 0] = True
            action_mask = torch.where(has_valid, action_mask, fallback)
        return logits.masked_fill(~action_mask, -1.0e9)

    @torch.no_grad()
    def act(
        self,
        obs: torch.Tensor,
        actor_h: torch.Tensor,
        critic_h: torch.Tensor,
        deterministic: bool = False,
        action_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if obs.ndim == 1:
            obs = obs.unsqueeze(0)
        if actor_h.ndim == 1:
            actor_h = actor_h.unsqueeze(0)
        if critic_h.ndim == 1:
            critic_h = critic_h.unsqueeze(0)

        logits, next_actor_h = self._actor_step(obs, actor_h)
        logits = self._masked_logits(logits, action_mask)
        value, next_critic_h = self._critic_step(obs, critic_h)
        dist = Categorical(logits=logits)
        if deterministic:
            action = torch.argmax(logits, dim=-1)
        else:
            action = dist.sample()
        log_prob = dist.log_prob(action)
        return action, log_prob, value, next_actor_h, next_critic_h

    @torch.no_grad()
    def max_action_probability(
        self,
        obs: torch.Tensor,
        actor_h: torch.Tensor,
        action_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return self.action_probabilities(obs, actor_h, action_mask).max(dim=-1).values

    @torch.no_grad()
    def action_probabilities(
        self,
        obs: torch.Tensor,
        actor_h: torch.Tensor,
        action_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if obs.ndim == 1:
            obs = obs.unsqueeze(0)
        if actor_h.ndim == 1:
            actor_h = actor_h.unsqueeze(0)

        logits, _ = self._actor_step(obs, actor_h)
        logits = self._masked_logits(logits, action_mask)
        return torch.softmax(logits, dim=-1)

    def evaluate_actions(
        self,
        obs_seq: torch.Tensor,
        action_seq: torch.Tensor,
        done_seq: torch.Tensor,
        actor_h0: torch.Tensor,
        critic_h0: torch.Tensor,
        action_mask_seq: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Evaluate a time-major recurrent minibatch.

        Args:
            obs_seq: [T, B, obs_dim]
            action_seq: [T, B]
            done_seq: [T, B], terminal flags after each step.
            actor_h0: [B, H], hidden state before obs_seq[0].
            critic_h0: [B, H], hidden state before obs_seq[0].
            action_mask_seq: optional [T, B, action_dim] valid-action mask.
        """
        actor_h = actor_h0
        critic_h = critic_h0
        log_probs: List[torch.Tensor] = []
        entropies: List[torch.Tensor] = []
        values: List[torch.Tensor] = []

        for step_idx, (obs_t, action_t, done_t) in enumerate(
            zip(obs_seq, action_seq, done_seq)
        ):
            logits, actor_h = self._actor_step(obs_t, actor_h)
            if action_mask_seq is not None:
                logits = self._masked_logits(logits, action_mask_seq[step_idx])
            value, critic_h = self._critic_step(obs_t, critic_h)
            dist = Categorical(logits=logits)
            log_probs.append(dist.log_prob(action_t))
            entropies.append(dist.entropy())
            values.append(value)

            keep_hidden = (1.0 - done_t.float()).unsqueeze(-1)
            actor_h = actor_h * keep_hidden
            critic_h = critic_h * keep_hidden

        return (
            torch.stack(log_probs, dim=0),
            torch.stack(entropies, dim=0),
            torch.stack(values, dim=0),
        )


@dataclass
class Rollout:
    obs: torch.Tensor
    actions: torch.Tensor
    action_masks: torch.Tensor
    old_log_probs: torch.Tensor
    rewards: torch.Tensor
    dones: torch.Tensor
    values: torch.Tensor
    actor_h: torch.Tensor
    critic_h: torch.Tensor
    last_values: torch.Tensor
    timeout_values: Optional[torch.Tensor] = None


@dataclass
class RunnerState:
    obs: Sequence[np.ndarray]
    actor_h: torch.Tensor
    critic_h: torch.Tensor
    episode_returns: np.ndarray
    episode_task_returns: np.ndarray
    episode_shaping_returns: np.ndarray
    episode_length: int = 0
    episode_deliveries: float = 0.0
    episode_returns_home: float = 0.0
    episode_pickups: float = 0.0
    episode_requested_pickups: float = 0.0
    episode_invalid_toggles: float = 0.0
    episode_wrong_returns: float = 0.0
    episode_failed_forwards: float = 0.0
    episode_blocking_agents: float = 0.0
    episode_deadlocked_agents: float = 0.0
    episode_requested_shelf_seen: bool = False
    episode_first_requested_shelf_seen_step: Optional[int] = None
    episode_first_pickup_step: Optional[int] = None
    episode_first_delivery_step: Optional[int] = None
    episode_first_return_step: Optional[int] = None


@dataclass
class TrainConfig:
    env_id: str
    env_kwargs_json: str
    sensor_range: int
    agent_count: int
    team_count: int
    auto_scale_request_queue: bool
    curriculum_stages: str
    curriculum_stage: int
    init_checkpoint: str
    transfer_components: str
    total_timesteps: int
    rollout_steps: int
    sequence_length: int
    minibatch_chunks: int
    ppo_epochs: int
    learning_rate: float
    gamma: float
    gae_lambda: float
    clip_coef: float
    value_clip_coef: float
    value_loss_coef: float
    entropy_coef: float
    action_mask: bool
    normalize_advantages: bool
    min_advantage_std: float
    skip_zero_reward_actor_update: bool
    skip_no_positive_reward_actor_update: bool
    reward_epsilon: float
    max_grad_norm: float
    mlp_hidden_dim: int
    recurrent_hidden_dim: int
    seed: int
    device: str
    observation_format: str
    obs_encoder: str
    graph_mode: str
    probe_interval: int
    policy_probe_batch_size: int
    probe_source: str
    objective_probe_fraction: float
    policy_probe_team_conditioning: str
    policy_similarity_temperature: float
    comm_graph_mode: str
    probe_episodes: int
    probe_horizon: int
    edge_threshold: float
    min_probe_count: int
    uncertainty_scale: float
    value_uncertainty_decay: float
    graph_join_threshold: float
    graph_leave_threshold: float
    graph_dwell_updates: int
    peer_transfer_mode: str
    pgct_peer_loss_coef: float
    pgct_probe_sequence_length: int
    pgct_distance_ema_beta: float
    pgct_distance_temperature: float
    pgct_distance_threshold: float
    pgct_gate_power: float
    pgct_alpha_epsilon: float
    pgct_warmup_updates: int
    pgct_min_policy_information: float
    critic_consensus_tau: float
    consensus_interval: int
    save_dir: str
    exp_name: str
    save_interval: int
    log_interval: int
    eval_episodes: int
    eval_horizon: int
    eval_interval: int
    render_eval: bool
    eval_render_episodes: int
    eval_render_mode: str
    eval_render_fps: int
    track: bool
    wandb_project: str
    wandb_entity: str
    wandb_group: str
    wandb_name: str
    wandb_mode: str
    wandb_tags: str
    wandb_log_eval_video: bool
    wandb_video_interval: int
    communication_enabled: bool = True
    pgct_gate_style: str = "threshold"
    fixed_canonical_probes: bool = False
    save_eval_video: bool = False


class TeamGraphEstimator:
    """Neighbor-sparse latent team graph with confidence-weighted edge scores.

    The intended update path compares policies on a small common probe set for
    communication-neighbor pairs only. `update_from_counterfactual` is kept as a
    legacy influence baseline behind --graph-mode influence.
    """

    def __init__(
        self,
        n_agents: int,
        edge_threshold: float,
        min_probe_count: int,
        uncertainty_scale: float,
        value_uncertainty_decay: float,
        join_threshold: Optional[float] = None,
        leave_threshold: Optional[float] = None,
        dwell_updates: int = 1,
        min_policy_information: float = 0.0,
        gate_style: str = "soft",
    ):
        self.n_agents = int(n_agents)
        self.edge_threshold = float(edge_threshold)
        self.min_probe_count = int(min_probe_count)
        self.uncertainty_scale = float(uncertainty_scale)
        self.value_uncertainty_decay = float(value_uncertainty_decay)
        self.join_threshold = (
            self.edge_threshold if join_threshold is None else float(join_threshold)
        )
        self.leave_threshold = (
            self.join_threshold if leave_threshold is None else float(leave_threshold)
        )
        if self.leave_threshold > self.join_threshold:
            raise ValueError("leave_threshold must be <= join_threshold")
        self.dwell_updates = max(int(dwell_updates), 1)
        self.gate_style = gate_style
        self.min_policy_information = max(float(min_policy_information), 0.0)

        shape = (self.n_agents, self.n_agents)
        self.count = np.zeros(shape, dtype=np.float64)
        self.mean = np.zeros(shape, dtype=np.float64)
        self.m2 = np.zeros(shape, dtype=np.float64)
        self.value_uncertainty = np.ones(self.n_agents, dtype=np.float64)
        self.distance_count = np.zeros(shape, dtype=np.float64)
        self.last_distance = np.full(shape, np.nan, dtype=np.float64)
        self.distance_ema = np.full(shape, np.nan, dtype=np.float64)
        self.last_neighbor_adjacency = np.zeros(shape, dtype=np.float32)
        self.stable_adjacency = np.zeros(shape, dtype=np.float32)
        self.join_streak = np.zeros(shape, dtype=np.int32)
        self.leave_streak = np.zeros(shape, dtype=np.int32)
        self.policy_information = np.zeros(self.n_agents, dtype=np.float64)

    def update_policy_information(self, values: np.ndarray) -> None:
        values = np.asarray(values, dtype=np.float64)
        if values.shape != (self.n_agents,):
            raise ValueError("policy information must have shape (n_agents,)")
        self.policy_information = np.maximum(values, 0.0)

    def update_pair_score(self, i: int, j: int, score: float) -> None:
        i, j = int(i), int(j)
        if i == j:
            return
        value = float(np.clip(score, 0.0, 1.0))
        a, b = min(i, j), max(i, j)
        self.count[a, b] += 1.0
        n = self.count[a, b]
        old_mean = self.mean[a, b]
        new_mean = old_mean + (value - old_mean) / n
        self.mean[a, b] = new_mean
        self.m2[a, b] += (value - old_mean) * (value - new_mean)

    def update_from_policy_similarity(
        self,
        similarity: np.ndarray,
        neighbor_adjacency: np.ndarray,
    ) -> int:
        similarity = np.asarray(similarity, dtype=np.float64)
        neighbor_adjacency = np.asarray(neighbor_adjacency, dtype=np.float32)
        if similarity.shape != (self.n_agents, self.n_agents):
            raise ValueError("similarity must have shape (n_agents, n_agents)")
        if neighbor_adjacency.shape != (self.n_agents, self.n_agents):
            raise ValueError("neighbor_adjacency must have shape (n_agents, n_agents)")

        self.last_neighbor_adjacency = (neighbor_adjacency > 0.0).astype(np.float32)
        updates = 0
        for i in range(self.n_agents):
            for j in range(i + 1, self.n_agents):
                if max(self.last_neighbor_adjacency[i, j], self.last_neighbor_adjacency[j, i]) <= 0.0:
                    continue
                self.update_pair_score(i, j, similarity[i, j])
                updates += 1
        return updates

    def update_from_policy_distance(
        self,
        distance: np.ndarray,
        neighbor_adjacency: np.ndarray,
        beta: float,
    ) -> int:
        distance = np.asarray(distance, dtype=np.float64)
        neighbor_adjacency = np.asarray(neighbor_adjacency, dtype=np.float32)
        if distance.shape != (self.n_agents, self.n_agents):
            raise ValueError("distance must have shape (n_agents, n_agents)")
        if neighbor_adjacency.shape != (self.n_agents, self.n_agents):
            raise ValueError("neighbor_adjacency must have shape (n_agents, n_agents)")

        beta = float(np.clip(beta, 1e-8, 1.0))
        self.last_neighbor_adjacency = (neighbor_adjacency > 0.0).astype(np.float32)
        updates = 0
        for i in range(self.n_agents):
            for j in range(i + 1, self.n_agents):
                observed = False
                for receiver, sender in ((i, j), (j, i)):
                    if self.last_neighbor_adjacency[receiver, sender] <= 0:
                        continue
                    observed = True
                    value = float(max(distance[receiver, sender], 0.0))
                    self.last_distance[receiver, sender] = value
                    old = self.distance_ema[receiver, sender]
                    smoothed = value if np.isnan(old) else (1.0 - beta) * old + beta * value
                    self.distance_ema[receiver, sender] = smoothed
                    self.distance_count[receiver, sender] += 1.0
                updates += int(observed)
        return updates

    def update_from_counterfactual(
        self,
        source_agent: int,
        delta_returns: np.ndarray,
    ) -> None:
        source_agent = int(source_agent)
        deltas = np.abs(np.asarray(delta_returns, dtype=np.float64))
        max_delta = float(np.max(deltas)) if deltas.size else 0.0
        for target_agent, value in enumerate(deltas):
            if target_agent == source_agent:
                continue
            normalized = float(value / max(max_delta, 1e-8))
            self.update_pair_score(source_agent, target_agent, normalized)

    def update_value_uncertainty(self, per_agent_abs_td_error: np.ndarray) -> None:
        errors = np.asarray(per_agent_abs_td_error, dtype=np.float64)
        if errors.shape != (self.n_agents,):
            raise ValueError("per_agent_abs_td_error must have shape (n_agents,)")
        decay = self.value_uncertainty_decay
        self.value_uncertainty = decay * self.value_uncertainty + (1.0 - decay) * errors

    def _pair_std(self, i: int, j: int) -> float:
        a, b = min(int(i), int(j)), max(int(i), int(j))
        n = self.count[a, b]
        if n <= 1.0:
            return 1.0
        return float(math.sqrt(max(self.m2[a, b] / (n - 1.0), 0.0)))

    def pair_score(self, i: int, j: int) -> Tuple[float, float, float]:
        a, b = min(int(i), int(j)), max(int(i), int(j))
        count = self.count[a, b]
        if count <= 0.0:
            return 0.0, 0.0, 1.0

        mean_score = self.mean[a, b]
        standard_error = self._pair_std(a, b) / math.sqrt(max(count, 1.0))
        critic_uncertainty = 0.5 * (
            self.value_uncertainty[a] + self.value_uncertainty[b]
        )
        critic_penalty = critic_uncertainty / (1.0 + critic_uncertainty)
        count_confidence = 1.0 - math.exp(-count / max(self.min_probe_count, 1))
        confidence = count_confidence * math.exp(
            -self.uncertainty_scale * (standard_error + critic_penalty)
        )
        weight = float(np.clip(mean_score * confidence, 0.0, 1.0))
        return weight, float(mean_score), float(standard_error)

    def weight_matrix(self) -> np.ndarray:
        weights = np.zeros((self.n_agents, self.n_agents), dtype=np.float32)
        for i in range(self.n_agents):
            for j in range(i + 1, self.n_agents):
                count = self.count[min(i, j), max(i, j)]
                if count < self.min_probe_count:
                    continue
                weight, _, _ = self.pair_score(i, j)
                weights[i, j] = weight
                weights[j, i] = weight
        return weights

    def update_stable_adjacency(self) -> np.ndarray:
        weights = self.weight_matrix()
        if (
            self.dwell_updates <= 1
            and abs(self.join_threshold - self.leave_threshold) < 1e-12
        ):
            self.stable_adjacency = (
                weights >= self.join_threshold
            ).astype(np.float32)
            np.fill_diagonal(self.stable_adjacency, 0.0)
            return self.stable_adjacency.copy()

        for i in range(self.n_agents):
            for j in range(i + 1, self.n_agents):
                weight = float(weights[i, j])
                active = self.stable_adjacency[i, j] > 0.0

                if active:
                    if weight <= self.leave_threshold:
                        self.leave_streak[i, j] += 1
                        self.leave_streak[j, i] = self.leave_streak[i, j]
                        self.join_streak[i, j] = 0
                        self.join_streak[j, i] = 0
                        if self.leave_streak[i, j] >= self.dwell_updates:
                            self.stable_adjacency[i, j] = 0.0
                            self.stable_adjacency[j, i] = 0.0
                            self.leave_streak[i, j] = 0
                            self.leave_streak[j, i] = 0
                    else:
                        self.leave_streak[i, j] = 0
                        self.leave_streak[j, i] = 0
                else:
                    if weight >= self.join_threshold:
                        self.join_streak[i, j] += 1
                        self.join_streak[j, i] = self.join_streak[i, j]
                        self.leave_streak[i, j] = 0
                        self.leave_streak[j, i] = 0
                        if self.join_streak[i, j] >= self.dwell_updates:
                            self.stable_adjacency[i, j] = 1.0
                            self.stable_adjacency[j, i] = 1.0
                            self.join_streak[i, j] = 0
                            self.join_streak[j, i] = 0
                    else:
                        self.join_streak[i, j] = 0
                        self.join_streak[j, i] = 0

        np.fill_diagonal(self.stable_adjacency, 0.0)
        return self.stable_adjacency.copy()

    def adjacency(self) -> np.ndarray:
        if (
            self.dwell_updates <= 1
            and abs(self.join_threshold - self.leave_threshold) < 1e-12
        ):
            return (self.weight_matrix() >= self.join_threshold).astype(np.float32)
        return self.stable_adjacency.copy()

    def consensus_weight_matrix(self) -> np.ndarray:
        return self.weight_matrix() * self.adjacency()

    def pgct_affinity_matrix(self, distance_temperature: float) -> np.ndarray:
        temperature = max(float(distance_temperature), 1e-8)
        affinity = np.zeros((self.n_agents, self.n_agents), dtype=np.float32)
        valid = (
            ~np.isnan(self.distance_ema)
            & (self.distance_count >= float(self.min_probe_count))
        )
        affinity[valid] = np.exp(-self.distance_ema[valid] / temperature).astype(
            np.float32
        )
        np.fill_diagonal(affinity, 0.0)
        return affinity

    def pgct_gate_matrix(
        self,
        warmup_complete: bool,
        distance_threshold: float,
        distance_temperature: float,
        gate_power: float,
    ) -> np.ndarray:
        gate = np.zeros((self.n_agents, self.n_agents), dtype=np.float32)
        if not warmup_complete or distance_threshold <= 0.0:
            return gate

        threshold = float(distance_threshold)
        temperature = max(float(distance_temperature), 1e-8)
        power = max(float(gate_power), 1e-8)
        tau_safe = math.exp(-threshold / temperature)
        denom = max(1.0 - tau_safe, 1e-8)
        valid = (
            ~np.isnan(self.distance_ema)
            & (self.distance_count >= float(self.min_probe_count))
            & (self.distance_ema < threshold)
            & (self.policy_information[:, None] >= self.min_policy_information)
            & (self.policy_information[None, :] >= self.min_policy_information)
        )
        if np.any(valid):
            affinity = np.exp(-self.distance_ema[valid] / temperature)
            raw_gate = np.clip((affinity - tau_safe) / denom, 0.0, 1.0)
            gate[valid] = (1.0 if self.gate_style == "threshold"
                           else np.power(raw_gate, power).astype(np.float32))
        np.fill_diagonal(gate, 0.0)
        return gate

    def pgct_allocation_matrix(
        self,
        warmup_complete: bool,
        distance_threshold: float,
        distance_temperature: float,
        gate_power: float,
        alpha_epsilon: float,
    ) -> np.ndarray:
        gate = self.pgct_gate_matrix(
            warmup_complete=warmup_complete,
            distance_threshold=distance_threshold,
            distance_temperature=distance_temperature,
            gate_power=gate_power,
        )
        denom = np.sum(gate, axis=1, keepdims=True) + max(float(alpha_epsilon), 1e-8)
        return (gate / denom).astype(np.float32)

    def clusters(self) -> List[List[int]]:
        return clusters_from_adjacency(self.adjacency())

    def state_dict(self) -> Dict[str, np.ndarray]:
        return {
            "count": self.count.copy(),
            "mean": self.mean.copy(),
            "m2": self.m2.copy(),
            "value_uncertainty": self.value_uncertainty.copy(),
            "distance_count": self.distance_count.copy(),
            "last_distance": self.last_distance.copy(),
            "distance_ema": self.distance_ema.copy(),
            "last_neighbor_adjacency": self.last_neighbor_adjacency.copy(),
            "stable_adjacency": self.stable_adjacency.copy(),
            "join_streak": self.join_streak.copy(),
            "leave_streak": self.leave_streak.copy(),
            "policy_information": self.policy_information.copy(),
        }


_BARE_JSON_KEY_RE = re.compile(
    r'(?P<prefix>[{,]\s*)(?P<key>[A-Za-z_][A-Za-z0-9_]*)\s*:'
)


def _quote_bare_json_keys(value: str) -> str:
    return _BARE_JSON_KEY_RE.sub(r'\g<prefix>"\g<key>":', value)


def parse_env_kwargs(env_kwargs_json: str) -> Dict:
    env_kwargs_json = env_kwargs_json.strip()
    if env_kwargs_json.startswith("@"):
        env_kwargs_path = Path(env_kwargs_json[1:])
        env_kwargs_json = env_kwargs_path.read_text(encoding="utf-8")

    try:
        value = json.loads(env_kwargs_json)
    except json.JSONDecodeError as exc:
        repaired_json = _quote_bare_json_keys(env_kwargs_json)
        if repaired_json != env_kwargs_json:
            try:
                value = json.loads(repaired_json)
            except json.JSONDecodeError:
                raise ValueError(f"Invalid --env-kwargs-json: {exc}") from exc
        else:
            raise ValueError(f"Invalid --env-kwargs-json: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError("--env-kwargs-json must decode to a JSON object")
    return value


def parse_curriculum_stages(value: str) -> List[int]:
    """Parse agent-count stages such as '4,8,12' or '4->8->12'."""
    if not value.strip():
        return []
    normalized = value.replace("->", ",").replace(";", ",")
    stages = [int(item.strip()) for item in normalized.split(",") if item.strip()]
    if not stages:
        return []
    if any(stage <= 0 for stage in stages):
        raise ValueError("--curriculum-stages must contain positive agent counts")
    return stages


def resolve_agent_count(
    explicit_agent_count: int,
    curriculum_stages: Sequence[int],
    curriculum_stage: int,
) -> Optional[int]:
    if curriculum_stage < 0:
        raise ValueError("--curriculum-stage must be non-negative")
    if explicit_agent_count < 0:
        raise ValueError("--agent-count must be non-negative")
    if curriculum_stage == 0:
        return int(explicit_agent_count) if explicit_agent_count > 0 else None
    if not curriculum_stages:
        raise ValueError("--curriculum-stage requires --curriculum-stages")
    if curriculum_stage > len(curriculum_stages):
        raise ValueError(
            f"--curriculum-stage {curriculum_stage} is outside "
            f"--curriculum-stages length {len(curriculum_stages)}"
        )
    stage_agent_count = int(curriculum_stages[curriculum_stage - 1])
    if explicit_agent_count > 0 and explicit_agent_count != stage_agent_count:
        raise ValueError(
            "--agent-count conflicts with the selected --curriculum-stage "
            f"({explicit_agent_count} != {stage_agent_count})"
        )
    return stage_agent_count


def apply_agent_overrides(
    env_id: str,
    env_kwargs: Dict[str, Any],
    agent_count: Optional[int],
    team_count: int,
    auto_scale_request_queue: bool = True,
) -> Dict[str, Any]:
    updated = dict(env_kwargs)
    if team_count < 0:
        raise ValueError("--team-count must be non-negative")
    if team_count > 0:
        updated["n_teams"] = int(team_count)
    if agent_count is None:
        return updated
    if agent_count <= 0:
        raise ValueError("agent_count must be positive")

    if "team_assignments" in updated and len(updated["team_assignments"]) != agent_count:
        raise ValueError(
            "--env-kwargs-json team_assignments length must match the selected "
            "agent count"
        )

    updated["n_agents"] = int(agent_count)
    effective_team_count = infer_team_count(env_id, updated)
    if env_id.startswith("rware-multiteam-") and effective_team_count is not None:
        _apply_multiteam_request_queue_override(
            updated,
            agent_count=int(agent_count),
            team_count=int(effective_team_count),
            auto_scale=auto_scale_request_queue,
        )
    return updated


def infer_team_count(env_id: str, env_kwargs: Dict[str, Any]) -> Optional[int]:
    if "n_teams" in env_kwargs:
        return int(env_kwargs["n_teams"])
    for token in env_id.split("-"):
        if token.endswith("teams") and token[: -len("teams")].isdigit():
            return int(token[: -len("teams")])
    return None


def _balanced_team_agent_counts(agent_count: int, team_count: int) -> List[int]:
    counts = [0 for _ in range(team_count)]
    for agent_idx in range(agent_count):
        counts[agent_idx % team_count] += 1
    return counts


def _team_agent_counts_from_kwargs(
    env_kwargs: Dict[str, Any],
    agent_count: int,
    team_count: int,
) -> List[int]:
    assignments = env_kwargs.get("team_assignments")
    if assignments is None:
        return _balanced_team_agent_counts(agent_count, team_count)
    counts = [0 for _ in range(team_count)]
    for team_id in assignments:
        team_index = int(team_id)
        if team_index < 0 or team_index >= team_count:
            raise ValueError("team_assignments contains an out-of-range team id")
        counts[team_index] += 1
    return counts


def _request_queue_sizes_per_team(value: Any, team_count: int) -> List[int]:
    if isinstance(value, int):
        return [int(value) for _ in range(team_count)]
    values = [int(item) for item in value]
    if len(values) != team_count:
        raise ValueError("request_queue_size_per_team must have one value per team")
    return values


def _compact_team_request_sizes(values: Sequence[int]) -> Any:
    normalized = [int(value) for value in values]
    if normalized and all(value == normalized[0] for value in normalized):
        return normalized[0]
    return normalized


def _apply_multiteam_request_queue_override(
    env_kwargs: Dict[str, Any],
    agent_count: int,
    team_count: int,
    auto_scale: bool,
) -> None:
    if team_count <= 0:
        raise ValueError("team_count must be positive for multiteam RWARE")

    target_per_team = [
        max(1, count)
        for count in _team_agent_counts_from_kwargs(env_kwargs, agent_count, team_count)
    ]
    target_total = int(sum(target_per_team))
    has_total = "request_queue_size" in env_kwargs
    has_per_team = "request_queue_size_per_team" in env_kwargs

    if not has_total and not has_per_team:
        env_kwargs["request_queue_size_per_team"] = _compact_team_request_sizes(
            target_per_team
        )
        env_kwargs["request_queue_size"] = target_total
        return

    if not auto_scale:
        return

    if has_per_team:
        current_per_team = _request_queue_sizes_per_team(
            env_kwargs["request_queue_size_per_team"],
            team_count,
        )
        scaled_per_team = [
            max(current, target)
            for current, target in zip(current_per_team, target_per_team)
        ]
        env_kwargs["request_queue_size_per_team"] = _compact_team_request_sizes(
            scaled_per_team
        )
        env_kwargs["request_queue_size"] = max(
            int(env_kwargs.get("request_queue_size", 0)),
            int(sum(scaled_per_team)),
        )
        return

    current_total = int(env_kwargs["request_queue_size"])
    if current_total < target_total:
        env_kwargs["request_queue_size_per_team"] = _compact_team_request_sizes(
            target_per_team
        )
        env_kwargs["request_queue_size"] = target_total


TRANSFER_COMPONENT_PREFIXES: Dict[str, Tuple[str, ...]] = {
    "actor": ("actor_encoder.", "actor_rnn.", "actor_head."),
    "actor_body": ("actor_encoder.", "actor_rnn."),
    "actor_encoder": ("actor_encoder.",),
    "actor_rnn": ("actor_rnn.",),
    "actor_head": ("actor_head.",),
    "critic": ("critic_encoder.", "critic_rnn.", "critic_head."),
    "critic_body": ("critic_encoder.", "critic_rnn."),
    "critic_encoder": ("critic_encoder.",),
    "critic_rnn": ("critic_rnn.",),
    "critic_head": ("critic_head.",),
}


def parse_transfer_components(value: str) -> Set[str]:
    raw_components = [item.strip().lower() for item in value.split(",") if item.strip()]
    if not raw_components:
        return {"actor"}
    if "all" in raw_components:
        return set(TRANSFER_COMPONENT_PREFIXES)
    if "none" in raw_components:
        if len(raw_components) > 1:
            raise ValueError("'none' cannot be combined with transfer components")
        return set()
    components = set(raw_components)
    invalid = components - set(TRANSFER_COMPONENT_PREFIXES)
    if invalid:
        valid = ", ".join(sorted(TRANSFER_COMPONENT_PREFIXES))
        raise ValueError(
            f"Unknown --transfer-components {sorted(invalid)}; valid: {valid}"
        )
    return components


def _component_prefixes(components: Set[str]) -> Tuple[str, ...]:
    prefixes: List[str] = []
    for component in sorted(components):
        prefixes.extend(TRANSFER_COMPONENT_PREFIXES[component])
    return tuple(prefixes)


def _load_checkpoint(path: str, device: torch.device) -> Dict[str, Any]:
    checkpoint_path = Path(path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"--init-checkpoint does not exist: {checkpoint_path}")
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location=device)
    if not isinstance(checkpoint, dict) or "agents" not in checkpoint:
        raise ValueError("--init-checkpoint must be a training checkpoint with agents")
    return checkpoint


def transfer_checkpoint_to_agents(
    checkpoint_path: str,
    agents: Sequence[RecurrentActorCritic],
    components: Set[str],
    device: torch.device,
) -> Dict[str, Any]:
    """Initialize target agents from a previous stage checkpoint.

    Matching agent indices are copied first. Newly added agents copy source
    agents round-robin, while graph statistics and optimizer state are left fresh.
    """
    if not components:
        return {
            "checkpoint": checkpoint_path,
            "source_agents": 0,
            "target_agents": len(agents),
            "components": [],
            "copied_tensors": 0,
            "skipped_tensors": 0,
            "mapping": [],
        }

    checkpoint = _load_checkpoint(checkpoint_path, device)
    source_agent_states = checkpoint["agents"]
    if not isinstance(source_agent_states, (list, tuple)) or not source_agent_states:
        raise ValueError("--init-checkpoint does not contain any agent states")

    prefixes = _component_prefixes(components)
    copied_tensors = 0
    skipped_tensors = 0
    mapping: List[int] = []

    for target_idx, target_agent in enumerate(agents):
        source_idx = (
            target_idx
            if target_idx < len(source_agent_states)
            else target_idx % len(source_agent_states)
        )
        mapping.append(int(source_idx))
        source_state = source_agent_states[source_idx]
        target_state = target_agent.state_dict()
        load_state: Dict[str, torch.Tensor] = {}

        for key, value in source_state.items():
            if not key.startswith(prefixes):
                continue
            if key not in target_state or target_state[key].shape != value.shape:
                skipped_tensors += 1
                continue
            load_state[key] = value

        missing, unexpected = target_agent.load_state_dict(load_state, strict=False)
        copied_tensors += len(load_state)
        skipped_tensors += len(unexpected)
        del missing

    return {
        "checkpoint": str(Path(checkpoint_path)),
        "source_agents": len(source_agent_states),
        "target_agents": len(agents),
        "components": sorted(components),
        "copied_tensors": int(copied_tensors),
        "skipped_tensors": int(skipped_tensors),
        "mapping": mapping,
    }


def init_wandb(cfg: TrainConfig, save_dir: Path):
    if not cfg.track:
        return None
    try:
        import wandb
    except ImportError as exc:
        raise ImportError(
            "wandb is required when --track is set. Install with "
            '`pip install -e ".[rl]"` or `pip install wandb`.'
        ) from exc

    tags = [tag.strip() for tag in cfg.wandb_tags.split(",") if tag.strip()]
    return wandb.init(
        project=cfg.wandb_project,
        entity=cfg.wandb_entity or None,
        group=cfg.wandb_group or None,
        name=cfg.wandb_name or (cfg.exp_name or None),
        mode=cfg.wandb_mode,
        tags=tags or None,
        config=asdict(cfg),
        dir=str(save_dir),
        save_code=True,
    )


def make_render_env_kwargs(env_kwargs: Dict, render_mode: str) -> Dict:
    render_kwargs = dict(env_kwargs)
    render_kwargs["render_mode"] = render_mode
    return render_kwargs


def make_env(
    env_id: str,
    env_kwargs: Dict,
    observation_format: str = "flat",
) -> gym.Env:
    env = gym.make(env_id, **env_kwargs)
    if observation_format == "semantic":
        return RwareSemanticObservationWrapper(env)
    if observation_format != "flat":
        raise ValueError("--observation-format must be 'flat' or 'semantic'")
    return env


def tuple_spaces(space: gym.Space) -> List[gym.Space]:
    if not isinstance(space, gym.spaces.Tuple):
        raise ValueError("Expected a multi-agent Tuple space")
    return list(space.spaces)


def get_flat_obs_dims(env: gym.Env) -> List[int]:
    return [gym.spaces.flatdim(space) for space in tuple_spaces(env.observation_space)]


def get_discrete_action_dims(env: gym.Env) -> List[int]:
    dims = []
    for space in tuple_spaces(env.action_space):
        if not isinstance(space, gym.spaces.Discrete):
            raise ValueError(
                "This implementation expects msg_bits=0 and Discrete actions. "
                f"Got {space!r}."
            )
        dims.append(int(space.n))
    return dims


def flatten_multi_agent_obs(env: gym.Env, obs: Sequence) -> np.ndarray:
    flat_obs = [
        gym.spaces.flatten(space, agent_obs).astype(np.float32)
        for space, agent_obs in zip(tuple_spaces(env.observation_space), obs)
    ]
    return np.stack(flat_obs, axis=0)


def _rware_forward_target(agent, grid_size: Tuple[int, int]) -> Tuple[int, int]:
    if agent.dir == Direction.UP:
        return int(agent.x), max(0, int(agent.y) - 1)
    if agent.dir == Direction.DOWN:
        return int(agent.x), min(int(grid_size[0]) - 1, int(agent.y) + 1)
    if agent.dir == Direction.LEFT:
        return max(0, int(agent.x) - 1), int(agent.y)
    if agent.dir == Direction.RIGHT:
        return min(int(grid_size[1]) - 1, int(agent.x) + 1), int(agent.y)
    return int(agent.x), int(agent.y)


def local_rware_action_masks(
    env: gym.Env,
    n_agents: int,
    action_dim: int,
    enabled: bool,
) -> np.ndarray:
    masks = np.ones((n_agents, action_dim), dtype=np.float32)
    if not enabled or action_dim < len(Action):
        return masks

    unwrapped = env.unwrapped
    if not all(hasattr(unwrapped, name) for name in ["agents", "grid", "grid_size"]):
        return masks

    agents = list(getattr(unwrapped, "agents", []))
    for agent_id, agent in enumerate(agents[:n_agents]):
        forward_idx = int(Action.FORWARD.value)
        toggle_idx = int(Action.TOGGLE_LOAD.value)

        target_x, target_y = _rware_forward_target(agent, unwrapped.grid_size)
        if (target_x, target_y) == (int(agent.x), int(agent.y)):
            masks[agent_id, forward_idx] = 0.0
        elif (
            agent.carrying_shelf is not None
            and int(unwrapped.grid[_LAYER_SHELFS, target_y, target_x]) > 0
        ):
            occupant_id = int(unwrapped.grid[_LAYER_AGENTS, target_y, target_x])
            # A loaded neighbor may move simultaneously. The environment's
            # collision resolver decides this; do not prohibit legal convoys.
            if not (occupant_id > 0 and agents[occupant_id - 1].carrying_shelf is not None):
                masks[agent_id, forward_idx] = 0.0

        if agent.carrying_shelf is None:
            shelf_id = int(unwrapped.grid[_LAYER_SHELFS, int(agent.y), int(agent.x)])
            if shelf_id <= 0:
                masks[agent_id, toggle_idx] = 0.0
        elif hasattr(unwrapped, "_is_highway") and unwrapped._is_highway(
            int(agent.x),
            int(agent.y),
        ):
            masks[agent_id, toggle_idx] = 0.0

        if not np.any(masks[agent_id] > 0.0):
            masks[agent_id, int(Action.NOOP.value)] = 1.0

    return masks


def rware_requested_shelf_visibility(env: gym.Env, n_agents: int) -> np.ndarray:
    unwrapped = env.unwrapped
    if not all(hasattr(unwrapped, name) for name in ["agents", "sensor_range"]):
        return np.zeros((n_agents,), dtype=bool)

    visibility = np.zeros((n_agents,), dtype=bool)
    sensor_range = int(getattr(unwrapped, "sensor_range", 0))
    agents = list(getattr(unwrapped, "agents", []))
    for agent_id, agent in enumerate(agents[:n_agents]):
        if hasattr(unwrapped, "_requested_shelves_for_agent"):
            shelves = unwrapped._requested_shelves_for_agent(agent)
        else:
            shelves = list(getattr(unwrapped, "request_queue", []))
        visibility[agent_id] = any(
            max(abs(int(agent.x) - int(shelf.x)), abs(int(agent.y) - int(shelf.y)))
            <= sensor_range
            for shelf in shelves
        )
    return visibility


def rware_requested_pickup_opportunities(env: gym.Env, n_agents: int) -> np.ndarray:
    unwrapped = env.unwrapped
    if not all(hasattr(unwrapped, name) for name in ["agents", "grid", "shelfs"]):
        return np.zeros((n_agents,), dtype=bool)

    opportunities = np.zeros((n_agents,), dtype=bool)
    agents = list(getattr(unwrapped, "agents", []))
    shelves = list(getattr(unwrapped, "shelfs", []))
    for agent_id, agent in enumerate(agents[:n_agents]):
        if getattr(agent, "carrying_shelf", None) is not None:
            continue
        shelf_id = int(unwrapped.grid[_LAYER_SHELFS, int(agent.y), int(agent.x)])
        if shelf_id <= 0 or shelf_id > len(shelves):
            continue
        shelf = shelves[shelf_id - 1]
        if hasattr(unwrapped, "_is_agent_requested_shelf"):
            opportunities[agent_id] = bool(
                unwrapped._is_agent_requested_shelf(agent, shelf)
            )
        else:
            opportunities[agent_id] = shelf in list(
                getattr(unwrapped, "request_queue", [])
            )
    return opportunities


def rware_return_diagnostics(
    env: gym.Env,
    n_agents: int,
) -> Tuple[np.ndarray, np.ndarray, List[float]]:
    unwrapped = env.unwrapped
    if not hasattr(unwrapped, "agents"):
        return (
            np.zeros((n_agents,), dtype=bool),
            np.zeros((n_agents,), dtype=bool),
            [],
        )

    awaiting_return = set(getattr(unwrapped, "_shelves_awaiting_return", set()))
    home_positions = getattr(unwrapped, "_shelf_home_positions", {})
    return_opportunities = np.zeros((n_agents,), dtype=bool)
    return_phase = np.zeros((n_agents,), dtype=bool)
    return_distances: List[float] = []
    agents = list(getattr(unwrapped, "agents", []))
    for agent_id, agent in enumerate(agents[:n_agents]):
        shelf = getattr(agent, "carrying_shelf", None)
        if shelf is None or int(shelf.id) not in awaiting_return:
            continue
        return_phase[agent_id] = True
        home = home_positions.get(int(shelf.id))
        if home is None:
            continue
        distance = abs(int(agent.x) - int(home[0])) + abs(int(agent.y) - int(home[1]))
        return_distances.append(float(distance))
        on_highway = (
            bool(unwrapped._is_highway(int(agent.x), int(agent.y)))
            if hasattr(unwrapped, "_is_highway")
            else False
        )
        return_opportunities[agent_id] = (
            distance == 0
            and not on_highway
            and bool(getattr(agent, "has_delivered", False))
        )
    return return_opportunities, return_phase, return_distances


def rware_agent_positions(env: gym.Env, n_agents: int) -> List[Tuple[int, int]]:
    agents = list(getattr(env.unwrapped, "agents", []))
    return [(int(agent.x), int(agent.y)) for agent in agents[:n_agents]]


def add_agent_event_counts(
    events: object,
    counts: np.ndarray,
    key: str = "agent_id",
) -> None:
    if not isinstance(events, list):
        return
    for event in events:
        if not isinstance(event, dict):
            continue
        try:
            agent_idx = int(event.get(key, 0)) - 1
        except (TypeError, ValueError):
            continue
        if 0 <= agent_idx < len(counts):
            counts[agent_idx] += 1.0


def per_agent_metrics(
    prefix: str,
    n_agents: int,
    agent_step_counts: np.ndarray,
    position_change_counts: np.ndarray,
    pickup_counts: np.ndarray,
    delivery_counts: np.ndarray,
    return_counts: np.ndarray,
    failed_forward_counts: np.ndarray,
    blocking_counts: np.ndarray,
    deadlock_counts: np.ndarray,
) -> Dict[str, float]:
    metrics: Dict[str, float] = {}
    for agent_idx in range(n_agents):
        denominator = max(float(agent_step_counts[agent_idx]), 1.0)
        key_prefix = f"{prefix}agent_{agent_idx + 1}"
        metrics[f"{key_prefix}_position_change_rate"] = float(
            position_change_counts[agent_idx] / denominator
        )
        metrics[f"{key_prefix}_pickup_count"] = float(pickup_counts[agent_idx])
        metrics[f"{key_prefix}_delivery_count"] = float(delivery_counts[agent_idx])
        metrics[f"{key_prefix}_return_count"] = float(return_counts[agent_idx])
        metrics[f"{key_prefix}_failed_forward_rate"] = float(
            failed_forward_counts[agent_idx] / denominator
        )
        metrics[f"{key_prefix}_blocking_agent_rate"] = float(
            blocking_counts[agent_idx] / denominator
        )
        metrics[f"{key_prefix}_deadlock_agent_rate"] = float(
            deadlock_counts[agent_idx] / denominator
        )
    return metrics


def format_agent_rates(metrics: Dict[str, float], key_suffix: str, n_agents: int) -> str:
    values = [
        metrics.get(f"agent_{agent_idx + 1}_{key_suffix}", float("nan"))
        for agent_idx in range(n_agents)
    ]
    return "[" + ",".join(f"{value:.2f}" for value in values) + "]"


def format_eval_agent_rates(
    metrics: Dict[str, float],
    key_suffix: str,
    n_agents: int,
) -> str:
    values = [
        metrics.get(f"eval_agent_{agent_idx + 1}_{key_suffix}", float("nan"))
        for agent_idx in range(n_agents)
    ]
    return "[" + ",".join(f"{value:.2f}" for value in values) + "]"


def finite_mean(values: Sequence[float]) -> float:
    finite_values = [float(value) for value in values if math.isfinite(float(value))]
    return float(np.mean(finite_values)) if finite_values else float("nan")


def optional_step_delta(
    later_step: Optional[int],
    earlier_step: Optional[int],
) -> float:
    if later_step is None or earlier_step is None:
        return float("nan")
    return float(max(0, int(later_step) - int(earlier_step)))


def clusters_from_adjacency(adjacency: np.ndarray) -> List[List[int]]:
    adj = np.asarray(adjacency)
    # Display weak components only; this is not directed temporal reachability.
    adj = np.maximum(adj, adj.T)
    n_agents = int(adj.shape[0])
    seen = np.zeros(n_agents, dtype=bool)
    clusters: List[List[int]] = []
    for start in range(n_agents):
        if seen[start]:
            continue
        stack = [start]
        seen[start] = True
        cluster = []
        while stack:
            node = stack.pop()
            cluster.append(node)
            neighbors = np.flatnonzero(adj[node] > 0.0).tolist()
            for nxt in neighbors:
                if not seen[nxt]:
                    seen[nxt] = True
                    stack.append(int(nxt))
        clusters.append(sorted(cluster))
    return clusters


def set_global_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def set_env_training_edges(env: gym.Env, adjacency: np.ndarray) -> None:
    unwrapped = env.unwrapped
    if hasattr(unwrapped, "set_training_comm_weights"):
        unwrapped.set_training_comm_weights(adjacency)
        return
    if not hasattr(unwrapped, "set_training_comm_edges"):
        return
    edges = []
    for i in range(adjacency.shape[0]):
        for j in range(i + 1, adjacency.shape[1]):
            if adjacency[i, j] > 0.0:
                edges.append((i, j))
    unwrapped.set_training_comm_edges(edges)


def oracle_clusters(env: gym.Env) -> Optional[List[List[int]]]:
    unwrapped = env.unwrapped
    if not hasattr(unwrapped, "get_team_members"):
        return None
    return [list(map(int, members)) for members in unwrapped.get_team_members()]


def communication_adjacency(
    env: gym.Env,
    n_agents: int,
    mode: str = "physical",
) -> np.ndarray:
    if mode == "none" or not getattr(env.unwrapped, "communication_enabled", True):
        return np.zeros((n_agents, n_agents), dtype=np.float32)
    if mode == "complete":
        adj = np.ones((n_agents, n_agents), dtype=np.float32)
        np.fill_diagonal(adj, 0.0)
        return adj
    if mode not in {"physical"}:
        raise ValueError("unknown communication graph mode")

    unwrapped = env.unwrapped
    if hasattr(unwrapped, "get_neighbor_adjacency"):
        adj = np.asarray(unwrapped.get_neighbor_adjacency(), dtype=np.float32)
        if adj.shape == (n_agents, n_agents):
            np.fill_diagonal(adj, 0.0)
            return (adj > 0.0).astype(np.float32)

    if hasattr(unwrapped, "agents"):
        agents = list(unwrapped.agents)
        comm_range = int(
            getattr(unwrapped, "communication_range", getattr(unwrapped, "sensor_range", 1))
        )
        adj = np.zeros((n_agents, n_agents), dtype=np.float32)
        for i in range(n_agents):
            for j in range(i + 1, n_agents):
                dx = abs(int(agents[i].x) - int(agents[j].x))
                dy = abs(int(agents[i].y) - int(agents[j].y))
                if max(dx, dy) <= comm_range:
                    adj[i, j] = 1.0
                    adj[j, i] = 1.0
        return adj

    return np.zeros((n_agents, n_agents), dtype=np.float32)


def communication_weights(cfg, env, estimator, iteration):
    """Eligibility before current geometry masking; frozen during each rollout."""
    n = env.unwrapped.n_agents
    zero = np.zeros((n, n), dtype=np.float32)
    if not cfg.communication_enabled or cfg.graph_mode == "none" or cfg.peer_transfer_mode == "none":
        return zero
    if cfg.peer_transfer_mode == "pgct" and cfg.pgct_peer_loss_coef <= 0:
        return zero
    if cfg.peer_transfer_mode == "consensus" and (cfg.critic_consensus_tau <= 0 or cfg.consensus_interval <= 0):
        return zero
    if cfg.graph_mode in {"oracle", "wrong", "unrestricted"}:
        return baseline_weight_matrix(env, n, cfg.graph_mode)
    if cfg.peer_transfer_mode == "pgct":
        return estimator.pgct_gate_matrix(
            iteration >= cfg.pgct_warmup_updates, cfg.pgct_distance_threshold,
            cfg.pgct_distance_temperature, cfg.pgct_gate_power)
    return estimator.consensus_weight_matrix()


def enforce_communication_config(cfg):
    """A single off flag wins over every transfer/graph option."""
    if not cfg.communication_enabled or cfg.graph_mode == "none" or cfg.comm_graph_mode == "none":
        cfg.communication_enabled = False
        cfg.graph_mode = "none"
        cfg.comm_graph_mode = "none"
        cfg.peer_transfer_mode = "none"
        cfg.probe_interval = 0
        cfg.critic_consensus_tau = 0.0
        cfg.consensus_interval = 0
        cfg.pgct_peer_loss_coef = 0.0


def baseline_weight_matrix(env, n_agents, mode):
    complete = np.ones((n_agents, n_agents), dtype=np.float32) - np.eye(n_agents, dtype=np.float32)
    if mode == "unrestricted":
        return complete
    same = oracle_weight_matrix(env, n_agents)
    if mode == "wrong":
        return complete - same
    return same


def oracle_weight_matrix(env: gym.Env, n_agents: int) -> np.ndarray:
    clusters = oracle_clusters(env)
    weights = np.zeros((n_agents, n_agents), dtype=np.float32)
    if not clusters:
        return weights
    for cluster in clusters:
        for i in cluster:
            for j in cluster:
                if i != j:
                    weights[int(i), int(j)] = 1.0
    return weights


def sample_common_probe_obs(
    rollout: Rollout,
    batch_size: int,
    rng: np.random.Generator,
) -> torch.Tensor:
    pool = rollout.obs.reshape(-1, rollout.obs.shape[-1])
    if pool.shape[0] == 0:
        raise ValueError("Cannot sample probes from an empty rollout")
    count = min(max(int(batch_size), 1), int(pool.shape[0]))
    replace = count > pool.shape[0]
    indices = rng.choice(pool.shape[0], size=count, replace=replace)
    return pool[torch.as_tensor(indices, dtype=torch.long)]


def _sensor_location_index(sensor_range: int, dy: int, dx: int) -> Optional[int]:
    if abs(dy) > sensor_range or abs(dx) > sensor_range:
        return None
    width = 1 + 2 * sensor_range
    return int((dy + sensor_range) * width + (dx + sensor_range))


def _sample_cardinal_offsets(
    sensor_range: int,
    rng: np.random.Generator,
) -> List[Tuple[int, int]]:
    if sensor_range <= 0:
        return [(0, 0)]
    offsets = [(0, 1), (1, 0), (0, -1), (-1, 0)]
    rng.shuffle(offsets)
    return offsets


def _build_mtgrid_objective_probe_bank(
    env: gym.Env,
    obs_dim: int,
    batch_size: int,
    rng: np.random.Generator,
) -> Optional[torch.Tensor]:
    unwrapped = env.unwrapped
    required = ["n_teams", "msg_bits", "sensor_range", "grid_size"]
    if not all(hasattr(unwrapped, name) for name in required):
        return None

    n_teams = int(unwrapped.n_teams)
    msg_bits = int(unwrapped.msg_bits)
    sensor_range = int(unwrapped.sensor_range)
    cell_features = 2 + msg_bits + n_teams
    expected_dim = 2 + (1 + 2 * sensor_range) ** 2 * cell_features
    if expected_dim != obs_dim:
        return None

    target_offset = 2 + msg_bits
    probes: List[np.ndarray] = []
    categories = ["goal", "conflict", "neutral"]
    height, width = tuple(map(int, unwrapped.grid_size))

    for probe_id in range(max(int(batch_size), 1)):
        obs = np.zeros(obs_dim, dtype=np.float32)
        if getattr(unwrapped, "normalised_coordinates", True):
            obs[0] = 0.5
            obs[1] = 0.5
        else:
            obs[0] = max(0, height // 2)
            obs[1] = max(0, width // 2)

        category = categories[probe_id % len(categories)]
        offsets = _sample_cardinal_offsets(sensor_range, rng)

        if category == "goal":
            dy, dx = offsets[0]
            team_id = probe_id % n_teams
            loc = _sensor_location_index(sensor_range, dy, dx)
            if loc is not None:
                base = 2 + loc * cell_features
                obs[base + target_offset + team_id] = 1.0

        elif category == "conflict":
            for team_id, (dy, dx) in enumerate(offsets[: min(n_teams, len(offsets))]):
                loc = _sensor_location_index(sensor_range, dy, dx)
                if loc is None:
                    continue
                base = 2 + loc * cell_features
                obs[base + target_offset + (team_id % n_teams)] = 1.0

        else:
            # Neutral probes expose generic navigation/avoidance pressure without
            # a team-specific target. They help separate task preference from
            # shared obstacle-avoidance behavior.
            dy, dx = offsets[0]
            loc = _sensor_location_index(sensor_range, dy, dx)
            if loc is not None:
                base = 2 + loc * cell_features
                obs[base] = 1.0

        probes.append(obs)

    return torch.as_tensor(np.stack(probes, axis=0), dtype=torch.float32)


def _build_rware_objective_probe_bank(
    env: gym.Env,
    obs_dim: int,
    batch_size: int,
    rng: np.random.Generator,
) -> Optional[torch.Tensor]:
    return _build_rware_probe_map_bank(env, obs_dim, batch_size, rng)


def _agent_team_ids_from_env(env: gym.Env, n_agents: int) -> Optional[np.ndarray]:
    unwrapped = env.unwrapped
    if hasattr(unwrapped, "agent_team_ids"):
        team_ids = np.asarray(getattr(unwrapped, "agent_team_ids"), dtype=np.int32)
        if team_ids.shape[0] >= n_agents:
            return team_ids[:n_agents].copy()

    if hasattr(unwrapped, "get_team_members"):
        team_ids = np.full((n_agents,), -1, dtype=np.int32)
        for team_id, members in enumerate(unwrapped.get_team_members()):
            for agent_id in members:
                if 0 <= int(agent_id) < n_agents:
                    team_ids[int(agent_id)] = int(team_id)
        if np.all(team_ids >= 0):
            return team_ids

    return None


def _repeat_shared_probes_for_agents(
    probes: torch.Tensor,
    n_agents: int,
) -> torch.Tensor:
    if probes.ndim != 2:
        raise ValueError("shared probe observations must have shape (M, obs_dim)")
    return probes.unsqueeze(0).repeat(int(n_agents), 1, 1)


def _build_agent_conditioned_objective_probe_bank(
    env: gym.Env,
    obs_dim: int,
    batch_size: int,
    rng: np.random.Generator,
    agent_team_ids: np.ndarray,
) -> Optional[torch.Tensor]:
    probes = _build_rware_agent_probe_map_bank(
        env,
        obs_dim,
        batch_size,
        rng,
        agent_team_ids,
    )
    if probes is not None:
        return probes

    shared = build_objective_probe_bank(env, obs_dim, batch_size, rng)
    if shared is None:
        return None
    return _repeat_shared_probes_for_agents(shared, len(agent_team_ids))


def _objective_visibility_scores(env: gym.Env, obs_pool: torch.Tensor) -> np.ndarray:
    unwrapped = env.unwrapped
    pool = obs_pool.cpu().numpy()
    semantic_spec = get_semantic_observation_spec(env)
    if pool.shape[1] == semantic_spec.obs_dim:
        spatial = pool[:, semantic_spec.ego_dim :].reshape(
            pool.shape[0],
            semantic_spec.spatial_channels,
            semantic_spec.spatial_size,
            semantic_spec.spatial_size,
        )
        requested_shelf_channel = 4
        return spatial[:, requested_shelf_channel].sum(axis=(1, 2))

    if all(
        hasattr(unwrapped, name)
        for name in ["n_teams", "msg_bits", "sensor_range", "grid_size"]
    ):
        n_teams = int(unwrapped.n_teams)
        msg_bits = int(unwrapped.msg_bits)
        sensor_range = int(unwrapped.sensor_range)
        cell_features = 2 + msg_bits + n_teams
        expected_dim = 2 + (1 + 2 * sensor_range) ** 2 * cell_features
        if pool.shape[1] == expected_dim:
            target_offset = 2 + msg_bits
            scores = np.zeros(pool.shape[0], dtype=np.float32)
            for loc in range((1 + 2 * sensor_range) ** 2):
                base = 2 + loc * cell_features
                scores += pool[:, base + target_offset : base + target_offset + n_teams].sum(axis=1)
            return scores

    if all(
        hasattr(unwrapped, name)
        for name in [
            "_obs_bits_for_self",
            "_obs_bits_per_agent",
            "_obs_bits_per_shelf",
            "sensor_range",
        ]
    ):
        self_bits = int(unwrapped._obs_bits_for_self)
        per_agent = int(unwrapped._obs_bits_per_agent)
        per_shelf = int(unwrapped._obs_bits_per_shelf)
        sensor_range = int(unwrapped.sensor_range)
        cell_stride = per_agent + per_shelf
        expected_dim = self_bits + (1 + 2 * sensor_range) ** 2 * cell_stride
        if pool.shape[1] == expected_dim:
            scores = np.zeros(pool.shape[0], dtype=np.float32)
            for loc in range((1 + 2 * sensor_range) ** 2):
                base = self_bits + loc * cell_stride + per_agent
                scores += pool[:, base + 1]
            return scores

    return np.zeros(pool.shape[0], dtype=np.float32)


def sample_objective_rollout_probe_obs(
    env: gym.Env,
    rollout: Rollout,
    batch_size: int,
    rng: np.random.Generator,
) -> torch.Tensor:
    pool = rollout.obs.reshape(-1, rollout.obs.shape[-1])
    scores = _objective_visibility_scores(env, pool)
    candidate_indices = np.flatnonzero(scores > 0.0)
    if candidate_indices.size == 0:
        return sample_common_probe_obs(rollout, batch_size, rng)

    count = min(max(int(batch_size), 1), int(candidate_indices.size))
    replace = count > candidate_indices.size
    sampled = rng.choice(candidate_indices, size=count, replace=replace)
    return pool[torch.as_tensor(sampled, dtype=torch.long)]


def build_objective_probe_bank(
    env: gym.Env,
    obs_dim: int,
    batch_size: int,
    rng: np.random.Generator,
) -> Optional[torch.Tensor]:
    for builder in (
        _build_mtgrid_objective_probe_bank,
        _build_rware_objective_probe_bank,
    ):
        probes = builder(env, obs_dim, batch_size, rng)
        if probes is not None:
            return probes
    return None


def sample_policy_probe_obs(
    cfg: TrainConfig,
    env: gym.Env,
    rollout: Rollout,
    rng: np.random.Generator,
    batch_size: Optional[int] = None,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    batch_size = max(
        int(cfg.policy_probe_batch_size if batch_size is None else batch_size),
        1,
    )
    obs_dim = int(rollout.obs.shape[-1])
    source = cfg.probe_source
    meta = {
        "objective_probe_count": 0.0,
        "rollout_probe_count": 0.0,
    }

    if source == "rollout":
        probes = sample_common_probe_obs(rollout, batch_size, rng)
        meta["rollout_probe_count"] = float(probes.shape[0])
        return probes, meta

    if source == "objective-rollout":
        probes = sample_objective_rollout_probe_obs(env, rollout, batch_size, rng)
        meta["objective_probe_count"] = float(probes.shape[0])
        return probes, meta

    if source == "objective":
        probes = build_objective_probe_bank(env, obs_dim, batch_size, rng)
        if probes is None:
            probes = sample_objective_rollout_probe_obs(env, rollout, batch_size, rng)
        meta["objective_probe_count"] = float(probes.shape[0])
        return probes, meta

    if source == "mixed":
        objective_count = int(round(batch_size * cfg.objective_probe_fraction))
        objective_count = int(np.clip(objective_count, 1, batch_size))
        rollout_count = batch_size - objective_count
        objective_probes = build_objective_probe_bank(env, obs_dim, objective_count, rng)
        if objective_probes is None:
            objective_probes = sample_objective_rollout_probe_obs(
                env,
                rollout,
                objective_count,
                rng,
            )
        probes = [objective_probes]
        meta["objective_probe_count"] = float(objective_probes.shape[0])
        if rollout_count > 0:
            rollout_probes = sample_common_probe_obs(rollout, rollout_count, rng)
            probes.append(rollout_probes)
            meta["rollout_probe_count"] = float(rollout_probes.shape[0])
        return torch.cat(probes, dim=0), meta

    raise ValueError("--probe-source must be rollout, objective-rollout, objective, or mixed")


def sample_agent_conditioned_policy_probe_obs(
    cfg: TrainConfig,
    env: gym.Env,
    rollout: Rollout,
    rng: np.random.Generator,
    n_agents: int,
    batch_size: Optional[int] = None,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    batch_size = max(
        int(cfg.policy_probe_batch_size if batch_size is None else batch_size),
        1,
    )
    obs_dim = int(rollout.obs.shape[-1])
    team_ids = _agent_team_ids_from_env(env, n_agents)
    meta = {
        "objective_probe_count": 0.0,
        "rollout_probe_count": 0.0,
        "agent_conditioned_probe": 1.0,
    }

    if team_ids is None:
        shared, shared_meta = sample_policy_probe_obs(
            cfg,
            env,
            rollout,
            rng,
            batch_size=batch_size,
        )
        meta.update(shared_meta)
        meta["agent_conditioned_probe_fallback"] = 1.0
        return _repeat_shared_probes_for_agents(shared, n_agents), meta

    source = cfg.probe_source
    if source == "rollout":
        probes = sample_common_probe_obs(rollout, batch_size, rng)
        meta["rollout_probe_count"] = float(probes.shape[0])
        return _repeat_shared_probes_for_agents(probes, n_agents), meta

    if source == "objective-rollout":
        probes = sample_objective_rollout_probe_obs(env, rollout, batch_size, rng)
        meta["objective_probe_count"] = float(probes.shape[0])
        return _repeat_shared_probes_for_agents(probes, n_agents), meta

    if source == "objective":
        probes = _build_agent_conditioned_objective_probe_bank(
            env,
            obs_dim,
            batch_size,
            rng,
            team_ids,
        )
        if probes is None:
            shared = sample_objective_rollout_probe_obs(env, rollout, batch_size, rng)
            probes = _repeat_shared_probes_for_agents(shared, n_agents)
            meta["agent_conditioned_probe_fallback"] = 1.0
        meta["objective_probe_count"] = float(probes.shape[1])
        meta["agent_conditioned_probe_agents"] = float(probes.shape[0])
        return probes, meta

    if source == "mixed":
        objective_count = int(round(batch_size * cfg.objective_probe_fraction))
        objective_count = int(np.clip(objective_count, 1, batch_size))
        rollout_count = batch_size - objective_count
        objective_probes = _build_agent_conditioned_objective_probe_bank(
            env,
            obs_dim,
            objective_count,
            rng,
            team_ids,
        )
        if objective_probes is None:
            shared = sample_objective_rollout_probe_obs(
                env,
                rollout,
                objective_count,
                rng,
            )
            objective_probes = _repeat_shared_probes_for_agents(shared, n_agents)
            meta["agent_conditioned_probe_fallback"] = 1.0
        probes = [objective_probes]
        meta["objective_probe_count"] = float(objective_probes.shape[1])
        if rollout_count > 0:
            rollout_probes = sample_common_probe_obs(rollout, rollout_count, rng)
            probes.append(_repeat_shared_probes_for_agents(rollout_probes, n_agents))
            meta["rollout_probe_count"] = float(rollout_probes.shape[0])
        stacked = torch.cat(probes, dim=1)
        meta["agent_conditioned_probe_agents"] = float(stacked.shape[0])
        return stacked, meta

    raise ValueError("--probe-source must be rollout, objective-rollout, objective, or mixed")


def _as_probe_sequences(probe_obs: torch.Tensor) -> torch.Tensor:
    if probe_obs.ndim == 2:
        return probe_obs.unsqueeze(1)
    if probe_obs.ndim == 3:
        return probe_obs
    raise ValueError("probe observations must have shape (M, obs_dim) or (M, Lp, obs_dim)")


def _as_agent_conditioned_probe_sequences(
    probe_obs: torch.Tensor,
    n_agents: int,
) -> torch.Tensor:
    if probe_obs.ndim == 4:
        if probe_obs.shape[0] != n_agents:
            raise ValueError(
                "agent-conditioned probe observations must have shape "
                "(n_agents, M, Lp, obs_dim)"
            )
        return probe_obs
    shared = _as_probe_sequences(probe_obs)
    return shared.unsqueeze(0).expand(int(n_agents), -1, -1, -1)


def _canonical_probe_count(probe_sequences: torch.Tensor) -> float:
    if probe_sequences.ndim == 4:
        return float(probe_sequences.shape[1] * probe_sequences.shape[2])
    if probe_sequences.ndim == 3:
        return float(probe_sequences.shape[0] * probe_sequences.shape[1])
    if probe_sequences.ndim == 2:
        return float(probe_sequences.shape[0])
    return 0.0


def sample_policy_probe_sequences(
    cfg: TrainConfig,
    env: gym.Env,
    rollout: Rollout,
    rng: np.random.Generator,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    if cfg.fixed_canonical_probes:
        if cfg.probe_source != "objective" or cfg.policy_probe_team_conditioning != "shared":
            raise ValueError("Fixed canonical probes require objective source and shared conditioning")
        cached = getattr(env.unwrapped, "_pgct_canonical_bank", None)
        if cached is not None:
            return cached[0].clone(), dict(cached[1])
        # Dedicated seed: neither PPO shuffle nor rollout RNG is consumed.
        rng = np.random.default_rng(cfg.seed + 9173)
    sequence_length = max(int(cfg.pgct_probe_sequence_length), 1)
    sequence_count = max(int(cfg.policy_probe_batch_size), 1)
    flat_count = sequence_count * sequence_length
    n_agents = int(getattr(env.unwrapped, "n_agents", 1))
    if cfg.fixed_canonical_probes:
        obs_dim = int(env.observation_space[0].shape[-1])
        flat_probes = build_objective_probe_bank(env, obs_dim, flat_count, rng)
        if flat_probes is None:
            raise ValueError("This environment has no external objective probe builder")
        meta = {"objective_probe_count": float(flat_count), "rollout_probe_count": 0.0}
        probe_axis = 0
    elif cfg.policy_probe_team_conditioning == "agent-team":
        flat_probes, meta = sample_agent_conditioned_policy_probe_obs(
            cfg,
            env,
            rollout,
            rng,
            n_agents=n_agents,
            batch_size=flat_count,
        )
        probe_axis = 1
    else:
        flat_probes, meta = sample_policy_probe_obs(
            cfg,
            env,
            rollout,
            rng,
            batch_size=flat_count,
        )
        probe_axis = 0

    if flat_probes.shape[probe_axis] <= 0:
        raise ValueError("Cannot build an empty policy probe sequence bank")

    if cfg.policy_probe_team_conditioning == "agent-team":
        if flat_probes.shape[1] < flat_count:
            indices = torch.arange(flat_count, dtype=torch.long) % flat_probes.shape[1]
            flat_probes = flat_probes[:, indices]
        elif flat_probes.shape[1] > flat_count:
            flat_probes = flat_probes[:, :flat_count]
        sequences = flat_probes.reshape(
            flat_probes.shape[0],
            sequence_count,
            sequence_length,
            -1,
        ).contiguous()
    else:
        if flat_probes.shape[0] < flat_count:
            indices = torch.arange(flat_count, dtype=torch.long) % flat_probes.shape[0]
            flat_probes = flat_probes[indices]
        elif flat_probes.shape[0] > flat_count:
            flat_probes = flat_probes[:flat_count]
        sequences = flat_probes.reshape(sequence_count, sequence_length, -1).contiguous()

    meta.update(
        {
            "probe_sequence_count": float(sequence_count),
            "probe_sequence_length": float(sequence_length),
            "probe_team_conditioned": float(sequences.ndim == 4),
        }
    )
    if cfg.fixed_canonical_probes:
        env.unwrapped._pgct_canonical_bank = (sequences.clone(), dict(meta))
    return sequences, meta


@torch.no_grad()
def policy_distance_similarity_matrices(
    agents: Sequence[RecurrentActorCritic],
    probe_obs: torch.Tensor,
    device: torch.device,
    temperature: float,
    return_information: bool = False,
):
    n_agents = len(agents)
    probe_sequences = _as_agent_conditioned_probe_sequences(probe_obs, n_agents).to(
        device
    )
    batch_size = int(probe_sequences.shape[1])
    sequence_length = int(probe_sequences.shape[2])
    policy_probs = []
    for agent_id, net in enumerate(agents):
        actor_h, _ = net.initial_hidden(batch_size, device)
        step_probs = []
        agent_probe_sequences = probe_sequences[agent_id]
        for step_idx in range(sequence_length):
            logits, actor_h = net._actor_step(
                agent_probe_sequences[:, step_idx],
                actor_h,
            )
            step_probs.append(torch.softmax(logits, dim=-1))
        policy_probs.append(torch.stack(step_probs, dim=1).cpu())

    # KL(pi || Uniform) = log(|A|) - H(pi). It is near zero for the
    # uninformative random policies produced by small initial actor logits.
    policy_information = np.asarray([
        float((math.log(probs.shape[-1]) - (
            -torch.sum(probs.clamp_min(1e-8) * torch.log(probs.clamp_min(1e-8)), dim=-1)
        )).mean().item())
        for probs in policy_probs
    ], dtype=np.float32)

    distance = np.zeros((n_agents, n_agents), dtype=np.float32)
    similarity = np.eye(n_agents, dtype=np.float32)
    eps = 1e-8
    temp = max(float(temperature), eps)
    for i in range(n_agents):
        for j in range(i + 1, n_agents):
            p = policy_probs[i].clamp_min(eps)
            q = policy_probs[j].clamp_min(eps)
            m = 0.5 * (p + q)
            kl_pm = torch.sum(p * (torch.log(p) - torch.log(m)), dim=-1)
            kl_qm = torch.sum(q * (torch.log(q) - torch.log(m)), dim=-1)
            js = 0.5 * (kl_pm + kl_qm)
            raw_distance = float(js.mean().item())
            score = float(math.exp(-raw_distance / temp))
            distance[i, j] = raw_distance
            distance[j, i] = raw_distance
            similarity[i, j] = score
            similarity[j, i] = score
    if return_information:
        return distance, similarity, policy_information
    return distance, similarity


@torch.no_grad()
def policy_similarity_matrix(
    agents: Sequence[RecurrentActorCritic],
    probe_obs: torch.Tensor,
    device: torch.device,
    temperature: float,
) -> np.ndarray:
    _, similarity = policy_distance_similarity_matrices(
        agents,
        probe_obs,
        device=device,
        temperature=temperature,
    )
    return similarity


def collect_rollout(
    env: gym.Env,
    agents: Sequence[RecurrentActorCritic],
    state: RunnerState,
    rollout_steps: int,
    obs_dim: int,
    action_dim: int,
    recurrent_hidden_dim: int,
    device: torch.device,
    use_action_mask: bool,
) -> Tuple[Rollout, RunnerState, Dict[str, float]]:
    n_agents = len(agents)

    obs_buf = torch.zeros((rollout_steps, n_agents, obs_dim), dtype=torch.float32)
    actions_buf = torch.zeros((rollout_steps, n_agents), dtype=torch.long)
    action_masks_buf = torch.ones(
        (rollout_steps, n_agents, action_dim),
        dtype=torch.float32,
    )
    logprob_buf = torch.zeros((rollout_steps, n_agents), dtype=torch.float32)
    rewards_buf = torch.zeros((rollout_steps, n_agents), dtype=torch.float32)
    timeout_values_buf = torch.zeros_like(rewards_buf)
    dones_buf = torch.zeros((rollout_steps,), dtype=torch.float32)
    values_buf = torch.zeros((rollout_steps, n_agents), dtype=torch.float32)
    actor_h_buf = torch.zeros(
        (rollout_steps, n_agents, recurrent_hidden_dim), dtype=torch.float32
    )
    critic_h_buf = torch.zeros_like(actor_h_buf)

    completed_returns: List[float] = []
    completed_task_returns: List[float] = []
    completed_shaping_returns: List[float] = []
    completed_lengths: List[int] = []
    completed_team_events: List[float] = []
    completed_deliveries: List[float] = []
    completed_returns_home: List[float] = []
    completed_pickups: List[float] = []
    completed_requested_pickups: List[float] = []
    completed_wrong_returns: List[float] = []
    completed_requested_shelf_seen: List[float] = []
    completed_carrying_at_end: List[float] = []
    completed_invalid_toggle_rates: List[float] = []
    completed_failed_forward_rates: List[float] = []
    completed_blocking_agent_rates: List[float] = []
    completed_deadlock_agent_rates: List[float] = []
    completed_steps_to_seen: List[float] = []
    completed_steps_to_pickup: List[float] = []
    completed_steps_pickup_to_goal: List[float] = []
    completed_steps_delivery_to_return: List[float] = []
    max_action_probs: List[float] = []
    actor_hidden_norms: List[float] = []
    critic_hidden_norms: List[float] = []
    action_prob_sums = np.zeros((action_dim,), dtype=np.float64)
    action_prob_count = 0.0
    valid_pickup_opportunity_count = 0.0
    valid_return_opportunity_count = 0.0
    return_phase_step_count = 0.0
    toggle_when_pickup_valid_count = 0.0
    toggle_when_return_valid_count = 0.0
    toggle_policy_prob_when_pickup_valid_sum = 0.0
    toggle_policy_prob_when_return_valid_sum = 0.0
    toggle_count = 0.0
    requested_pickup_count = 0.0
    return_success_count = 0.0
    position_change_count = 0.0
    forward_failure_count = 0.0
    agent_step_count = 0.0
    wall_state_count = 0.0
    wall_action_counts = np.zeros((action_dim,), dtype=np.float64)
    wall_action_prob_sums = np.zeros((action_dim,), dtype=np.float64)
    wall_action_prob_count = 0.0
    per_agent_step_counts = np.zeros((n_agents,), dtype=np.float64)
    per_agent_position_change_counts = np.zeros((n_agents,), dtype=np.float64)
    per_agent_pickup_counts = np.zeros((n_agents,), dtype=np.float64)
    per_agent_delivery_counts = np.zeros((n_agents,), dtype=np.float64)
    per_agent_return_counts = np.zeros((n_agents,), dtype=np.float64)
    per_agent_failed_forward_counts = np.zeros((n_agents,), dtype=np.float64)
    per_agent_blocking_counts = np.zeros((n_agents,), dtype=np.float64)
    per_agent_deadlock_counts = np.zeros((n_agents,), dtype=np.float64)
    return_distances: List[float] = []

    for step in range(rollout_steps):
        requested_shelf_visible = rware_requested_shelf_visibility(env, n_agents)
        if np.any(requested_shelf_visible):
            state.episode_requested_shelf_seen = True
            if state.episode_first_requested_shelf_seen_step is None:
                state.episode_first_requested_shelf_seen_step = state.episode_length
        pickup_opportunities = rware_requested_pickup_opportunities(env, n_agents)
        return_opportunities, return_phase, step_return_distances = (
            rware_return_diagnostics(env, n_agents)
        )
        pre_positions = rware_agent_positions(env, n_agents)
        return_distances.extend(step_return_distances)

        flat_obs = flatten_multi_agent_obs(env, state.obs)
        obs_tensor = torch.as_tensor(flat_obs, dtype=torch.float32, device=device)
        action_masks = local_rware_action_masks(
            env,
            n_agents,
            action_dim,
            enabled=use_action_mask,
        )
        diagnostic_action_masks = (
            action_masks
            if use_action_mask
            else local_rware_action_masks(
                env,
                n_agents,
                action_dim,
                enabled=True,
            )
        )
        action_mask_tensor = torch.as_tensor(
            action_masks,
            dtype=torch.float32,
            device=device,
        )
        blocked_forward_states = np.zeros((n_agents,), dtype=bool)
        if action_dim > int(Action.FORWARD.value):
            blocked_forward_states = (
                diagnostic_action_masks[:, int(Action.FORWARD.value)] <= 0.0
            )
        valid_pickup_opportunity_count += float(np.sum(pickup_opportunities))
        valid_return_opportunity_count += float(np.sum(return_opportunities))
        return_phase_step_count += float(np.sum(return_phase))
        wall_state_count += float(np.sum(blocked_forward_states))
        agent_step_count += float(n_agents)
        obs_buf[step] = obs_tensor.cpu()
        action_masks_buf[step] = action_mask_tensor.cpu()
        actor_h_buf[step] = state.actor_h.cpu()
        critic_h_buf[step] = state.critic_h.cpu()

        actions: List[int] = []
        for agent_id, net in enumerate(agents):
            action_probs = net.action_probabilities(
                obs_tensor[agent_id],
                state.actor_h[agent_id],
                action_mask=action_mask_tensor[agent_id],
            )
            action_probs_np = action_probs.squeeze(0).detach().cpu().numpy()
            action_prob_sums += action_probs_np
            action_prob_count += 1.0
            if blocked_forward_states[agent_id]:
                wall_action_prob_sums += action_probs_np
                wall_action_prob_count += 1.0
            max_action_probs.append(float(np.max(action_probs_np)))
            if (
                action_dim > int(Action.TOGGLE_LOAD.value)
                and pickup_opportunities[agent_id]
            ):
                toggle_policy_prob_when_pickup_valid_sum += float(
                    action_probs_np[int(Action.TOGGLE_LOAD.value)]
                )
            if (
                action_dim > int(Action.TOGGLE_LOAD.value)
                and return_opportunities[agent_id]
            ):
                toggle_policy_prob_when_return_valid_sum += float(
                    action_probs_np[int(Action.TOGGLE_LOAD.value)]
                )
            action, log_prob, value, next_actor_h, next_critic_h = net.act(
                obs_tensor[agent_id],
                state.actor_h[agent_id],
                state.critic_h[agent_id],
                deterministic=False,
                action_mask=action_mask_tensor[agent_id],
            )
            actions.append(int(action.item()))
            actions_buf[step, agent_id] = int(action.item())
            logprob_buf[step, agent_id] = float(log_prob.item())
            values_buf[step, agent_id] = float(value.item())
            state.actor_h[agent_id] = next_actor_h.squeeze(0)
            state.critic_h[agent_id] = next_critic_h.squeeze(0)
            actor_hidden_norms.append(float(state.actor_h[agent_id].norm().item()))
            critic_hidden_norms.append(float(state.critic_h[agent_id].norm().item()))

        actions_array = np.asarray(actions, dtype=np.int64)
        if action_dim > int(Action.TOGGLE_LOAD.value):
            toggle_actions = actions_array == int(Action.TOGGLE_LOAD.value)
            toggle_count += float(np.sum(toggle_actions))
            toggle_when_pickup_valid_count += float(
                np.sum(toggle_actions & pickup_opportunities)
            )
            toggle_when_return_valid_count += float(
                np.sum(toggle_actions & return_opportunities)
            )
        for action in actions_array[blocked_forward_states]:
            if 0 <= int(action) < action_dim:
                wall_action_counts[int(action)] += 1.0

        masked_failed_forwards = 0.0
        if action_dim > int(Action.FORWARD.value):
            masked_failed_forwards = float(
                sum(
                    action == int(Action.FORWARD.value)
                    and diagnostic_action_masks[agent_id, int(Action.FORWARD.value)]
                    <= 0.0
                    for agent_id, action in enumerate(actions)
                )
            )
        next_obs, rewards, done, truncated, info = env.step(actions)
        failed_forwards = float(
            info.get("failed_forwards", masked_failed_forwards)
        )
        blocking_agents = float(info.get("blocking_agents", 0.0))
        deadlocked_agents = float(info.get("deadlocked_agents", 0.0))
        forward_failure_count += failed_forwards
        post_positions = rware_agent_positions(env, n_agents)
        moved_agents = np.asarray(
            [
                before != after
                for before, after in zip(pre_positions, post_positions)
            ],
            dtype=np.float64,
        )
        position_change_count += float(np.sum(moved_agents))
        per_agent_step_counts += 1.0
        per_agent_position_change_counts += moved_agents
        add_agent_event_counts(
            info.get("pickup_events", []),
            per_agent_pickup_counts,
            "agent_id",
        )
        add_agent_event_counts(
            info.get("delivery_events", []),
            per_agent_delivery_counts,
            "carrier_id",
        )
        add_agent_event_counts(
            info.get("return_events", []),
            per_agent_return_counts,
            "carrier_id",
        )
        add_agent_event_counts(
            info.get("failed_forward_events", []),
            per_agent_failed_forward_counts,
            "agent_id",
        )
        add_agent_event_counts(
            info.get("blocking_agent_events", []),
            per_agent_blocking_counts,
            "agent_id",
        )
        add_agent_event_counts(
            info.get("deadlock_events", []),
            per_agent_deadlock_counts,
            "agent_id",
        )
        terminal = bool(done or truncated)
        if truncated and not done:
            # Bootstrap the final observation with its pre-reset recurrent state.
            # GAE still stops at the reset, so the next episode cannot leak in.
            timeout_obs = torch.as_tensor(
                flatten_multi_agent_obs(env, next_obs), dtype=torch.float32, device=device
            )
            with torch.no_grad():
                for agent_id, net in enumerate(agents):
                    timeout_value, _ = net._critic_step(
                        timeout_obs[agent_id].unsqueeze(0),
                        state.critic_h[agent_id].unsqueeze(0),
                    )
                    timeout_values_buf[step, agent_id] = float(timeout_value.item())
        rewards_array = np.asarray(rewards, dtype=np.float32)
        shaping_array = np.asarray(
            info.get("reward_shaping", np.zeros((n_agents,), dtype=np.float32)),
            dtype=np.float32,
        )
        if shaping_array.shape != rewards_array.shape:
            shaping_array = np.zeros_like(rewards_array)
        task_rewards_array = rewards_array - shaping_array

        rewards_buf[step] = torch.as_tensor(rewards_array, dtype=torch.float32)
        dones_buf[step] = float(terminal)
        state.episode_returns += rewards_array
        state.episode_task_returns += task_rewards_array
        state.episode_shaping_returns += shaping_array
        state.episode_length += 1
        deliveries = float(info.get("deliveries", info.get("collections", 0)))
        returns_home = float(info.get("returns", 0))
        pickups = float(info.get("pickups", 0))
        pickup_events = info.get("pickup_events", [])
        requested_pickups = (
            float(sum(1 for event in pickup_events if event.get("requested", False)))
            if isinstance(pickup_events, list)
            else pickups
        )
        requested_pickup_count += requested_pickups
        return_success_count += returns_home
        invalid_toggles = float(info.get("invalid_toggles", 0))
        wrong_returns = float(info.get("wrong_returns", 0))
        state.episode_deliveries += deliveries
        state.episode_returns_home += returns_home
        state.episode_pickups += pickups
        state.episode_requested_pickups += requested_pickups
        state.episode_invalid_toggles += invalid_toggles
        state.episode_wrong_returns += wrong_returns
        state.episode_failed_forwards += failed_forwards
        state.episode_blocking_agents += blocking_agents
        state.episode_deadlocked_agents += deadlocked_agents

        if pickups > 0.0 and state.episode_first_pickup_step is None:
            state.episode_first_pickup_step = state.episode_length
        if deliveries > 0.0 and state.episode_first_delivery_step is None:
            state.episode_first_delivery_step = state.episode_length
        if returns_home > 0.0 and state.episode_first_return_step is None:
            state.episode_first_return_step = state.episode_length

        if terminal:
            episode_action_count = max(float(state.episode_length * n_agents), 1.0)
            carrying_at_end = any(
                getattr(agent, "carrying_shelf", None) is not None
                for agent in getattr(env.unwrapped, "agents", [])
            )
            completed_returns.append(float(np.mean(state.episode_returns)))
            completed_task_returns.append(float(np.mean(state.episode_task_returns)))
            completed_shaping_returns.append(
                float(np.mean(state.episode_shaping_returns))
            )
            completed_lengths.append(float(state.episode_length))
            completed_deliveries.append(float(state.episode_deliveries))
            completed_returns_home.append(float(state.episode_returns_home))
            completed_pickups.append(float(state.episode_pickups))
            completed_requested_pickups.append(
                float(state.episode_requested_pickups)
            )
            completed_wrong_returns.append(float(state.episode_wrong_returns))
            completed_requested_shelf_seen.append(
                float(state.episode_requested_shelf_seen)
            )
            completed_carrying_at_end.append(float(carrying_at_end))
            completed_invalid_toggle_rates.append(
                float(state.episode_invalid_toggles / episode_action_count)
            )
            completed_failed_forward_rates.append(
                float(state.episode_failed_forwards / episode_action_count)
            )
            completed_blocking_agent_rates.append(
                float(state.episode_blocking_agents / episode_action_count)
            )
            completed_deadlock_agent_rates.append(
                float(state.episode_deadlocked_agents / episode_action_count)
            )
            completed_steps_to_seen.append(
                float(state.episode_first_requested_shelf_seen_step)
                if state.episode_first_requested_shelf_seen_step is not None
                else float("nan")
            )
            completed_steps_to_pickup.append(
                float(state.episode_first_pickup_step)
                if state.episode_first_pickup_step is not None
                else float("nan")
            )
            completed_steps_pickup_to_goal.append(
                optional_step_delta(
                    state.episode_first_delivery_step,
                    state.episode_first_pickup_step,
                )
            )
            completed_steps_delivery_to_return.append(
                optional_step_delta(
                    state.episode_first_return_step,
                    state.episode_first_delivery_step,
                )
            )
            completed_team_events.append(
                float(state.episode_deliveries + state.episode_returns_home)
            )

            next_obs, _ = env.reset()
            state.actor_h.zero_()
            state.critic_h.zero_()
            state.episode_returns[:] = 0.0
            state.episode_task_returns[:] = 0.0
            state.episode_shaping_returns[:] = 0.0
            state.episode_length = 0
            state.episode_deliveries = 0.0
            state.episode_returns_home = 0.0
            state.episode_pickups = 0.0
            state.episode_requested_pickups = 0.0
            state.episode_invalid_toggles = 0.0
            state.episode_wrong_returns = 0.0
            state.episode_failed_forwards = 0.0
            state.episode_blocking_agents = 0.0
            state.episode_deadlocked_agents = 0.0
            state.episode_requested_shelf_seen = False
            state.episode_first_requested_shelf_seen_step = None
            state.episode_first_pickup_step = None
            state.episode_first_delivery_step = None
            state.episode_first_return_step = None

        state.obs = next_obs

    flat_last_obs = flatten_multi_agent_obs(env, state.obs)
    last_obs_tensor = torch.as_tensor(flat_last_obs, dtype=torch.float32, device=device)
    last_values = torch.zeros((n_agents,), dtype=torch.float32)
    for agent_id, net in enumerate(agents):
        with torch.no_grad():
            value, _ = net._critic_step(
                last_obs_tensor[agent_id].unsqueeze(0),
                state.critic_h[agent_id].unsqueeze(0),
            )
        last_values[agent_id] = float(value.item())

    rollout = Rollout(
        obs=obs_buf,
        actions=actions_buf,
        action_masks=action_masks_buf,
        old_log_probs=logprob_buf,
        rewards=rewards_buf,
        dones=dones_buf,
        values=values_buf,
        actor_h=actor_h_buf,
        critic_h=critic_h_buf,
        last_values=last_values,
        timeout_values=timeout_values_buf,
    )
    flat_actions = actions_buf.reshape(-1).numpy()
    action_metrics = {
        f"action_{action.name.lower()}_rate": float(
            np.mean(flat_actions == int(action.value))
        )
        for action in Action
        if int(action.value) < action_dim
    }
    action_probability_metrics = {
        f"action_{action.name.lower()}_probability": float(
            action_prob_sums[int(action.value)] / max(action_prob_count, 1.0)
        )
        for action in Action
        if int(action.value) < action_dim
    }
    wall_action_metrics = {
        f"wall_action_{action.name.lower()}_rate": (
            float(wall_action_counts[int(action.value)] / wall_state_count)
            if wall_state_count > 0.0
            else float("nan")
        )
        for action in Action
        if int(action.value) < action_dim
    }
    wall_action_probability_metrics = {
        f"wall_action_{action.name.lower()}_probability": (
            float(wall_action_prob_sums[int(action.value)] / wall_action_prob_count)
            if wall_action_prob_count > 0.0
            else float("nan")
        )
        for action in Action
        if int(action.value) < action_dim
    }
    agent_step_denominator = max(agent_step_count, 1.0)
    pickup_opportunity_denominator = max(valid_pickup_opportunity_count, 1.0)
    return_opportunity_denominator = max(valid_return_opportunity_count, 1.0)
    metrics = {
        "episode_return": float(np.mean(completed_returns))
        if completed_returns
        else float("nan"),
        "episode_task_return": float(np.mean(completed_task_returns))
        if completed_task_returns
        else float("nan"),
        "episode_shaping_return": float(np.mean(completed_shaping_returns))
        if completed_shaping_returns
        else float("nan"),
        "episode_length": float(np.mean(completed_lengths))
        if completed_lengths
        else float("nan"),
        "team_events": float(np.mean(completed_team_events))
        if completed_team_events
        else float("nan"),
        "deliveries": float(np.mean(completed_deliveries))
        if completed_deliveries
        else float("nan"),
        "delivery_count": float(np.mean(completed_deliveries))
        if completed_deliveries
        else float("nan"),
        "returns_home": float(np.mean(completed_returns_home))
        if completed_returns_home
        else float("nan"),
        "requested_shelf_pickups": float(np.mean(completed_requested_pickups))
        if completed_requested_pickups
        else float("nan"),
        "requested_shelf_seen_rate": float(np.mean(completed_requested_shelf_seen))
        if completed_requested_shelf_seen
        else float("nan"),
        "pickup_success_rate": float(np.mean(np.asarray(completed_pickups) > 0.0))
        if completed_pickups
        else float("nan"),
        "requested_shelf_pickup_rate": float(
            np.mean(np.asarray(completed_requested_pickups) > 0.0)
        )
        if completed_requested_pickups
        else float("nan"),
        "delivery_success_rate": float(np.mean(np.asarray(completed_deliveries) > 0.0))
        if completed_deliveries
        else float("nan"),
        "return_success_rate": float(np.mean(np.asarray(completed_returns_home) > 0.0))
        if completed_returns_home
        else float("nan"),
        "full_cycle_success_rate": float(
            np.mean(
                (np.asarray(completed_deliveries) > 0.0)
                & (np.asarray(completed_returns_home) > 0.0)
            )
        )
        if completed_deliveries and completed_returns_home
        else float("nan"),
        "delivered_but_not_returned_rate": float(
            np.mean(np.asarray(completed_deliveries) > np.asarray(completed_returns_home))
        )
        if completed_deliveries and completed_returns_home
        else float("nan"),
        "wrong_returns": float(np.mean(completed_wrong_returns))
        if completed_wrong_returns
        else float("nan"),
        "invalid_toggle_rate": float(np.mean(completed_invalid_toggle_rates))
        if completed_invalid_toggle_rates
        else float("nan"),
        "failed_forward_rate": float(np.mean(completed_failed_forward_rates))
        if completed_failed_forward_rates
        else float("nan"),
        "blocking_agent_rate": float(np.mean(completed_blocking_agent_rates))
        if completed_blocking_agent_rates
        else float("nan"),
        "deadlock_agent_rate": float(np.mean(completed_deadlock_agent_rates))
        if completed_deadlock_agent_rates
        else float("nan"),
        "carrying_at_episode_end": float(np.mean(completed_carrying_at_end))
        if completed_carrying_at_end
        else float("nan"),
        "mean_steps_to_first_requested_shelf": finite_mean(completed_steps_to_seen),
        "mean_steps_to_pickup": finite_mean(completed_steps_to_pickup),
        "mean_steps_pickup_to_goal": finite_mean(completed_steps_pickup_to_goal),
        "mean_steps_delivery_to_return": finite_mean(
            completed_steps_delivery_to_return
        ),
        "episodes": float(len(completed_returns)),
        "valid_action_fraction": float(action_masks_buf.mean().item()),
        "max_action_probability": float(np.mean(max_action_probs))
        if max_action_probs
        else float("nan"),
        "valid_pickup_opportunity_count": float(valid_pickup_opportunity_count),
        "valid_pickup_opportunity_rate": float(
            valid_pickup_opportunity_count / agent_step_denominator
        ),
        "valid_return_opportunity_count": float(valid_return_opportunity_count),
        "valid_return_opportunity_rate": float(
            valid_return_opportunity_count / agent_step_denominator
        ),
        "return_phase_step_count": float(return_phase_step_count),
        "return_phase_step_rate": float(
            return_phase_step_count / agent_step_denominator
        ),
        "toggle_count": float(toggle_count),
        "toggle_probability_when_pickup_valid": float(
            toggle_when_pickup_valid_count / pickup_opportunity_denominator
        ),
        "toggle_probability_when_return_valid": float(
            toggle_when_return_valid_count / return_opportunity_denominator
        ),
        "toggle_policy_probability_when_pickup_valid": float(
            toggle_policy_prob_when_pickup_valid_sum / pickup_opportunity_denominator
        ),
        "toggle_policy_probability_when_return_valid": float(
            toggle_policy_prob_when_return_valid_sum / return_opportunity_denominator
        ),
        "pickup_conversion_rate": float(
            requested_pickup_count / pickup_opportunity_denominator
        ),
        "return_conversion_rate": float(
            return_success_count / return_opportunity_denominator
        ),
        "position_change_rate": float(
            position_change_count / agent_step_denominator
        ),
        "forward_failure_count": float(forward_failure_count),
        "wall_state_count": float(wall_state_count),
        "wall_state_rate": float(wall_state_count / agent_step_denominator),
        "mean_return_home_distance": finite_mean(return_distances),
        "actor_hidden_norm": float(np.mean(actor_hidden_norms))
        if actor_hidden_norms
        else float("nan"),
        "critic_hidden_norm": float(np.mean(critic_hidden_norms))
        if critic_hidden_norms
        else float("nan"),
        **action_probability_metrics,
        **wall_action_metrics,
        **wall_action_probability_metrics,
        **per_agent_metrics(
            "",
            n_agents,
            per_agent_step_counts,
            per_agent_position_change_counts,
            per_agent_pickup_counts,
            per_agent_delivery_counts,
            per_agent_return_counts,
            per_agent_failed_forward_counts,
            per_agent_blocking_counts,
            per_agent_deadlock_counts,
        ),
        **action_metrics,
    }
    return rollout, state, metrics


def compute_gae(
    rewards: torch.Tensor,
    dones: torch.Tensor,
    values: torch.Tensor,
    last_values: torch.Tensor,
    gamma: float,
    gae_lambda: float,
    timeout_values: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    rollout_steps, n_agents = rewards.shape
    advantages = torch.zeros_like(rewards)
    last_gae = torch.zeros((n_agents,), dtype=torch.float32)
    for step in reversed(range(rollout_steps)):
        if step == rollout_steps - 1:
            next_value = last_values
        else:
            next_value = values[step + 1]
        next_nonterminal = 1.0 - dones[step]
        delta = rewards[step] + gamma * next_value * next_nonterminal - values[step]
        if timeout_values is not None:
            delta = delta + gamma * timeout_values[step]
        last_gae = delta + gamma * gae_lambda * next_nonterminal * last_gae
        advantages[step] = last_gae
    returns = advantages + values
    return advantages, returns


def recurrent_minibatches(
    rollout: Rollout,
    agent_id: int,
    sequence_length: int,
    minibatch_chunks: int,
    rng: np.random.Generator,
):
    rollout_steps = rollout.obs.shape[0]
    starts = np.arange(0, rollout_steps, sequence_length, dtype=np.int64)
    rng.shuffle(starts)
    for offset in range(0, len(starts), minibatch_chunks):
        mb_starts = starts[offset : offset + minibatch_chunks]
        obs_seq = torch.stack(
            [rollout.obs[s : s + sequence_length, agent_id] for s in mb_starts], dim=1
        )
        actions_seq = torch.stack(
            [rollout.actions[s : s + sequence_length, agent_id] for s in mb_starts],
            dim=1,
        )
        action_mask_seq = torch.stack(
            [
                rollout.action_masks[s : s + sequence_length, agent_id]
                for s in mb_starts
            ],
            dim=1,
        )
        old_logprob_seq = torch.stack(
            [
                rollout.old_log_probs[s : s + sequence_length, agent_id]
                for s in mb_starts
            ],
            dim=1,
        )
        old_value_seq = torch.stack(
            [rollout.values[s : s + sequence_length, agent_id] for s in mb_starts],
            dim=1,
        )
        done_seq = torch.stack(
            [rollout.dones[s : s + sequence_length] for s in mb_starts],
            dim=1,
        )
        actor_h0 = rollout.actor_h[mb_starts, agent_id]
        critic_h0 = rollout.critic_h[mb_starts, agent_id]
        yield (
            obs_seq,
            actions_seq,
            action_mask_seq,
            old_logprob_seq,
            old_value_seq,
            done_seq,
            actor_h0,
            critic_h0,
            mb_starts,
        )


def recurrent_critic_probe_values(
    net: RecurrentActorCritic,
    probe_sequences: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    probe_sequences = _as_probe_sequences(probe_sequences).to(device)
    batch_size = int(probe_sequences.shape[0])
    sequence_length = int(probe_sequences.shape[1])
    _, critic_h = net.initial_hidden(batch_size, device)
    values = []
    for step_idx in range(sequence_length):
        value, critic_h = net._critic_step(probe_sequences[:, step_idx], critic_h)
        values.append(value)
    return torch.stack(values, dim=1)


@torch.no_grad()
def recurrent_critic_probe_targets(
    agents: Sequence[RecurrentActorCritic],
    probe_sequences: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    agent_probe_sequences = _as_agent_conditioned_probe_sequences(
        probe_sequences,
        len(agents),
    )
    return torch.stack(
        [
            recurrent_critic_probe_values(
                agent,
                agent_probe_sequences[agent_id],
                device,
            ).detach()
            for agent_id, agent in enumerate(agents)
        ],
        dim=0,
    )


def update_ippo(
    agents: Sequence[RecurrentActorCritic],
    optimizers: Sequence[torch.optim.Optimizer],
    rollout: Rollout,
    cfg: TrainConfig,
    graph_estimator: TeamGraphEstimator,
    device: torch.device,
    rng: np.random.Generator,
    peer_allocation: Optional[np.ndarray] = None,
    peer_probe_sequences: Optional[torch.Tensor] = None,
) -> Dict[str, float]:
    advantages, returns = compute_gae(
        rollout.rewards,
        rollout.dones,
        rollout.values,
        rollout.last_values,
        cfg.gamma,
        cfg.gae_lambda,
        rollout.timeout_values,
    )

    n_agents = len(agents)
    mean_abs_td_error = np.zeros((n_agents,), dtype=np.float64)
    agent_diagnostics: Dict[str, float] = {}
    metrics = {
        "policy_loss": [],
        "value_loss": [],
        "entropy": [],
        "approx_kl": [],
        "clip_fraction": [],
        "advantage_std": [],
        "reward_density": [],
        "positive_reward_density": [],
        "negative_reward_density": [],
        "actor_update_skipped": [],
        "pgct_peer_loss": [],
        "pgct_peer_weight": [],
    }

    peer_allocation_array: Optional[np.ndarray] = None
    peer_probe_tensor: Optional[torch.Tensor] = None
    peer_target_values: Optional[torch.Tensor] = None
    peer_donors: List[List[int]] = [[] for _ in range(n_agents)]
    if (
        cfg.peer_transfer_mode == "pgct"
        and cfg.pgct_peer_loss_coef > 0.0
        and peer_allocation is not None
        and peer_probe_sequences is not None
    ):
        peer_allocation_array = np.asarray(peer_allocation, dtype=np.float32)
        if peer_allocation_array.shape != (n_agents, n_agents):
            raise ValueError("peer_allocation must have shape (n_agents, n_agents)")
        np.fill_diagonal(peer_allocation_array, 0.0)
        peer_donors = [
            np.flatnonzero(peer_allocation_array[agent_id] > 0.0).astype(int).tolist()
            for agent_id in range(n_agents)
        ]
        if any(peer_donors):
            peer_probe_tensor = _as_agent_conditioned_probe_sequences(
                peer_probe_sequences,
                n_agents,
            ).to(device)
            peer_target_values = recurrent_critic_probe_targets(
                agents,
                peer_probe_tensor,
                device,
            )

    for agent_id, (net, optimizer) in enumerate(zip(agents, optimizers)):
        raw_agent_adv = advantages[:, agent_id]
        adv_std = raw_agent_adv.std(unbiased=False)
        if cfg.normalize_advantages:
            centered_adv = raw_agent_adv - raw_agent_adv.mean()
            if float(adv_std.item()) >= cfg.min_advantage_std:
                agent_adv = centered_adv / (adv_std + 1e-8)
            else:
                agent_adv = centered_adv
        else:
            agent_adv = raw_agent_adv
        agent_rewards = rollout.rewards[:, agent_id]
        reward_density = torch.mean(
            (torch.abs(agent_rewards) > cfg.reward_epsilon).float()
        )
        positive_reward_density = torch.mean(
            (agent_rewards > cfg.reward_epsilon).float()
        )
        negative_reward_density = torch.mean(
            (agent_rewards < -cfg.reward_epsilon).float()
        )
        skip_actor_update = (
            (
                cfg.skip_zero_reward_actor_update
                and bool(torch.all(torch.abs(agent_rewards) <= cfg.reward_epsilon).item())
            )
            or (
                cfg.skip_no_positive_reward_actor_update
                and bool(torch.all(agent_rewards <= cfg.reward_epsilon).item())
            )
        )
        agent_returns = returns[:, agent_id]
        agent_diagnostics[f"agent_{agent_id}/actor_update_skipped"] = float(skip_actor_update)
        agent_diagnostics[f"agent_{agent_id}/positive_reward_density"] = float(positive_reward_density.item())
        agent_diagnostics[f"agent_{agent_id}/advantage_std"] = float(adv_std.item())
        mean_abs_td_error[agent_id] = float(
            torch.mean(torch.abs(agent_returns - rollout.values[:, agent_id])).item()
        )

        for _ in range(cfg.ppo_epochs):
            for (
                obs_seq,
                actions_seq,
                action_mask_seq,
                old_logprob_seq,
                old_value_seq,
                done_seq,
                actor_h0,
                critic_h0,
                mb_starts,
            ) in recurrent_minibatches(
                rollout,
                agent_id,
                cfg.sequence_length,
                cfg.minibatch_chunks,
                rng,
            ):
                adv_seq = torch.stack(
                    [
                        agent_adv[s : s + cfg.sequence_length]
                        for s in mb_starts
                    ],
                    dim=1,
                )
                return_seq = torch.stack(
                    [
                        agent_returns[s : s + cfg.sequence_length]
                        for s in mb_starts
                    ],
                    dim=1,
                )

                obs_seq = obs_seq.to(device)
                actions_seq = actions_seq.to(device)
                action_mask_seq = action_mask_seq.to(device)
                old_logprob_seq = old_logprob_seq.to(device)
                old_value_seq = old_value_seq.to(device)
                done_seq = done_seq.to(device)
                actor_h0 = actor_h0.to(device)
                critic_h0 = critic_h0.to(device)
                adv_seq = adv_seq.to(device)
                return_seq = return_seq.to(device)

                new_logprob, entropy, value = net.evaluate_actions(
                    obs_seq,
                    actions_seq,
                    done_seq,
                    actor_h0,
                    critic_h0,
                    action_mask_seq,
                )
                log_ratio = new_logprob - old_logprob_seq
                ratio = log_ratio.exp()
                unclipped_policy_loss = -adv_seq * ratio
                clipped_policy_loss = -adv_seq * torch.clamp(
                    ratio,
                    1.0 - cfg.clip_coef,
                    1.0 + cfg.clip_coef,
                )
                policy_loss = torch.max(
                    unclipped_policy_loss,
                    clipped_policy_loss,
                ).mean()

                value_clipped = old_value_seq + torch.clamp(
                    value - old_value_seq,
                    -cfg.value_clip_coef,
                    cfg.value_clip_coef,
                )
                unclipped_value_loss = (value - return_seq).pow(2)
                clipped_value_loss = (value_clipped - return_seq).pow(2)
                value_loss = 0.5 * torch.max(
                    unclipped_value_loss,
                    clipped_value_loss,
                ).mean()
                entropy_loss = entropy.mean()

                peer_loss = torch.zeros((), dtype=torch.float32, device=device)
                peer_weight = 0.0
                if (
                    peer_allocation_array is not None
                    and peer_probe_tensor is not None
                    and peer_target_values is not None
                    and peer_donors[agent_id]
                ):
                    receiver_values = recurrent_critic_probe_values(
                        net,
                        peer_probe_tensor[agent_id],
                        device,
                    )
                    for donor_id in peer_donors[agent_id]:
                        weight = float(peer_allocation_array[agent_id, donor_id])
                        if weight <= 0.0:
                            continue
                        target_values = peer_target_values[donor_id].to(device)
                        peer_loss = peer_loss + weight * (
                            receiver_values - target_values
                        ).pow(2).mean()
                        peer_weight += weight

                actor_loss = policy_loss - cfg.entropy_coef * entropy_loss
                if skip_actor_update:
                    actor_loss = -cfg.entropy_coef * entropy_loss
                loss = (
                    actor_loss
                    + cfg.value_loss_coef * value_loss
                    + cfg.pgct_peer_loss_coef * peer_loss
                )

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(net.parameters(), cfg.max_grad_norm)
                optimizer.step()

                with torch.no_grad():
                    approx_kl = ((ratio - 1.0) - log_ratio).mean()
                    clip_fraction = (
                        (torch.abs(ratio - 1.0) > cfg.clip_coef)
                        .float()
                        .mean()
                    )
                metrics["policy_loss"].append(float(policy_loss.item()))
                metrics["value_loss"].append(float(value_loss.item()))
                metrics["entropy"].append(float(entropy_loss.item()))
                metrics["approx_kl"].append(float(approx_kl.item()))
                metrics["clip_fraction"].append(float(clip_fraction.item()))
                metrics["advantage_std"].append(float(adv_std.item()))
                metrics["reward_density"].append(float(reward_density.item()))
                metrics["positive_reward_density"].append(
                    float(positive_reward_density.item())
                )
                metrics["negative_reward_density"].append(
                    float(negative_reward_density.item())
                )
                metrics["actor_update_skipped"].append(float(skip_actor_update))
                metrics["pgct_peer_loss"].append(float(peer_loss.detach().item()))
                metrics["pgct_peer_weight"].append(float(peer_weight))

    graph_estimator.update_value_uncertainty(mean_abs_td_error)
    return {
        **agent_diagnostics,
        **{
            name: float(np.mean(values)) if values else float("nan")
            for name, values in metrics.items()
        },
    }


def critic_parameter_groups(net: RecurrentActorCritic) -> List[nn.Parameter]:
    return list(net.critic_parameters())


@torch.no_grad()
def apply_confidence_weighted_critic_consensus(
    agents: Sequence[RecurrentActorCritic],
    weights: np.ndarray,
    tau: float,
    min_weight: float,
) -> int:
    if tau <= 0.0:
        return 0
    weights = np.asarray(weights, dtype=np.float32)
    n_agents = len(agents)
    if weights.shape != (n_agents, n_agents):
        raise ValueError("weights must have shape (n_agents, n_agents)")

    active_weights = weights.copy()
    active_weights[active_weights < float(min_weight)] = 0.0
    np.fill_diagonal(active_weights, 0.0)
    critic_snapshots = [
        [param.data.clone() for param in critic_parameter_groups(agent)]
        for agent in agents
    ]

    for i, agent in enumerate(agents):
        params_i = critic_parameter_groups(agent)
        neighbors = np.flatnonzero(active_weights[i] > 0.0).tolist()
        if not neighbors:
            continue
        for param_idx, param in enumerate(params_i):
            delta = torch.zeros_like(param.data)
            source = critic_snapshots[i][param_idx]
            for j in neighbors:
                weight = float(active_weights[i, j])
                delta.add_(critic_snapshots[j][param_idx] - source, alpha=weight)
            param.data.add_(delta, alpha=float(tau))

    active = active_weights > 0
    return int(np.triu(active | active.T, k=1).sum())


@torch.no_grad()
def run_probe_episode(
    env_id: str,
    env_kwargs: Dict,
    observation_format: str,
    agents: Sequence[RecurrentActorCritic],
    obs_dim: int,
    action_dim: int,
    recurrent_hidden_dim: int,
    device: torch.device,
    seed: int,
    horizon: int,
    gamma: float,
    use_action_mask: bool,
    forced_step: Optional[int] = None,
    forced_agent: Optional[int] = None,
    forced_offset: int = 1,
) -> np.ndarray:
    env = make_env(env_id, env_kwargs, observation_format=observation_format)
    obs, _ = env.reset(seed=seed)
    n_agents = len(agents)
    actor_h = torch.zeros((n_agents, recurrent_hidden_dim), device=device)
    critic_h = torch.zeros_like(actor_h)
    discounted_returns = np.zeros((n_agents,), dtype=np.float64)
    discount = 1.0

    for step in range(horizon):
        flat_obs = flatten_multi_agent_obs(env, obs)
        obs_tensor = torch.as_tensor(flat_obs, dtype=torch.float32, device=device)
        action_masks = local_rware_action_masks(
            env,
            n_agents,
            action_dim,
            enabled=use_action_mask,
        )
        action_mask_tensor = torch.as_tensor(
            action_masks,
            dtype=torch.float32,
            device=device,
        )
        actions: List[int] = []
        for agent_id, net in enumerate(agents):
            action, _, _, next_actor_h, next_critic_h = net.act(
                obs_tensor[agent_id],
                actor_h[agent_id],
                critic_h[agent_id],
                deterministic=True,
                action_mask=action_mask_tensor[agent_id],
            )
            act_int = int(action.item())
            if forced_step == step and forced_agent == agent_id:
                act_int = int((act_int + forced_offset) % action_dim)
            actions.append(act_int)
            actor_h[agent_id] = next_actor_h.squeeze(0)
            critic_h[agent_id] = next_critic_h.squeeze(0)

        obs, rewards, done, truncated, _ = env.step(actions)
        discounted_returns += discount * np.asarray(rewards, dtype=np.float64)
        discount *= gamma
        if done or truncated:
            break
    env.close()
    del obs_dim
    return discounted_returns


def run_sparse_counterfactual_probes(
    cfg: TrainConfig,
    env_kwargs: Dict,
    agents: Sequence[RecurrentActorCritic],
    graph_estimator: TeamGraphEstimator,
    obs_dim: int,
    action_dim: int,
    recurrent_hidden_dim: int,
    device: torch.device,
    iteration: int,
    rng: np.random.Generator,
) -> Dict[str, float]:
    if cfg.graph_mode != "influence" or cfg.probe_episodes <= 0:
        return {"probe_delta": float("nan")}

    deltas = []
    for probe_id in range(cfg.probe_episodes):
        source_agent = int(rng.integers(0, len(agents)))
        forced_step = int(rng.integers(0, max(cfg.probe_horizon, 1)))
        forced_offset = int(rng.integers(1, action_dim))
        probe_seed = cfg.seed + 1_000_000 + iteration * 1000 + probe_id

        base_returns = run_probe_episode(
            cfg.env_id,
            env_kwargs,
            cfg.observation_format,
            agents,
            obs_dim,
            action_dim,
            recurrent_hidden_dim,
            device,
            seed=probe_seed,
            horizon=cfg.probe_horizon,
            gamma=cfg.gamma,
            use_action_mask=cfg.action_mask,
        )
        cf_returns = run_probe_episode(
            cfg.env_id,
            env_kwargs,
            cfg.observation_format,
            agents,
            obs_dim,
            action_dim,
            recurrent_hidden_dim,
            device,
            seed=probe_seed,
            horizon=cfg.probe_horizon,
            gamma=cfg.gamma,
            use_action_mask=cfg.action_mask,
            forced_step=forced_step,
            forced_agent=source_agent,
            forced_offset=forced_offset,
        )
        delta = cf_returns - base_returns
        graph_estimator.update_from_counterfactual(source_agent, delta)
        deltas.append(float(np.mean(np.abs(delta))))

    return {"probe_delta": float(np.mean(deltas)) if deltas else float("nan")}


def _masked_finite_mean(values: np.ndarray, mask: np.ndarray) -> float:
    selected = np.asarray(values, dtype=np.float64)[np.asarray(mask, dtype=bool)]
    selected = selected[np.isfinite(selected)]
    if selected.size == 0:
        return float("nan")
    return float(np.mean(selected))


def oracle_pairwise_matrix_metrics(
    values: np.ndarray,
    true_clusters: Optional[Sequence[Sequence[int]]],
    n_agents: int,
    metric_name: str,
    gap_name: Optional[str] = None,
    gap_direction: str = "cross_minus_same",
    mask: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    if not true_clusters:
        return {}
    values = np.asarray(values, dtype=np.float64)
    if values.shape != (n_agents, n_agents):
        raise ValueError("values must have shape (n_agents, n_agents)")

    true_labels = labels_from_clusters(true_clusters, n_agents)
    upper_mask = np.triu(np.ones((n_agents, n_agents), dtype=bool), k=1)
    if mask is not None:
        upper_mask &= np.asarray(mask, dtype=bool)
    same_mask = upper_mask & (true_labels[:, None] == true_labels[None, :])
    cross_mask = upper_mask & (true_labels[:, None] != true_labels[None, :])

    same_value = _masked_finite_mean(values, same_mask)
    cross_value = _masked_finite_mean(values, cross_mask)
    metrics = {
        f"same_team_{metric_name}": same_value,
        f"cross_team_{metric_name}": cross_value,
    }
    if gap_name:
        if math.isnan(same_value) or math.isnan(cross_value):
            gap = float("nan")
        elif gap_direction == "same_minus_cross":
            gap = float(same_value - cross_value)
        elif gap_direction == "cross_minus_same":
            gap = float(cross_value - same_value)
        else:
            raise ValueError(
                "gap_direction must be 'same_minus_cross' or 'cross_minus_same'"
            )
        metrics[gap_name] = gap
    return metrics


def pairwise_matrix_metrics(
    values: np.ndarray,
    metric_name: str,
    true_clusters: Optional[Sequence[Sequence[int]]] = None,
) -> Dict[str, float]:
    values = np.asarray(values, dtype=np.float64)
    n_agents = int(values.shape[0])
    true_labels = (
        labels_from_clusters(true_clusters, n_agents) if true_clusters else None
    )
    metrics: Dict[str, float] = {}
    for i in range(n_agents):
        for j in range(i + 1, n_agents):
            value = float(values[i, j])
            if math.isnan(value):
                continue
            metrics[f"pair_{i}_{j}_{metric_name}"] = value
            if true_labels is not None:
                metrics[f"pair_{i}_{j}_same_team"] = float(
                    true_labels[i] == true_labels[j]
                )
    return metrics


def mask_gate_by_current_neighbors(
    gate: np.ndarray,
    current_neighbor_adjacency: np.ndarray,
) -> np.ndarray:
    gate = np.asarray(gate, dtype=np.float32)
    current_neighbor_adjacency = np.asarray(current_neighbor_adjacency, dtype=np.float32)
    if gate.shape != current_neighbor_adjacency.shape:
        raise ValueError("gate and current_neighbor_adjacency must have the same shape")
    masked = gate * (current_neighbor_adjacency > 0.0).astype(np.float32)
    np.fill_diagonal(masked, 0.0)
    return masked.astype(np.float32)


def run_policy_similarity_probe_round(
    cfg: TrainConfig,
    env: gym.Env,
    agents: Sequence[RecurrentActorCritic],
    graph_estimator: TeamGraphEstimator,
    rollout: Rollout,
    device: torch.device,
    rng: np.random.Generator,
) -> Tuple[Dict[str, float], torch.Tensor]:
    n_agents = len(agents)
    neighbor_adj = communication_adjacency(env, n_agents, mode=cfg.comm_graph_mode)
    probe_sequences, probe_meta = sample_policy_probe_sequences(cfg, env, rollout, rng)
    distance, similarity, policy_information = policy_distance_similarity_matrices(
        agents,
        probe_sequences,
        device=device,
        temperature=cfg.policy_similarity_temperature,
        return_information=True,
    )
    graph_estimator.update_policy_information(policy_information)
    updates = graph_estimator.update_from_policy_similarity(similarity, neighbor_adj)
    distance_updates = graph_estimator.update_from_policy_distance(
        distance,
        neighbor_adj,
        beta=cfg.pgct_distance_ema_beta,
    )
    weights = graph_estimator.weight_matrix()
    smoothed_distance = graph_estimator.distance_ema
    affinity = graph_estimator.pgct_affinity_matrix(cfg.pgct_distance_temperature)

    upper_mask = np.triu(np.ones((n_agents, n_agents), dtype=bool), k=1)
    upper_neighbor = (~np.eye(n_agents, dtype=bool)) & (neighbor_adj > 0.0)
    upper_active = upper_mask & (weights >= cfg.edge_threshold)
    neighbor_scores = similarity[upper_neighbor]
    neighbor_distances = distance[upper_neighbor]
    neighbor_smoothed = smoothed_distance[upper_neighbor]
    neighbor_affinity = affinity[upper_neighbor]
    active_weights = weights[upper_active]
    true_clusters = oracle_clusters(env)
    metrics = {
        "policy_similarity": float(np.mean(neighbor_scores))
        if neighbor_scores.size
        else float("nan"),
        "policy_distance": float(np.mean(neighbor_distances))
        if neighbor_distances.size
        else float("nan"),
        "smoothed_policy_distance": float(np.nanmean(neighbor_smoothed))
        if neighbor_smoothed.size and not np.all(np.isnan(neighbor_smoothed))
        else float("nan"),
        "pgct_affinity": float(np.mean(neighbor_affinity))
        if neighbor_affinity.size
        else float("nan"),
        "neighbor_arcs": float(np.sum(upper_neighbor)),
        "updated_edges": float(updates),
        "distance_updated_edges": float(distance_updates),
        "active_edges": float(np.sum(upper_active)),
        "active_weight": float(np.mean(active_weights))
        if active_weights.size
        else float("nan"),
        "probe_count": _canonical_probe_count(probe_sequences),
        "policy_information_mean": float(np.mean(policy_information)),
        "policy_information_min": float(np.min(policy_information)),
        "policy_information_ready_agents": float(
            np.sum(policy_information >= cfg.pgct_min_policy_information)
        ),
        "directed_probe_count_min": float(np.min(
            graph_estimator.distance_count[~np.eye(n_agents, dtype=bool)]
        )),
        "directed_probe_count_max": float(np.max(
            graph_estimator.distance_count[~np.eye(n_agents, dtype=bool)]
        )),
        **probe_meta,
    }
    for agent_id, information in enumerate(policy_information):
        metrics[f"agent_{agent_id}_policy_information"] = float(information)
    metrics.update(pairwise_matrix_metrics(distance, "distance", true_clusters))
    metrics.update(
        oracle_pairwise_matrix_metrics(
            distance,
            true_clusters,
            n_agents,
            metric_name="distance",
            gap_name="distance_gap",
            gap_direction="cross_minus_same",
        )
    )
    metrics.update(
        oracle_pairwise_matrix_metrics(
            distance,
            true_clusters,
            n_agents,
            metric_name="neighbor_distance",
            gap_name="neighbor_distance_gap",
            gap_direction="cross_minus_same",
            mask=neighbor_adj > 0.0,
        )
    )
    metrics.update(
        oracle_pairwise_matrix_metrics(
            smoothed_distance,
            true_clusters,
            n_agents,
            metric_name="smoothed_distance",
            gap_name="smoothed_distance_gap",
            gap_direction="cross_minus_same",
        )
    )
    metrics.update(
        oracle_pairwise_matrix_metrics(
            affinity,
            true_clusters,
            n_agents,
            metric_name="affinity",
            gap_name="affinity_gap",
            gap_direction="same_minus_cross",
        )
    )
    return metrics, probe_sequences


def run_policy_similarity_probes(
    cfg: TrainConfig,
    env: gym.Env,
    agents: Sequence[RecurrentActorCritic],
    graph_estimator: TeamGraphEstimator,
    rollout: Rollout,
    device: torch.device,
    rng: np.random.Generator,
) -> Dict[str, float]:
    metrics, _ = run_policy_similarity_probe_round(
        cfg,
        env,
        agents,
        graph_estimator,
        rollout,
        device,
        rng,
    )
    return metrics


@torch.no_grad()
def evaluate(
    env_id: str,
    env_kwargs: Dict,
    observation_format: str,
    agents: Sequence[RecurrentActorCritic],
    episodes: int,
    horizon: int,
    gamma: float,
    obs_dim: int,
    recurrent_hidden_dim: int,
    device: torch.device,
    seed: int,
    render_episodes: int = 0,
    render_mode: str = "human",
    collect_video: bool = False,
    use_action_mask: bool = True,
    communication_gate: Optional[np.ndarray] = None,
) -> Tuple[Dict[str, float], Optional[np.ndarray]]:
    if episodes <= 0:
        return {}, None
    del obs_dim
    action_dim = int(agents[0].action_dim)
    n_agents = len(agents)
    returns = []
    discounted_returns = []
    task_returns = []
    shaping_returns = []
    lengths = []
    delivery_counts = []
    return_counts = []
    pickup_counts = []
    requested_pickup_counts = []
    wrong_return_counts = []
    requested_shelf_seen = []
    carrying_at_end = []
    invalid_toggle_rates = []
    failed_forward_rates = []
    blocking_agent_rates = []
    deadlock_agent_rates = []
    steps_to_seen = []
    steps_to_pickup = []
    steps_pickup_to_goal = []
    steps_delivery_to_return = []
    max_action_probs = []
    actor_hidden_norms = []
    critic_hidden_norms = []
    action_counts = np.zeros((action_dim,), dtype=np.float64)
    action_prob_sums = np.zeros((action_dim,), dtype=np.float64)
    action_prob_count = 0.0
    valid_pickup_opportunity_count = 0.0
    valid_return_opportunity_count = 0.0
    return_phase_step_count = 0.0
    toggle_when_pickup_valid_count = 0.0
    toggle_when_return_valid_count = 0.0
    toggle_policy_prob_when_pickup_valid_sum = 0.0
    toggle_policy_prob_when_return_valid_sum = 0.0
    toggle_count = 0.0
    requested_pickup_count = 0.0
    return_success_count = 0.0
    position_change_count = 0.0
    forward_failure_count = 0.0
    agent_step_count = 0.0
    wall_state_count = 0.0
    wall_action_counts = np.zeros((action_dim,), dtype=np.float64)
    wall_action_prob_sums = np.zeros((action_dim,), dtype=np.float64)
    wall_action_prob_count = 0.0
    per_agent_step_counts = np.zeros((n_agents,), dtype=np.float64)
    per_agent_position_change_counts = np.zeros((n_agents,), dtype=np.float64)
    per_agent_pickup_counts = np.zeros((n_agents,), dtype=np.float64)
    per_agent_delivery_counts = np.zeros((n_agents,), dtype=np.float64)
    per_agent_return_counts = np.zeros((n_agents,), dtype=np.float64)
    per_agent_failed_forward_counts = np.zeros((n_agents,), dtype=np.float64)
    per_agent_blocking_counts = np.zeros((n_agents,), dtype=np.float64)
    per_agent_deadlock_counts = np.zeros((n_agents,), dtype=np.float64)
    return_distances: List[float] = []
    video_frames: List[np.ndarray] = []
    for episode in range(episodes):
        should_render = episode < render_episodes
        should_collect_video = collect_video and episode == 0
        episode_render_mode = "rgb_array" if should_collect_video else render_mode
        eval_env_kwargs = (
            make_render_env_kwargs(env_kwargs, episode_render_mode)
            if should_render or should_collect_video
            else env_kwargs
        )
        env = make_env(
            env_id,
            eval_env_kwargs,
            observation_format=observation_format,
        )
        obs, _ = env.reset(seed=seed + episode)
        set_env_training_edges(env, np.zeros((n_agents, n_agents), dtype=np.float32)
                               if communication_gate is None else communication_gate)
        actor_h = torch.zeros((n_agents, recurrent_hidden_dim), device=device)
        critic_h = torch.zeros_like(actor_h)
        ep_return = np.zeros((n_agents,), dtype=np.float64)
        ep_discounted_return = np.zeros((n_agents,), dtype=np.float64)
        ep_task_return = np.zeros((n_agents,), dtype=np.float64)
        ep_shaping_return = np.zeros((n_agents,), dtype=np.float64)
        ep_deliveries = 0.0
        ep_returns_home = 0.0
        ep_pickups = 0.0
        ep_requested_pickups = 0.0
        ep_invalid_toggles = 0.0
        ep_wrong_returns = 0.0
        ep_failed_forwards = 0.0
        ep_blocking_agents = 0.0
        ep_deadlocked_agents = 0.0
        seen_requested_shelf = False
        first_seen_step: Optional[int] = None
        first_pickup_step: Optional[int] = None
        first_delivery_step: Optional[int] = None
        first_return_step: Optional[int] = None
        discount = 1.0
        length = 0
        if should_render or should_collect_video:
            frame = env.render()
            if should_collect_video and frame is not None:
                video_frames.append(np.asarray(frame, dtype=np.uint8))
        for step in range(horizon):
            visible = rware_requested_shelf_visibility(env, n_agents)
            if np.any(visible):
                seen_requested_shelf = True
                if first_seen_step is None:
                    first_seen_step = length
            pickup_opportunities = rware_requested_pickup_opportunities(env, n_agents)
            return_opportunities, return_phase, step_return_distances = (
                rware_return_diagnostics(env, n_agents)
            )
            pre_positions = rware_agent_positions(env, n_agents)
            return_distances.extend(step_return_distances)

            flat_obs = flatten_multi_agent_obs(env, obs)
            obs_tensor = torch.as_tensor(flat_obs, dtype=torch.float32, device=device)
            action_masks = local_rware_action_masks(
                env,
                n_agents,
                action_dim,
                enabled=use_action_mask,
            )
            diagnostic_action_masks = (
                action_masks
                if use_action_mask
                else local_rware_action_masks(
                    env,
                    n_agents,
                    action_dim,
                    enabled=True,
                )
            )
            action_mask_tensor = torch.as_tensor(
                action_masks,
                dtype=torch.float32,
                device=device,
            )
            blocked_forward_states = np.zeros((n_agents,), dtype=bool)
            if action_dim > int(Action.FORWARD.value):
                blocked_forward_states = (
                    diagnostic_action_masks[:, int(Action.FORWARD.value)] <= 0.0
                )
            valid_pickup_opportunity_count += float(np.sum(pickup_opportunities))
            valid_return_opportunity_count += float(np.sum(return_opportunities))
            return_phase_step_count += float(np.sum(return_phase))
            wall_state_count += float(np.sum(blocked_forward_states))
            agent_step_count += float(n_agents)
            actions = []
            for agent_id, net in enumerate(agents):
                action_probs = net.action_probabilities(
                    obs_tensor[agent_id],
                    actor_h[agent_id],
                    action_mask=action_mask_tensor[agent_id],
                )
                action_probs_np = action_probs.squeeze(0).detach().cpu().numpy()
                action_prob_sums += action_probs_np
                action_prob_count += 1.0
                if blocked_forward_states[agent_id]:
                    wall_action_prob_sums += action_probs_np
                    wall_action_prob_count += 1.0
                max_action_probs.append(float(np.max(action_probs_np)))
                if (
                    action_dim > int(Action.TOGGLE_LOAD.value)
                    and pickup_opportunities[agent_id]
                ):
                    toggle_policy_prob_when_pickup_valid_sum += float(
                        action_probs_np[int(Action.TOGGLE_LOAD.value)]
                    )
                if (
                    action_dim > int(Action.TOGGLE_LOAD.value)
                    and return_opportunities[agent_id]
                ):
                    toggle_policy_prob_when_return_valid_sum += float(
                        action_probs_np[int(Action.TOGGLE_LOAD.value)]
                    )
                action, _, _, next_actor_h, next_critic_h = net.act(
                    obs_tensor[agent_id],
                    actor_h[agent_id],
                    critic_h[agent_id],
                    deterministic=True,
                    action_mask=action_mask_tensor[agent_id],
                )
                actions.append(int(action.item()))
                actor_h[agent_id] = next_actor_h.squeeze(0)
                critic_h[agent_id] = next_critic_h.squeeze(0)
                actor_hidden_norms.append(float(actor_h[agent_id].norm().item()))
                critic_hidden_norms.append(float(critic_h[agent_id].norm().item()))
            actions_array = np.asarray(actions, dtype=np.int64)
            if action_dim > int(Action.TOGGLE_LOAD.value):
                toggle_actions = actions_array == int(Action.TOGGLE_LOAD.value)
                toggle_count += float(np.sum(toggle_actions))
                toggle_when_pickup_valid_count += float(
                    np.sum(toggle_actions & pickup_opportunities)
                )
                toggle_when_return_valid_count += float(
                    np.sum(toggle_actions & return_opportunities)
                )
            for action in actions_array[blocked_forward_states]:
                if 0 <= int(action) < action_dim:
                    wall_action_counts[int(action)] += 1.0
            for action in actions:
                if 0 <= action < action_dim:
                    action_counts[action] += 1.0
            masked_failed_forwards = 0.0
            if action_dim > int(Action.FORWARD.value):
                masked_failed_forwards = float(
                    sum(
                        action == int(Action.FORWARD.value)
                        and diagnostic_action_masks[
                            agent_id,
                            int(Action.FORWARD.value),
                        ]
                        <= 0.0
                        for agent_id, action in enumerate(actions)
                    )
                )
            obs, rewards, done, truncated, info = env.step(actions)
            step_failed_forwards = float(
                info.get("failed_forwards", masked_failed_forwards)
            )
            step_blocking_agents = float(info.get("blocking_agents", 0.0))
            step_deadlocked_agents = float(info.get("deadlocked_agents", 0.0))
            ep_failed_forwards += step_failed_forwards
            ep_blocking_agents += step_blocking_agents
            ep_deadlocked_agents += step_deadlocked_agents
            forward_failure_count += step_failed_forwards
            post_positions = rware_agent_positions(env, n_agents)
            moved_agents = np.asarray(
                [
                    before != after
                    for before, after in zip(pre_positions, post_positions)
                ],
                dtype=np.float64,
            )
            position_change_count += float(np.sum(moved_agents))
            per_agent_step_counts += 1.0
            per_agent_position_change_counts += moved_agents
            add_agent_event_counts(
                info.get("pickup_events", []),
                per_agent_pickup_counts,
                "agent_id",
            )
            add_agent_event_counts(
                info.get("delivery_events", []),
                per_agent_delivery_counts,
                "carrier_id",
            )
            add_agent_event_counts(
                info.get("return_events", []),
                per_agent_return_counts,
                "carrier_id",
            )
            add_agent_event_counts(
                info.get("failed_forward_events", []),
                per_agent_failed_forward_counts,
                "agent_id",
            )
            add_agent_event_counts(
                info.get("blocking_agent_events", []),
                per_agent_blocking_counts,
                "agent_id",
            )
            add_agent_event_counts(
                info.get("deadlock_events", []),
                per_agent_deadlock_counts,
                "agent_id",
            )
            rewards_array = np.asarray(rewards, dtype=np.float64)
            shaping_array = np.asarray(
                info.get("reward_shaping", np.zeros((n_agents,), dtype=np.float64)),
                dtype=np.float64,
            )
            if shaping_array.shape != rewards_array.shape:
                shaping_array = np.zeros_like(rewards_array)
            task_rewards_array = rewards_array - shaping_array
            ep_return += rewards_array
            ep_discounted_return += discount * rewards_array
            ep_task_return += task_rewards_array
            ep_shaping_return += shaping_array
            deliveries = float(info.get("deliveries", info.get("collections", 0)))
            returns_home = float(info.get("returns", 0))
            pickups = float(info.get("pickups", 0))
            pickup_events = info.get("pickup_events", [])
            requested_pickups = (
                float(
                    sum(
                        1
                        for event in pickup_events
                        if event.get("requested", False)
                    )
                )
                if isinstance(pickup_events, list)
                else pickups
            )
            requested_pickup_count += requested_pickups
            return_success_count += returns_home
            invalid_toggles = float(info.get("invalid_toggles", 0))
            wrong_returns = float(info.get("wrong_returns", 0))
            ep_deliveries += deliveries
            ep_returns_home += returns_home
            ep_pickups += pickups
            ep_requested_pickups += requested_pickups
            ep_invalid_toggles += invalid_toggles
            ep_wrong_returns += wrong_returns
            discount *= gamma
            length = step + 1
            if pickups > 0.0 and first_pickup_step is None:
                first_pickup_step = length
            if deliveries > 0.0 and first_delivery_step is None:
                first_delivery_step = length
            if returns_home > 0.0 and first_return_step is None:
                first_return_step = length
            if should_render or should_collect_video:
                frame = env.render()
                if should_collect_video and frame is not None:
                    video_frames.append(np.asarray(frame, dtype=np.uint8))
            if done or truncated:
                break
        episode_action_count = max(float(length * n_agents), 1.0)
        carrying = any(
            getattr(agent, "carrying_shelf", None) is not None
            for agent in getattr(env.unwrapped, "agents", [])
        )
        env.close()
        returns.append(float(np.mean(ep_return)))
        discounted_returns.append(float(np.mean(ep_discounted_return)))
        task_returns.append(float(np.mean(ep_task_return)))
        shaping_returns.append(float(np.mean(ep_shaping_return)))
        lengths.append(float(length))
        delivery_counts.append(float(ep_deliveries))
        return_counts.append(float(ep_returns_home))
        pickup_counts.append(float(ep_pickups))
        requested_pickup_counts.append(float(ep_requested_pickups))
        wrong_return_counts.append(float(ep_wrong_returns))
        requested_shelf_seen.append(float(seen_requested_shelf))
        carrying_at_end.append(float(carrying))
        invalid_toggle_rates.append(float(ep_invalid_toggles / episode_action_count))
        failed_forward_rates.append(float(ep_failed_forwards / episode_action_count))
        blocking_agent_rates.append(float(ep_blocking_agents / episode_action_count))
        deadlock_agent_rates.append(float(ep_deadlocked_agents / episode_action_count))
        steps_to_seen.append(
            float(first_seen_step) if first_seen_step is not None else float("nan")
        )
        steps_to_pickup.append(
            float(first_pickup_step) if first_pickup_step is not None else float("nan")
        )
        steps_pickup_to_goal.append(
            optional_step_delta(first_delivery_step, first_pickup_step)
        )
        steps_delivery_to_return.append(
            optional_step_delta(first_return_step, first_delivery_step)
        )
    video = None
    if video_frames:
        # wandb.Video expects time-major channels-first video.
        video = np.stack(video_frames, axis=0).transpose(0, 3, 1, 2)
    delivery_successes = np.asarray(delivery_counts) > 0.0
    return_successes = np.asarray(return_counts) > 0.0
    action_total = max(float(np.sum(action_counts)), 1.0)
    action_metrics = {
        f"eval_action_{action.name.lower()}_rate": float(
            action_counts[int(action.value)] / action_total
        )
        for action in Action
        if int(action.value) < action_dim
    }
    action_probability_metrics = {
        f"eval_action_{action.name.lower()}_probability": float(
            action_prob_sums[int(action.value)] / max(action_prob_count, 1.0)
        )
        for action in Action
        if int(action.value) < action_dim
    }
    wall_action_metrics = {
        f"eval_wall_action_{action.name.lower()}_rate": (
            float(wall_action_counts[int(action.value)] / wall_state_count)
            if wall_state_count > 0.0
            else float("nan")
        )
        for action in Action
        if int(action.value) < action_dim
    }
    wall_action_probability_metrics = {
        f"eval_wall_action_{action.name.lower()}_probability": (
            float(wall_action_prob_sums[int(action.value)] / wall_action_prob_count)
            if wall_action_prob_count > 0.0
            else float("nan")
        )
        for action in Action
        if int(action.value) < action_dim
    }
    agent_step_denominator = max(agent_step_count, 1.0)
    pickup_opportunity_denominator = max(valid_pickup_opportunity_count, 1.0)
    return_opportunity_denominator = max(valid_return_opportunity_count, 1.0)
    return {
        "eval_return": float(np.mean(returns)),
        "eval_discounted_return": float(np.mean(discounted_returns)),
        "eval_task_return": float(np.mean(task_returns)),
        "eval_shaping_return": float(np.mean(shaping_returns)),
        "eval_length": float(np.mean(lengths)),
        "eval_deliveries": float(np.mean(delivery_counts)),
        "eval_mean_deliveries": float(np.mean(delivery_counts)),
        "eval_returns_home": float(np.mean(return_counts)),
        "eval_events": float(np.mean(delivery_counts) + np.mean(return_counts)),
        "eval_requested_shelf_seen_rate": float(np.mean(requested_shelf_seen)),
        "eval_pickup_success_rate": float(np.mean(np.asarray(pickup_counts) > 0.0)),
        "eval_requested_shelf_pickups": float(np.mean(requested_pickup_counts)),
        "eval_requested_shelf_pickup_rate": float(
            np.mean(np.asarray(requested_pickup_counts) > 0.0)
        ),
        "eval_delivery_success_rate": float(np.mean(delivery_successes)),
        "eval_return_success_rate": float(np.mean(return_successes)),
        "eval_return_success_given_delivery": float(
            np.sum(delivery_successes & return_successes)
            / max(np.sum(delivery_successes), 1)
        ),
        "eval_full_cycle_success_rate": float(
            np.mean(delivery_successes & return_successes)
        ),
        "eval_delivered_but_not_returned_rate": float(
            np.mean(np.asarray(delivery_counts) > np.asarray(return_counts))
        ),
        "eval_wrong_returns": float(np.mean(wrong_return_counts)),
        "eval_invalid_toggle_rate": float(np.mean(invalid_toggle_rates)),
        "eval_failed_forward_rate": float(np.mean(failed_forward_rates)),
        "eval_blocking_agent_rate": float(np.mean(blocking_agent_rates)),
        "eval_deadlock_agent_rate": float(np.mean(deadlock_agent_rates)),
        "eval_carrying_at_episode_end": float(np.mean(carrying_at_end)),
        "eval_mean_steps_to_first_requested_shelf": finite_mean(steps_to_seen),
        "eval_mean_steps_to_pickup": finite_mean(steps_to_pickup),
        "eval_mean_steps_pickup_to_goal": finite_mean(steps_pickup_to_goal),
        "eval_mean_steps_delivery_to_return": finite_mean(steps_delivery_to_return),
        "eval_max_action_probability": float(np.mean(max_action_probs))
        if max_action_probs
        else float("nan"),
        "eval_valid_pickup_opportunity_count": float(
            valid_pickup_opportunity_count
        ),
        "eval_valid_pickup_opportunity_rate": float(
            valid_pickup_opportunity_count / agent_step_denominator
        ),
        "eval_valid_return_opportunity_count": float(
            valid_return_opportunity_count
        ),
        "eval_valid_return_opportunity_rate": float(
            valid_return_opportunity_count / agent_step_denominator
        ),
        "eval_return_phase_step_count": float(return_phase_step_count),
        "eval_return_phase_step_rate": float(
            return_phase_step_count / agent_step_denominator
        ),
        "eval_toggle_count": float(toggle_count),
        "eval_toggle_probability_when_pickup_valid": float(
            toggle_when_pickup_valid_count / pickup_opportunity_denominator
        ),
        "eval_toggle_probability_when_return_valid": float(
            toggle_when_return_valid_count / return_opportunity_denominator
        ),
        "eval_toggle_policy_probability_when_pickup_valid": float(
            toggle_policy_prob_when_pickup_valid_sum / pickup_opportunity_denominator
        ),
        "eval_toggle_policy_probability_when_return_valid": float(
            toggle_policy_prob_when_return_valid_sum / return_opportunity_denominator
        ),
        "eval_pickup_conversion_rate": float(
            requested_pickup_count / pickup_opportunity_denominator
        ),
        "eval_return_conversion_rate": float(
            return_success_count / return_opportunity_denominator
        ),
        "eval_position_change_rate": float(
            position_change_count / agent_step_denominator
        ),
        "eval_forward_failure_count": float(forward_failure_count),
        "eval_wall_state_count": float(wall_state_count),
        "eval_wall_state_rate": float(wall_state_count / agent_step_denominator),
        "eval_mean_return_home_distance": finite_mean(return_distances),
        "eval_actor_hidden_norm": float(np.mean(actor_hidden_norms))
        if actor_hidden_norms
        else float("nan"),
        "eval_critic_hidden_norm": float(np.mean(critic_hidden_norms))
        if critic_hidden_norms
        else float("nan"),
        **action_probability_metrics,
        **wall_action_metrics,
        **wall_action_probability_metrics,
        **per_agent_metrics(
            "eval_",
            n_agents,
            per_agent_step_counts,
            per_agent_position_change_counts,
            per_agent_pickup_counts,
            per_agent_delivery_counts,
            per_agent_return_counts,
            per_agent_failed_forward_counts,
            per_agent_blocking_counts,
            per_agent_deadlock_counts,
        ),
        **action_metrics,
    }, video


def save_checkpoint(
    path: Path,
    cfg: TrainConfig,
    agents: Sequence[RecurrentActorCritic],
    optimizers: Sequence[torch.optim.Optimizer],
    graph_estimator: TeamGraphEstimator,
    iteration: int,
    env_steps: int,
    env_kwargs: Optional[Dict[str, Any]] = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "config": asdict(cfg),
            "effective_env_kwargs": dict(env_kwargs or {}),
            "n_agents": len(agents),
            "iteration": iteration,
            "env_steps": env_steps,
            "agents": [agent.state_dict() for agent in agents],
            "optimizers": [optimizer.state_dict() for optimizer in optimizers],
            "graph": graph_estimator.state_dict(),
        },
        path,
    )


def format_clusters(clusters: Sequence[Sequence[int]]) -> str:
    return "[" + ", ".join(str(list(cluster)) for cluster in clusters) + "]"


def labels_from_clusters(
    clusters: Sequence[Sequence[int]],
    n_agents: int,
) -> np.ndarray:
    labels = -np.ones(n_agents, dtype=np.int32)
    for cluster_id, cluster in enumerate(clusters):
        for agent_id in cluster:
            labels[int(agent_id)] = int(cluster_id)
    missing = np.flatnonzero(labels < 0)
    for agent_id in missing:
        labels[int(agent_id)] = int(agent_id)
    return labels


def _comb2(value: int) -> float:
    value = int(value)
    return float(value * (value - 1) / 2)


def adjusted_rand_index(true_labels: np.ndarray, pred_labels: np.ndarray) -> float:
    true_labels = np.asarray(true_labels, dtype=np.int32)
    pred_labels = np.asarray(pred_labels, dtype=np.int32)
    n = int(true_labels.shape[0])
    total_pairs = _comb2(n)
    if total_pairs == 0.0:
        return 1.0

    true_values = np.unique(true_labels)
    pred_values = np.unique(pred_labels)
    contingency = np.zeros((len(true_values), len(pred_values)), dtype=np.int64)
    for i, t_value in enumerate(true_values):
        for j, p_value in enumerate(pred_values):
            contingency[i, j] = int(
                np.sum((true_labels == t_value) & (pred_labels == p_value))
            )

    sum_comb = float(np.sum([_comb2(v) for v in contingency.reshape(-1)]))
    true_comb = float(np.sum([_comb2(v) for v in contingency.sum(axis=1)]))
    pred_comb = float(np.sum([_comb2(v) for v in contingency.sum(axis=0)]))
    expected = true_comb * pred_comb / total_pairs
    max_index = 0.5 * (true_comb + pred_comb)
    denom = max_index - expected
    if abs(denom) < 1e-12:
        return 1.0 if abs(sum_comb - expected) < 1e-12 else 0.0
    return float((sum_comb - expected) / denom)


def normalized_mutual_info(true_labels: np.ndarray, pred_labels: np.ndarray) -> float:
    true_labels = np.asarray(true_labels, dtype=np.int32)
    pred_labels = np.asarray(pred_labels, dtype=np.int32)
    n = float(true_labels.shape[0])
    if n <= 0.0:
        return 1.0

    mi = 0.0
    h_true = 0.0
    h_pred = 0.0
    for true_value in np.unique(true_labels):
        p_true = float(np.sum(true_labels == true_value)) / n
        h_true -= p_true * math.log(max(p_true, 1e-12))
    for pred_value in np.unique(pred_labels):
        p_pred = float(np.sum(pred_labels == pred_value)) / n
        h_pred -= p_pred * math.log(max(p_pred, 1e-12))
    for true_value in np.unique(true_labels):
        true_mask = true_labels == true_value
        p_true = float(np.sum(true_mask)) / n
        for pred_value in np.unique(pred_labels):
            pred_mask = pred_labels == pred_value
            p_joint = float(np.sum(true_mask & pred_mask)) / n
            if p_joint <= 0.0:
                continue
            p_pred = float(np.sum(pred_mask)) / n
            mi += p_joint * math.log(p_joint / max(p_true * p_pred, 1e-12))
    denom = math.sqrt(max(h_true * h_pred, 0.0))
    if denom <= 1e-12:
        return 1.0 if np.array_equal(true_labels, pred_labels) else 0.0
    return float(mi / denom)


def team_discovery_metrics(
    predicted_clusters: Sequence[Sequence[int]],
    true_clusters: Optional[Sequence[Sequence[int]]],
    adjacency: np.ndarray,
    n_agents: int,
) -> Dict[str, float]:
    if not true_clusters:
        return {}
    pred_labels = labels_from_clusters(predicted_clusters, n_agents)
    true_labels = labels_from_clusters(true_clusters, n_agents)

    tp = fp = fn = tn = 0
    adjacency = np.asarray(adjacency)
    for i in range(n_agents):
        for j in range(i + 1, n_agents):
            pred_same = bool(adjacency[i, j] > 0.0)
            true_same = bool(true_labels[i] == true_labels[j])
            if pred_same and true_same:
                tp += 1
            elif pred_same and not true_same:
                fp += 1
            elif not pred_same and true_same:
                fn += 1
            else:
                tn += 1

    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2.0 * precision * recall / max(precision + recall, 1e-12)
    pred_edges = tp + fp
    return {
        "graph/ari": adjusted_rand_index(true_labels, pred_labels),
        "graph/nmi": normalized_mutual_info(true_labels, pred_labels),
        "graph/edge_precision": float(precision),
        "graph/edge_recall": float(recall),
        "graph/edge_f1": float(f1),
        "graph/wrong_edge_rate": float(fp / max(pred_edges, 1)),
        "graph/true_negative_edges": float(tn),
    }


def graph_metrics(
    graph_estimator: TeamGraphEstimator,
    clusters: Sequence[Sequence[int]],
    adjacency: np.ndarray,
    weights: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    cluster_sizes = [len(cluster) for cluster in clusters]
    if weights is None:
        weights = graph_estimator.weight_matrix()
    upper_mask = np.triu(np.ones_like(weights, dtype=bool), k=1)
    upper_weights = weights[(~np.eye(len(weights), dtype=bool)) & (weights > 0.0)]
    return {
        "graph/edges": float(np.triu((adjacency > 0) | (adjacency.T > 0), k=1).sum()),
        "graph/arcs": float(np.sum(adjacency > 0)),
        "graph/clusters": float(len(clusters)),
        "graph/max_cluster_size": float(max(cluster_sizes) if cluster_sizes else 0),
        "graph/value_uncertainty": float(np.mean(graph_estimator.value_uncertainty)),
        "graph/mean_weight": float(np.mean(upper_weights))
        if upper_weights.size
        else 0.0,
    }


def prefixed_metrics(prefix: str, metrics: Dict[str, float]) -> Dict[str, float]:
    return {
        f"{prefix}/{key}": value
        for key, value in metrics.items()
        if isinstance(value, (float, int)) and not math.isnan(float(value))
    }


def eval_payload_from_metrics(
    metrics: Dict[str, float],
    final: bool = False,
) -> Dict[str, float]:
    prefix = "eval/final_" if final else "eval/"
    payload: Dict[str, float] = {}
    for key, value in metrics.items():
        if not isinstance(value, (float, int)) or math.isnan(float(value)):
            continue
        metric_name = key[5:] if key.startswith("eval_") else key
        payload[f"{prefix}{metric_name}"] = float(value)
    return payload


def log_to_wandb(run, payload: Dict[str, object], step: int) -> None:
    if run is None:
        return
    run.log(payload, step=step)


def save_local_eval_video(video, path, fps):
    import imageio.v2 as imageio
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Preserve dimensions; yuv444p accepts odd sizes without silent resizing.
    with imageio.get_writer(str(path), fps=fps, codec="libx264", pixelformat="yuv444p",
                            macro_block_size=1) as writer:
        for frame in video.transpose(0, 2, 3, 1):
            writer.append_data(frame)
    print(f"video={path}")


def make_wandb_video(video: np.ndarray, fps: int):
    import wandb

    return wandb.Video(video, fps=fps, format="mp4")


def should_collect_periodic_eval_video(cfg: TrainConfig, eval_count: int) -> bool:
    return (
        cfg.track
        and cfg.wandb_log_eval_video
        and cfg.wandb_video_interval > 0
        and eval_count % cfg.wandb_video_interval == 0
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Recurrent IPPO + policy probes + weighted critic consensus"
    )
    parser.add_argument("--env-id", default="rware-multiteam-tiny-4ag-2teams-v0")
    parser.add_argument("--env-kwargs-json", default="{}")
    parser.add_argument(
        "--sensor-range",
        type=int,
        default=0,
        help=(
            "Override env sensor_range without passing JSON. "
            "0 leaves env defaults unchanged."
        ),
    )
    parser.add_argument(
        "--agent-count",
        type=int,
        default=0,
        help="Override env n_agents. 0 keeps the environment default.",
    )
    parser.add_argument(
        "--team-count",
        type=int,
        default=0,
        help="Override env n_teams. 0 keeps the environment default.",
    )
    parser.add_argument(
        "--no-auto-scale-request-queue",
        dest="auto_scale_request_queue",
        action="store_false",
        help=(
            "Keep explicit multiteam request_queue_size settings even when "
            "they are smaller than the selected agent/team count."
        ),
    )
    parser.add_argument(
        "--curriculum-stages",
        default="",
        help="Comma-separated agent-count curriculum, e.g. '4,8,12,20'.",
    )
    parser.add_argument(
        "--curriculum-stage",
        type=int,
        default=0,
        help="1-based stage index into --curriculum-stages. 0 disables stage lookup.",
    )
    parser.add_argument(
        "--init-checkpoint",
        default="",
        help=(
            "Initialize this run from a previous stage checkpoint. By default "
            "only actor parameters are copied and graph/optimizer state is reset."
        ),
    )
    parser.add_argument(
        "--transfer-components",
        default="actor",
        help=(
            "Comma-separated subset of actor,critic,all,none to copy "
            "from --init-checkpoint."
        ),
    )
    parser.add_argument("--total-timesteps", type=int, default=200_000)
    parser.add_argument("--rollout-steps", type=int, default=512)
    parser.add_argument("--sequence-length", type=int, default=32)
    parser.add_argument("--minibatch-chunks", type=int, default=8)
    parser.add_argument("--ppo-epochs", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--clip-coef", type=float, default=0.2)
    parser.add_argument("--value-clip-coef", type=float, default=0.2)
    parser.add_argument("--value-loss-coef", type=float, default=0.5)
    parser.add_argument("--entropy-coef", type=float, default=0.01)
    parser.add_argument(
        "--no-action-mask",
        dest="action_mask",
        action="store_false",
        help="Disable local RWARE valid-action masking during training/eval.",
    )
    parser.add_argument(
        "--no-normalize-advantages",
        dest="normalize_advantages",
        action="store_false",
        help="Disable per-agent advantage normalization.",
    )
    parser.add_argument(
        "--min-advantage-std",
        type=float,
        default=1e-6,
        help=(
            "Do not divide advantages by their standard deviation when the "
            "rollout std is below this threshold."
        ),
    )
    parser.add_argument(
        "--no-skip-zero-reward-actor-update",
        dest="skip_zero_reward_actor_update",
        action="store_false",
        help=(
            "Keep actor/entropy PPO updates even when an agent's rollout has "
            "no nonzero rewards."
        ),
    )
    parser.add_argument(
        "--skip-no-positive-reward-actor-update",
        action="store_true",
        help=(
            "Skip PPO policy-gradient updates when an agent's rollout has no "
            "positive rewards. Entropy regularization and critic updates still run."
        ),
    )
    parser.add_argument(
        "--reward-epsilon",
        type=float,
        default=1e-8,
        help="Absolute reward threshold used to detect zero-reward rollouts.",
    )
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
    parser.add_argument("--mlp-hidden-dim", type=int, default=128)
    parser.add_argument("--recurrent-hidden-dim", type=int, default=128)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument(
        "--observation-format",
        default="flat",
        choices=["flat", "semantic"],
        help=(
            "flat keeps the original RWARE flattened local observation; "
            "semantic returns ego features plus a CxSxS semantic local grid."
        ),
    )
    parser.add_argument(
        "--obs-encoder",
        default="mlp",
        choices=["mlp", "cnn"],
        help="Observation encoder used by the recurrent actor and critic.",
    )
    parser.add_argument(
        "--graph-mode",
        default="policy",
        choices=["policy", "probe", "influence", "none", "oracle", "wrong", "unrestricted"],
        help=(
            "policy/probe use common-probe policy similarity; influence keeps "
            "the legacy return-influence perturbation baseline; oracle is for "
            "debugging/baselines only."
        ),
    )
    parser.add_argument("--probe-interval", type=int, default=10)
    parser.add_argument("--policy-probe-batch-size", type=int, default=64)
    parser.add_argument(
        "--probe-source",
        default="mixed",
        choices=["rollout", "objective-rollout", "objective", "mixed"],
        help=(
            "Source for common policy probes. 'rollout' keeps the original "
            "random rollout probes; 'objective' uses environment-specific "
            "task-informed label-free probes; 'mixed' combines both."
        ),
    )
    parser.add_argument(
        "--objective-probe-fraction",
        type=float,
        default=0.5,
        help="Fraction of mixed probes drawn from the objective probe bank.",
    )
    parser.add_argument(
        "--policy-probe-team-conditioning",
        default="shared",
        choices=["shared", "agent-team"],
        help=(
            "shared compares every policy on the same canonical probe bank. "
            "agent-team uses ORACLE team IDs to build different inputs per team; "
            "use it only as a privileged diagnostic, not blind team discovery."
        ),
    )
    parser.add_argument("--policy-similarity-temperature", type=float, default=0.25)
    parser.add_argument(
        "--comm-graph-mode",
        default="physical",
        choices=["physical", "complete", "none"],
        help="Restrict policy comparisons to physical communication neighbors.",
    )
    parser.add_argument(
        "--probe-episodes",
        type=int,
        default=4,
        help="Legacy influence-mode perturbation episodes.",
    )
    parser.add_argument(
        "--probe-horizon",
        type=int,
        default=120,
        help="Legacy influence-mode perturbation horizon.",
    )
    parser.add_argument("--edge-threshold", type=float, default=0.2)
    parser.add_argument("--min-probe-count", type=int, default=2)
    parser.add_argument("--uncertainty-scale", type=float, default=1.0)
    parser.add_argument("--value-uncertainty-decay", type=float, default=0.95)
    parser.add_argument(
        "--graph-join-threshold",
        type=float,
        default=0.0,
        help="Stable edge join threshold. 0 uses --edge-threshold.",
    )
    parser.add_argument(
        "--graph-leave-threshold",
        type=float,
        default=0.0,
        help="Stable edge leave threshold. 0 uses the join threshold.",
    )
    parser.add_argument(
        "--graph-dwell-updates",
        type=int,
        default=1,
        help="Consecutive adjacency updates required before an edge changes state.",
    )
    parser.add_argument(
        "--peer-transfer-mode",
        default="consensus",
        choices=["consensus", "pgct", "none"],
        help=(
            "Peer transfer operator. consensus keeps the legacy critic parameter "
            "consensus baseline; pgct uses canonical critic function-space "
            "distillation; none disables peer transfer."
        ),
    )
    parser.add_argument(
        "--pgct-peer-loss-coef",
        type=float,
        default=0.05,
        help="lambda_peer for PGCT canonical critic distillation.",
    )
    parser.add_argument(
        "--pgct-probe-sequence-length",
        type=int,
        default=4,
        help="Lp, the recurrent length of each canonical probe sequence.",
    )
    parser.add_argument(
        "--pgct-distance-ema-beta",
        type=float,
        default=0.2,
        help="EMA coefficient beta for smoothed probe distances.",
    )
    parser.add_argument(
        "--pgct-distance-temperature",
        type=float,
        default=0.25,
        help="tau_D used to convert smoothed distance to affinity score.",
    )
    parser.add_argument(
        "--pgct-distance-threshold",
        type=float,
        default=0.2,
        help="tau_Q distance threshold for opening the PGCT transfer gate.",
    )
    parser.add_argument(
        "--pgct-gate-power",
        type=float,
        default=1.0,
        help="kappa exponent applied to the continuous PGCT gate.",
    )
    parser.add_argument(
        "--pgct-alpha-epsilon",
        type=float,
        default=1e-8,
        help="Receiver-side allocation denominator epsilon.",
    )
    parser.add_argument(
        "--pgct-warmup-updates",
        type=int,
        default=10,
        help="Disable PGCT gates before this many learning updates.",
    )
    parser.add_argument("--critic-consensus-tau", type=float, default=0.1)
    parser.add_argument("--consensus-interval", type=int, default=1)
    parser.add_argument("--save-dir", default="runs/recurrent_ippo_consensus")
    parser.add_argument("--exp-name", default="")
    parser.add_argument("--save-interval", type=int, default=50)
    parser.add_argument("--log-interval", type=int, default=1)
    parser.add_argument("--eval-episodes", type=int, default=5)
    parser.add_argument("--eval-horizon", type=int, default=500)
    parser.add_argument(
        "--eval-interval",
        type=int,
        default=0,
        help="Run eval every N training iterations. Final eval still runs once.",
    )
    parser.add_argument(
        "--render-eval",
        action="store_true",
        help="Render eval episodes with env.render().",
    )
    parser.add_argument("--eval-render-episodes", type=int, default=1)
    parser.add_argument(
        "--eval-render-mode",
        default="human",
        choices=["human", "rgb_array"],
    )
    parser.add_argument("--eval-render-fps", type=int, default=10)
    parser.add_argument(
        "--track",
        action="store_true",
        help="Log training and eval metrics to Weights & Biases.",
    )
    parser.add_argument("--wandb-project", default="rware-recurrent-ippo")
    parser.add_argument("--wandb-entity", default="")
    parser.add_argument("--wandb-group", default="")
    parser.add_argument("--wandb-name", default="")
    parser.add_argument(
        "--wandb-mode",
        default="online",
        choices=["online", "offline", "disabled"],
    )
    parser.add_argument("--wandb-tags", default="")
    parser.add_argument(
        "--wandb-log-eval-video",
        action="store_true",
        help="Log one rgb_array eval episode as a wandb.Video.",
    )
    parser.add_argument(
        "--wandb-video-interval",
        type=int,
        default=1,
        help=(
            "Log periodic eval video every N eval calls when "
            "--wandb-log-eval-video is set. 0 disables periodic eval videos; "
            "final eval video is still logged."
        ),
    )
    parser.set_defaults(
        action_mask=True,
        auto_scale_request_queue=True,
        normalize_advantages=True,
        skip_zero_reward_actor_update=True,
        skip_no_positive_reward_actor_update=False,
    )
    parser.add_argument(
        "--pgct-min-policy-information",
        type=float,
        default=0.0,
        help=(
            "Optional non-ex5 ablation; zero disables this heuristic. Minimum mean KL(policy || uniform) on common probes required for "
            "each endpoint before PGCT may open an edge."
        ),
    )
    parser.add_argument("--pgct-gate-style", choices=["threshold", "soft"], default="threshold")
    parser.add_argument("--fixed-canonical-probes", action="store_true",
                        help="Reuse one external shared objective probe bank throughout the run.")
    parser.add_argument("--no-communication", dest="communication_enabled", action="store_false",
                        help="Disable ALL graph probing, peer transfer, consensus and rendered links.")
    parser.add_argument("--save-eval-video", action="store_true",
                        help="Save evaluation MP4 under the run's videos directory without requiring W&B.")
    return parser


def resolve_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(value)


def main() -> None:
    args = build_arg_parser().parse_args()
    cfg = TrainConfig(**vars(args))
    enforce_communication_config(cfg)
    if cfg.graph_mode == "probe":
        cfg.graph_mode = "policy"
    curriculum_stages = parse_curriculum_stages(cfg.curriculum_stages)
    selected_agent_count = resolve_agent_count(
        cfg.agent_count,
        curriculum_stages,
        cfg.curriculum_stage,
    )
    transfer_components = parse_transfer_components(cfg.transfer_components)
    if cfg.rollout_steps % cfg.sequence_length != 0:
        raise ValueError("--rollout-steps must be divisible by --sequence-length")
    if cfg.minibatch_chunks < 1:
        raise ValueError("--minibatch-chunks must be positive")
    if not 0.0 <= cfg.objective_probe_fraction <= 1.0:
        raise ValueError("--objective-probe-fraction must be in [0, 1]")
    if cfg.graph_dwell_updates < 1:
        raise ValueError("--graph-dwell-updates must be positive")
    if cfg.sensor_range < 0:
        raise ValueError("--sensor-range must be non-negative")
    if cfg.wandb_video_interval < 0:
        raise ValueError("--wandb-video-interval must be non-negative")
    if cfg.min_advantage_std < 0.0:
        raise ValueError("--min-advantage-std must be non-negative")
    if cfg.reward_epsilon < 0.0:
        raise ValueError("--reward-epsilon must be non-negative")
    if cfg.pgct_peer_loss_coef < 0.0:
        raise ValueError("--pgct-peer-loss-coef must be non-negative")
    if cfg.pgct_probe_sequence_length < 1:
        raise ValueError("--pgct-probe-sequence-length must be positive")
    if not 0.0 < cfg.pgct_distance_ema_beta <= 1.0:
        raise ValueError("--pgct-distance-ema-beta must be in (0, 1]")
    if cfg.pgct_distance_temperature <= 0.0:
        raise ValueError("--pgct-distance-temperature must be positive")
    if cfg.pgct_distance_threshold < 0.0:
        raise ValueError("--pgct-distance-threshold must be non-negative")
    if cfg.pgct_gate_power <= 0.0:
        raise ValueError("--pgct-gate-power must be positive")
    if cfg.pgct_alpha_epsilon <= 0.0:
        raise ValueError("--pgct-alpha-epsilon must be positive")
    if cfg.pgct_warmup_updates < 0:
        raise ValueError("--pgct-warmup-updates must be non-negative")
    if cfg.pgct_min_policy_information < 0.0:
        raise ValueError("--pgct-min-policy-information must be non-negative")
    if cfg.obs_encoder == "cnn" and cfg.observation_format != "semantic":
        raise ValueError("--obs-encoder cnn requires --observation-format semantic")
    if transfer_components and not cfg.init_checkpoint:
        # This keeps the config explicit without forcing users to pass
        # --transfer-components none for ordinary from-scratch runs.
        transfer_components = set()

    set_global_seeds(cfg.seed)
    rng = np.random.default_rng(cfg.seed)
    probe_rng = np.random.default_rng(cfg.seed + 9173)
    device = resolve_device(cfg.device)
    env_kwargs = parse_env_kwargs(cfg.env_kwargs_json)
    if cfg.sensor_range > 0:
        env_kwargs["sensor_range"] = int(cfg.sensor_range)
    env_kwargs = apply_agent_overrides(
        cfg.env_id,
        env_kwargs,
        selected_agent_count,
        cfg.team_count,
        cfg.auto_scale_request_queue,
    )
    if "rware-multiteam" in cfg.env_id:
        env_kwargs.update(communication_enabled=cfg.communication_enabled,
                          communication_topology=cfg.comm_graph_mode)

    env = make_env(
        cfg.env_id,
        env_kwargs,
        observation_format=cfg.observation_format,
    )
    obs, _ = env.reset(seed=cfg.seed)
    n_agents = int(env.unwrapped.n_agents)
    obs_dims = get_flat_obs_dims(env)
    action_dims = get_discrete_action_dims(env)
    if len(set(obs_dims)) != 1:
        raise ValueError(f"Expected identical observation dims, got {obs_dims}")
    if len(set(action_dims)) != 1:
        raise ValueError(f"Expected identical action dims, got {action_dims}")
    obs_dim = int(obs_dims[0])
    action_dim = int(action_dims[0])

    encoder_kwargs: Dict[str, int] = {}
    if cfg.obs_encoder == "cnn":
        semantic_spec = get_semantic_observation_spec(env)
        encoder_kwargs = {
            "spatial_channels": int(semantic_spec.spatial_channels),
            "spatial_size": int(semantic_spec.spatial_size),
            "ego_dim": int(semantic_spec.ego_dim),
        }

    agents = [
        RecurrentActorCritic(
            obs_dim=obs_dim,
            action_dim=action_dim,
            mlp_hidden_dim=cfg.mlp_hidden_dim,
            recurrent_hidden_dim=cfg.recurrent_hidden_dim,
            encoder_type=cfg.obs_encoder,
            **encoder_kwargs,
        ).to(device)
        for _ in range(n_agents)
    ]
    transfer_report = None
    if cfg.init_checkpoint:
        transfer_report = transfer_checkpoint_to_agents(
            cfg.init_checkpoint,
            agents,
            transfer_components,
            device,
        )
    optimizers = [
        torch.optim.Adam(agent.parameters(), lr=cfg.learning_rate, eps=1e-5)
        for agent in agents
    ]
    graph_join_threshold = (
        cfg.edge_threshold
        if cfg.graph_join_threshold <= 0.0
        else cfg.graph_join_threshold
    )
    graph_leave_threshold = (
        graph_join_threshold
        if cfg.graph_leave_threshold <= 0.0
        else cfg.graph_leave_threshold
    )
    graph_estimator = TeamGraphEstimator(
        n_agents=n_agents,
        edge_threshold=cfg.edge_threshold,
        min_probe_count=cfg.min_probe_count,
        uncertainty_scale=cfg.uncertainty_scale,
        value_uncertainty_decay=cfg.value_uncertainty_decay,
        join_threshold=graph_join_threshold,
        leave_threshold=graph_leave_threshold,
        dwell_updates=cfg.graph_dwell_updates,
        min_policy_information=cfg.pgct_min_policy_information,
        gate_style=cfg.pgct_gate_style,
    )

    actor_h = torch.zeros((n_agents, cfg.recurrent_hidden_dim), device=device)
    critic_h = torch.zeros_like(actor_h)
    state = RunnerState(
        obs=obs,
        actor_h=actor_h,
        critic_h=critic_h,
        episode_returns=np.zeros((n_agents,), dtype=np.float64),
        episode_task_returns=np.zeros((n_agents,), dtype=np.float64),
        episode_shaping_returns=np.zeros((n_agents,), dtype=np.float64),
    )

    default_exp_name = cfg.env_id
    if selected_agent_count is not None:
        default_exp_name = f"{default_exp_name}_{n_agents}ag"
    if cfg.curriculum_stage > 0:
        default_exp_name = f"{default_exp_name}_stage{cfg.curriculum_stage}"
    exp_name = cfg.exp_name or f"{default_exp_name}_{int(time.time())}"
    cfg.exp_name = exp_name
    save_dir = Path(cfg.save_dir) / exp_name
    save_dir.mkdir(parents=True, exist_ok=True)
    wandb_run = init_wandb(cfg, save_dir)

    total_iterations = math.ceil(cfg.total_timesteps / cfg.rollout_steps)
    start_time = time.time()
    periodic_eval_count = 0
    print(
        f"env={cfg.env_id} agents={n_agents} obs_dim={obs_dim} "
        f"action_dim={action_dim} obs={cfg.observation_format}/{cfg.obs_encoder} "
        f"device={device}"
    )
    print(
        f"env_mechanics n_agents={env.unwrapped.n_agents} "
        f"num_observations={len(obs)} "
        f"action_space={env.action_space} "
        f"observation_space={env.observation_space} "
        f"request_queue_size={getattr(env.unwrapped, 'request_queue_size', 'unknown')} "
        f"request_queue_size_per_team="
        f"{getattr(env.unwrapped, 'request_queue_size_per_team', 'unknown')} "
        f"active_request_count={len(getattr(env.unwrapped, 'request_queue', []))}"
    )
    if curriculum_stages:
        print(
            f"curriculum_stages={curriculum_stages} "
            f"stage={cfg.curriculum_stage or 'manual'}"
        )
    if transfer_report is not None:
        print(
            "transfer="
            f"{transfer_report['components']} "
            f"source_agents={transfer_report['source_agents']} "
            f"target_agents={transfer_report['target_agents']} "
            f"copied_tensors={transfer_report['copied_tensors']} "
            f"skipped_tensors={transfer_report['skipped_tensors']}"
        )

    for iteration in range(1, total_iterations + 1):
        if cfg.graph_mode in {"oracle", "wrong", "unrestricted"}:
            graph_weights = baseline_weight_matrix(env, n_agents, cfg.graph_mode)
            adj = (graph_weights >= cfg.edge_threshold).astype(np.float32)
            clusters = clusters_from_adjacency(adj)
        elif cfg.graph_mode == "none":
            clusters = [[idx] for idx in range(n_agents)]
            adj = np.zeros((n_agents, n_agents), dtype=np.float32)
            graph_weights = np.zeros_like(adj)
        else:
            clusters = graph_estimator.clusters()
            adj = graph_estimator.adjacency()
            graph_weights = graph_estimator.weight_matrix()
        set_env_training_edges(env, communication_weights(cfg, env, graph_estimator, iteration))

        rollout, state, rollout_metrics = collect_rollout(
            env,
            agents,
            state,
            cfg.rollout_steps,
            obs_dim,
            action_dim,
            cfg.recurrent_hidden_dim,
            device,
            cfg.action_mask,
        )

        probe_metrics = {"probe_delta": float("nan")}
        peer_probe_sequences: Optional[torch.Tensor] = None
        if (
            cfg.graph_mode == "policy"
            and cfg.probe_interval > 0
            and iteration % cfg.probe_interval == 0
        ):
            probe_metrics, peer_probe_sequences = run_policy_similarity_probe_round(
                cfg,
                env,
                agents,
                graph_estimator,
                rollout,
                device,
                probe_rng,
            )
            probe_metrics["pgct_warmup_active"] = float(
                iteration < cfg.pgct_warmup_updates
            )
        elif (
            cfg.graph_mode == "influence"
            and cfg.probe_interval > 0
            and iteration % cfg.probe_interval == 0
        ):
            probe_metrics = run_sparse_counterfactual_probes(
                cfg,
                env_kwargs,
                agents,
                graph_estimator,
                obs_dim,
                action_dim,
                cfg.recurrent_hidden_dim,
                device,
                iteration,
                rng,
            )

        true_clusters = oracle_clusters(env)
        peer_allocation: Optional[np.ndarray] = None
        if cfg.peer_transfer_mode == "pgct":
            warmup_complete = iteration >= cfg.pgct_warmup_updates
            if cfg.graph_mode in {"oracle", "wrong", "unrestricted"}:
                pgct_gate = baseline_weight_matrix(env, n_agents, cfg.graph_mode)
            elif cfg.graph_mode == "none":
                pgct_gate = np.zeros((n_agents, n_agents), dtype=np.float32)
            else:
                pgct_gate = graph_estimator.pgct_gate_matrix(
                    warmup_complete=warmup_complete,
                    distance_threshold=cfg.pgct_distance_threshold,
                    distance_temperature=cfg.pgct_distance_temperature,
                    gate_power=cfg.pgct_gate_power,
                )
            unmasked_pgct_gate = pgct_gate.copy()
            if cfg.pgct_peer_loss_coef <= 0:
                pgct_gate.fill(0)
            if cfg.comm_graph_mode in {"physical", "none"}:
                current_neighbor_adj = communication_adjacency(
                    env,
                    n_agents,
                    mode=cfg.comm_graph_mode,
                )
                pgct_gate = mask_gate_by_current_neighbors(
                    pgct_gate,
                    current_neighbor_adj,
                )
                probe_metrics["pgct_stale_gate_arcs_suppressed"] = float(
                    (
                        np.sum(unmasked_pgct_gate > 0.0)
                        - np.sum(pgct_gate > 0.0)
                    )
                )
            allocation_denom = (
                np.sum(pgct_gate, axis=1, keepdims=True)
                + max(float(cfg.pgct_alpha_epsilon), 1e-8)
            )
            peer_allocation = (pgct_gate / allocation_denom).astype(np.float32)
            consensus_weights = pgct_gate
            log_clusters = clusters_from_adjacency((pgct_gate > 0.0).astype(np.float32))
            if np.any(peer_allocation > 0.0) and peer_probe_sequences is None:
                peer_probe_sequences, critic_probe_meta = sample_policy_probe_sequences(
                    cfg,
                    env,
                    rollout,
                    probe_rng,
                )
                probe_metrics.update(
                    {
                        "critic_probe_count": _canonical_probe_count(
                            peer_probe_sequences
                        ),
                        "critic_probe_sequence_count": critic_probe_meta[
                            "probe_sequence_count"
                        ],
                        "critic_probe_sequence_length": critic_probe_meta[
                            "probe_sequence_length"
                        ],
                    }
                )
            probe_metrics.update(
                {
                    "pgct_gate_arcs": float(np.sum(pgct_gate > 0.0)),
                    "pgct_allocation_mass": float(np.sum(peer_allocation)),
                }
            )
            probe_metrics.update(
                oracle_pairwise_matrix_metrics(
                    pgct_gate,
                    true_clusters,
                    n_agents,
                    metric_name="gate",
                    gap_name="gate_gap",
                    gap_direction="same_minus_cross",
                )
            )
        elif cfg.graph_mode in {"oracle", "wrong", "unrestricted"}:
            consensus_weights = baseline_weight_matrix(env, n_agents, cfg.graph_mode)
            log_clusters = clusters_from_adjacency(
                (consensus_weights >= cfg.edge_threshold).astype(np.float32)
            )
        elif cfg.graph_mode == "none":
            consensus_weights = np.zeros((n_agents, n_agents), dtype=np.float32)
            log_clusters = [[idx] for idx in range(n_agents)]
        else:
            graph_estimator.update_stable_adjacency()
            consensus_weights = graph_estimator.consensus_weight_matrix()
            log_clusters = graph_estimator.clusters()

        if not cfg.communication_enabled or cfg.peer_transfer_mode == "none":
            consensus_weights = np.zeros((n_agents, n_agents), dtype=np.float32)
        else:
            consensus_weights = mask_gate_by_current_neighbors(
                consensus_weights, communication_adjacency(env, n_agents, cfg.comm_graph_mode))
        set_env_training_edges(env, communication_weights(cfg, env, graph_estimator, iteration))

        canonical_bank = getattr(env.unwrapped, "_pgct_canonical_bank", None)
        if canonical_bank is not None and not (save_dir / "canonical_probes.pt").exists():
            torch.save({"sequences": canonical_bank[0].cpu(), "metadata": canonical_bank[1],
                        "seed": cfg.seed + 9173}, save_dir / "canonical_probes.pt")
        update_metrics = update_ippo(
            agents,
            optimizers,
            rollout,
            cfg,
            graph_estimator,
            device,
            rng,
            peer_allocation=peer_allocation,
            peer_probe_sequences=peer_probe_sequences,
        )

        consensus_updates = 0
        if (
            cfg.peer_transfer_mode == "consensus"
            and cfg.consensus_interval > 0
            and iteration % cfg.consensus_interval == 0
        ):
            consensus_updates = apply_confidence_weighted_critic_consensus(
                agents,
                consensus_weights,
                tau=cfg.critic_consensus_tau,
                min_weight=0.0,
            )

        env_steps = iteration * cfg.rollout_steps
        elapsed = max(time.time() - start_time, 1e-6)
        sps = env_steps / elapsed
        log_adj = (consensus_weights > 0.0).astype(np.float32)
        # Record actual optimization-time directed transfers independently of W&B.
        active_transfer = consensus_weights.copy()
        if cfg.peer_transfer_mode == "consensus" and consensus_updates == 0:
            active_transfer.fill(0)
        with (save_dir / "communication.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps({
                "env_steps": env_steps, "matrix_convention": "receiver_sender",
                "enabled": cfg.communication_enabled,
                "available": communication_adjacency(env, n_agents, cfg.comm_graph_mode).tolist(),
                "transfer_weights": active_transfer.tolist(),
                "policy_information": graph_estimator.policy_information.tolist(),
                "directed_probe_counts": graph_estimator.distance_count.tolist(),
                "policy_distance_ema": graph_estimator.distance_ema.tolist(),
                "gate_requirements": {
                    "gate_style": cfg.pgct_gate_style,
                    "warmup_updates": cfg.pgct_warmup_updates,
                    "min_directed_probe_count": cfg.min_probe_count,
                    "min_policy_information": cfg.pgct_min_policy_information,
                    "max_policy_distance": cfg.pgct_distance_threshold,
                },
            }) + "\n")
        wandb_payload: Dict[str, object] = {
            "train/env_steps": float(env_steps),
            "train/sps": float(sps),
            "train/consensus_updates": float(consensus_updates),
        }
        wandb_payload.update(prefixed_metrics("train", rollout_metrics))
        for action in Action:
            action_name = action.name.lower()
            action_key = f"action_{action.name.lower()}_rate"
            action_value = rollout_metrics.get(action_key)
            if isinstance(action_value, (float, int)) and not math.isnan(
                float(action_value)
            ):
                wandb_payload[f"policy/action_{action_name}_fraction"] = float(
                    action_value
                )
            action_probability_key = f"action_{action.name.lower()}_probability"
            action_probability = rollout_metrics.get(action_probability_key)
            if isinstance(action_probability, (float, int)) and not math.isnan(
                float(action_probability)
            ):
                wandb_payload[f"policy/action_{action_name}_probability"] = float(
                    action_probability
                )
        toggle_rate = rollout_metrics.get("action_toggle_load_rate")
        if isinstance(toggle_rate, (float, int)) and not math.isnan(float(toggle_rate)):
            wandb_payload["policy/action_toggle_fraction"] = float(toggle_rate)
        toggle_probability = rollout_metrics.get("action_toggle_load_probability")
        if isinstance(toggle_probability, (float, int)) and not math.isnan(
            float(toggle_probability)
        ):
            wandb_payload["policy/action_toggle_probability"] = float(
                toggle_probability
            )
        max_action_probability = rollout_metrics.get("max_action_probability")
        if isinstance(max_action_probability, (float, int)) and not math.isnan(
            float(max_action_probability)
        ):
            wandb_payload["policy/max_action_probability"] = float(
                max_action_probability
            )
        wandb_payload.update(prefixed_metrics("update", update_metrics))
        wandb_payload.update(prefixed_metrics("probe", probe_metrics))
        wandb_payload.update(
            graph_metrics(
                graph_estimator,
                log_clusters,
                log_adj,
                weights=consensus_weights,
            )
        )
        wandb_payload.update(
            team_discovery_metrics(log_clusters, true_clusters, log_adj, n_agents)
        )
        log_to_wandb(wandb_run, wandb_payload, step=env_steps)

        if iteration % cfg.log_interval == 0:
            print(
                f"iter={iteration:04d} steps={env_steps} sps={sps:.1f} "
                f"return={rollout_metrics['episode_return']:.3f} "
                f"task={rollout_metrics['episode_task_return']:.3f} "
                f"shape={rollout_metrics['episode_shaping_return']:.3f} "
                f"len={rollout_metrics['episode_length']:.1f} "
                f"events={rollout_metrics['team_events']:.2f} "
                f"deliveries={rollout_metrics['deliveries']:.2f} "
                f"returns={rollout_metrics['returns_home']:.2f} "
                f"pickup_sr={rollout_metrics['pickup_success_rate']:.2f} "
                f"delivery_sr={rollout_metrics['delivery_success_rate']:.2f} "
                f"full_cycle={rollout_metrics['full_cycle_success_rate']:.2f} "
                f"pi_loss={update_metrics['policy_loss']:.4f} "
                f"v_loss={update_metrics['value_loss']:.4f} "
                f"ent={update_metrics['entropy']:.3f} "
                f"rew_nz={update_metrics['reward_density']:.3f} "
                f"rew_pos={update_metrics['positive_reward_density']:.3f} "
                f"skip={update_metrics['actor_update_skipped']:.2f} "
                f"maxp={rollout_metrics['max_action_probability']:.2f} "
                f"pick_opp={rollout_metrics['valid_pickup_opportunity_count']:.0f} "
                f"ret_phase={rollout_metrics['return_phase_step_rate']:.2f} "
                f"ret_opp={rollout_metrics['valid_return_opportunity_count']:.0f} "
                f"tog_ret={rollout_metrics['toggle_probability_when_return_valid']:.2f} "
                f"ret_conv={rollout_metrics['return_conversion_rate']:.2f} "
                f"ret_dist={rollout_metrics['mean_return_home_distance']:.1f} "
                f"move={rollout_metrics['position_change_rate']:.2f} "
                f"agent_move={format_agent_rates(rollout_metrics, 'position_change_rate', n_agents)} "
                f"agent_deliv={format_agent_rates(rollout_metrics, 'delivery_count', n_agents)} "
                f"fail_fwd={rollout_metrics['failed_forward_rate']:.3f} "
                f"block={rollout_metrics['blocking_agent_rate']:.3f} "
                f"dead={rollout_metrics['deadlock_agent_rate']:.3f} "
                f"wall_fwd={rollout_metrics['wall_action_forward_rate']:.2f} "
                f"wall_pfwd={rollout_metrics['wall_action_forward_probability']:.2f} "
                f"turn={rollout_metrics['action_left_rate'] + rollout_metrics['action_right_rate']:.2f} "
                f"sim={probe_metrics.get('policy_similarity', float('nan')):.3f} "
                f"delta={probe_metrics.get('probe_delta', float('nan')):.4f} "
                f"arcs={np.sum(log_adj):.0f} "
                f"consensus_edges={consensus_updates} "
                f"clusters={format_clusters(log_clusters)}"
            )

        if cfg.eval_interval > 0 and iteration % cfg.eval_interval == 0:
            periodic_eval_count += 1
            collect_eval_video = should_collect_periodic_eval_video(
                cfg,
                periodic_eval_count,
            )
            eval_metrics, eval_video = evaluate(
                cfg.env_id,
                env_kwargs,
                cfg.observation_format,
                agents,
                cfg.eval_episodes,
                cfg.eval_horizon,
                cfg.gamma,
                obs_dim,
                cfg.recurrent_hidden_dim,
                device,
                seed=cfg.seed + 5_000_000 + env_steps,
                render_episodes=cfg.eval_render_episodes if cfg.render_eval else 0,
                render_mode=cfg.eval_render_mode,
                collect_video=collect_eval_video or cfg.save_eval_video,
                use_action_mask=cfg.action_mask,
                communication_gate=communication_weights(cfg, env, graph_estimator, iteration),
            )
            if cfg.save_eval_video and eval_video is not None:
                save_local_eval_video(eval_video, save_dir / "videos" / f"eval_{iteration:05d}.mp4", cfg.eval_render_fps)
            if eval_metrics:
                print(
                    f"eval iter={iteration:04d} "
                    f"return={eval_metrics['eval_return']:.3f} "
                    f"task={eval_metrics['eval_task_return']:.3f} "
                    f"shape={eval_metrics['eval_shaping_return']:.3f} "
                    f"length={eval_metrics['eval_length']:.1f} "
                    f"deliveries={eval_metrics['eval_deliveries']:.2f} "
                    f"returns={eval_metrics['eval_returns_home']:.2f} "
                    f"delivery_sr={eval_metrics['eval_delivery_success_rate']:.2f} "
                    f"full_cycle={eval_metrics['eval_full_cycle_success_rate']:.2f} "
                    f"maxp={eval_metrics['eval_max_action_probability']:.2f} "
                    f"ret_phase={eval_metrics['eval_return_phase_step_rate']:.2f} "
                    f"ret_opp={eval_metrics['eval_valid_return_opportunity_count']:.0f} "
                    f"tog_ret={eval_metrics['eval_toggle_probability_when_return_valid']:.2f} "
                    f"ret_conv={eval_metrics['eval_return_conversion_rate']:.2f} "
                    f"ret_dist={eval_metrics['eval_mean_return_home_distance']:.1f} "
                    f"move={eval_metrics['eval_position_change_rate']:.2f} "
                    f"agent_move={format_eval_agent_rates(eval_metrics, 'position_change_rate', n_agents)} "
                    f"agent_deliv={format_eval_agent_rates(eval_metrics, 'delivery_count', n_agents)} "
                    f"fail_fwd={eval_metrics['eval_failed_forward_rate']:.3f} "
                    f"block={eval_metrics['eval_blocking_agent_rate']:.3f} "
                    f"dead={eval_metrics['eval_deadlock_agent_rate']:.3f} "
                    f"wall_fwd={eval_metrics['eval_wall_action_forward_rate']:.2f} "
                    f"wall_pfwd={eval_metrics['eval_wall_action_forward_probability']:.2f} "
                    f"turn={eval_metrics['eval_action_left_rate'] + eval_metrics['eval_action_right_rate']:.2f}"
                )
                eval_payload: Dict[str, object] = eval_payload_from_metrics(
                    eval_metrics
                )
                if eval_video is not None and wandb_run is not None:
                    eval_payload["eval/video"] = make_wandb_video(
                        eval_video,
                        fps=cfg.eval_render_fps,
                    )
                log_to_wandb(wandb_run, eval_payload, step=env_steps)

        if cfg.save_interval > 0 and iteration % cfg.save_interval == 0:
            save_checkpoint(
                save_dir / f"checkpoint_{iteration:05d}.pt",
                cfg,
                agents,
                optimizers,
                graph_estimator,
                iteration,
                env_steps,
                env_kwargs,
            )

    eval_metrics, eval_video = evaluate(
        cfg.env_id,
        env_kwargs,
        cfg.observation_format,
        agents,
        cfg.eval_episodes,
        cfg.eval_horizon,
        cfg.gamma,
        obs_dim,
        cfg.recurrent_hidden_dim,
        device,
        seed=cfg.seed + 5_000_000,
        render_episodes=cfg.eval_render_episodes if cfg.render_eval else 0,
        render_mode=cfg.eval_render_mode,
        collect_video=cfg.save_eval_video or (cfg.track and cfg.wandb_log_eval_video),
        use_action_mask=cfg.action_mask,
        communication_gate=communication_weights(cfg, env, graph_estimator, total_iterations),
    )
    if cfg.save_eval_video and eval_video is not None:
        save_local_eval_video(eval_video, save_dir / "videos" / "final.mp4", cfg.eval_render_fps)
    with (save_dir / "final_eval_metrics.json").open("w", encoding="utf-8") as stream:
        json.dump(eval_metrics, stream, indent=2)
    if eval_metrics:
        print(
            f"eval_return={eval_metrics['eval_return']:.3f} "
            f"eval_task={eval_metrics['eval_task_return']:.3f} "
            f"eval_shape={eval_metrics['eval_shaping_return']:.3f} "
            f"eval_length={eval_metrics['eval_length']:.1f} "
            f"eval_deliveries={eval_metrics['eval_deliveries']:.2f} "
            f"eval_returns={eval_metrics['eval_returns_home']:.2f} "
            f"eval_delivery_sr={eval_metrics['eval_delivery_success_rate']:.2f} "
            f"eval_full_cycle={eval_metrics['eval_full_cycle_success_rate']:.2f} "
            f"eval_maxp={eval_metrics['eval_max_action_probability']:.2f} "
            f"eval_ret_phase={eval_metrics['eval_return_phase_step_rate']:.2f} "
            f"eval_ret_opp={eval_metrics['eval_valid_return_opportunity_count']:.0f} "
            f"eval_tog_ret={eval_metrics['eval_toggle_probability_when_return_valid']:.2f} "
            f"eval_ret_conv={eval_metrics['eval_return_conversion_rate']:.2f} "
            f"eval_ret_dist={eval_metrics['eval_mean_return_home_distance']:.1f} "
            f"eval_move={eval_metrics['eval_position_change_rate']:.2f} "
            f"eval_agent_move={format_eval_agent_rates(eval_metrics, 'position_change_rate', n_agents)} "
            f"eval_agent_deliv={format_eval_agent_rates(eval_metrics, 'delivery_count', n_agents)} "
            f"eval_fail_fwd={eval_metrics['eval_failed_forward_rate']:.3f} "
            f"eval_block={eval_metrics['eval_blocking_agent_rate']:.3f} "
            f"eval_dead={eval_metrics['eval_deadlock_agent_rate']:.3f} "
            f"eval_wall_fwd={eval_metrics['eval_wall_action_forward_rate']:.2f} "
            f"eval_wall_pfwd={eval_metrics['eval_wall_action_forward_probability']:.2f} "
            f"eval_turn={eval_metrics['eval_action_left_rate'] + eval_metrics['eval_action_right_rate']:.2f}"
        )
        final_eval_payload: Dict[str, object] = eval_payload_from_metrics(
            eval_metrics,
            final=True,
        )
        if eval_video is not None and wandb_run is not None:
            final_eval_payload["eval/final_video"] = make_wandb_video(
                eval_video,
                fps=cfg.eval_render_fps,
            )
        log_to_wandb(wandb_run, final_eval_payload, step=total_iterations * cfg.rollout_steps)

    final_path = save_dir / "final.pt"
    save_checkpoint(
        final_path,
        cfg,
        agents,
        optimizers,
        graph_estimator,
        total_iterations,
        total_iterations * cfg.rollout_steps,
        env_kwargs,
    )
    env.close()
    if wandb_run is not None:
        wandb_run.finish()
    print(f"saved={final_path}")


if __name__ == "__main__":
    main()
