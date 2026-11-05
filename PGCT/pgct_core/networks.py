"""Per-agent recurrent actor-critic (MLP encoder + GRU), same construction as the RWARE trainer's
``RecurrentActorCritic`` with ``encoder_type="mlp"``: identical module order, initialisation and state-dict
keys, so the same torch seed gives the same weights and RWARE MLP checkpoints load unchanged.

The actor and the critic only see the agent's own local observation; no global state, other agents'
actions or team labels.
"""
from __future__ import annotations

import math
from typing import Iterable, List, Optional, Tuple

import torch
from torch import nn
from torch.distributions import Categorical

TensorPair = Tuple[torch.Tensor, torch.Tensor]


def mlp(input_dim: int, hidden_dim: int, output_dim: int) -> nn.Sequential:
    return nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.Tanh(), nn.Linear(hidden_dim, hidden_dim), nn.Tanh(),
                         nn.Linear(hidden_dim, output_dim), nn.Tanh())


class RecurrentActorCritic(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int, mlp_hidden_dim: int = 128, recurrent_hidden_dim: int = 128):
        super().__init__()
        self.obs_dim = int(obs_dim)
        self.action_dim = int(action_dim)
        self.recurrent_hidden_dim = int(recurrent_hidden_dim)
        self.encoder_type = "mlp"
        self.actor_encoder = mlp(obs_dim, mlp_hidden_dim, recurrent_hidden_dim)
        self.critic_encoder = mlp(obs_dim, mlp_hidden_dim, recurrent_hidden_dim)
        self.actor_rnn = nn.GRUCell(recurrent_hidden_dim, recurrent_hidden_dim)
        self.actor_head = nn.Linear(recurrent_hidden_dim, action_dim)
        self.critic_rnn = nn.GRUCell(recurrent_hidden_dim, recurrent_hidden_dim)
        self.critic_head = nn.Linear(recurrent_hidden_dim, 1)
        for module in self.modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
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

    def _actor_step(self, obs, actor_h):
        next_h = self.actor_rnn(self.actor_encoder(obs), actor_h)
        return self.actor_head(next_h), next_h

    def _critic_step(self, obs, critic_h):
        next_h = self.critic_rnn(self.critic_encoder(obs), critic_h)
        return self.critic_head(next_h).squeeze(-1), next_h

    @staticmethod
    def _masked_logits(logits, action_mask: Optional[torch.Tensor]):
        if action_mask is None:
            return logits
        if action_mask.ndim == 1:
            action_mask = action_mask.unsqueeze(0)
        action_mask = action_mask.to(device=logits.device, dtype=torch.bool)
        has_valid = action_mask.any(dim=-1, keepdim=True)
        if not bool(has_valid.all()):
            fallback = torch.zeros_like(action_mask)
            fallback[..., 0] = True
            action_mask = torch.where(has_valid, action_mask, fallback)
        return logits.masked_fill(~action_mask, -1.0e9)

    @torch.no_grad()
    def act(self, obs, actor_h, critic_h, deterministic: bool = False, action_mask=None):
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
        action = torch.argmax(logits, dim=-1) if deterministic else dist.sample()
        return action, dist.log_prob(action), value, next_actor_h, next_critic_h

    def evaluate_actions(self, obs_seq, action_seq, done_seq, actor_h0, critic_h0, action_mask_seq=None):
        """Time-major recurrent minibatch [T, B, ...]; hidden states reset after each done."""
        actor_h, critic_h = actor_h0, critic_h0
        log_probs: List[torch.Tensor] = []
        entropies: List[torch.Tensor] = []
        values: List[torch.Tensor] = []
        for step_idx, (obs_t, action_t, done_t) in enumerate(zip(obs_seq, action_seq, done_seq)):
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
        return torch.stack(log_probs, dim=0), torch.stack(entropies, dim=0), torch.stack(values, dim=0)
