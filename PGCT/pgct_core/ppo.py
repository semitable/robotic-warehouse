"""Recurrent IPPO update shared by new environments.

``ppo_update`` reproduces the IPPO path of the RWARE trainer's ``update_ippo`` (per-agent advantage
normalisation, clipped policy and value losses, entropy bonus, zero-reward actor skip, gradient clipping,
the same minibatch order) without the legacy critic peer-distillation term, which is zero whenever no peer
allocation is given. RWARE's own trainer is not modified; ``RWARE/tests/test_pgct_core_equivalence.py``
checks that both produce bit-identical weights from the same inputs.

Max-GAE enters through ``advantages``/``returns``: the actor uses the supplied advantages while the critic
keeps the receiver's own returns.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Sequence

import numpy as np
import torch
from torch import nn

from .max_gae import compute_gae


@dataclass
class PPOConfig:
    # Defaults = the RWARE reference from-scratch configuration (e1 'pgct' template, lr/entropy of D-runs).
    rollout_steps: int = 1024
    sequence_length: int = 64
    minibatch_chunks: int = 8
    ppo_epochs: int = 4
    learning_rate: float = 2e-4
    gamma: float = .99
    gae_lambda: float = .95
    clip_coef: float = .2
    value_clip_coef: float = .2
    value_loss_coef: float = .5
    entropy_coef: float = .05
    normalize_advantages: bool = True
    min_advantage_std: float = 1e-6
    skip_zero_reward_actor_update: bool = True
    skip_no_positive_reward_actor_update: bool = False
    reward_epsilon: float = 1e-8
    max_grad_norm: float = .5
    mlp_hidden_dim: int = 128
    recurrent_hidden_dim: int = 128

    def __post_init__(self):
        if self.rollout_steps % self.sequence_length:
            raise ValueError("rollout_steps must be divisible by sequence_length")
        if self.minibatch_chunks < 1 or self.ppo_epochs < 1:
            raise ValueError("minibatch_chunks and ppo_epochs must be positive")


@dataclass
class Rollout:
    obs: torch.Tensor            # [T, n, D]
    actions: torch.Tensor        # [T, n]
    action_masks: torch.Tensor   # [T, n, A]
    old_log_probs: torch.Tensor  # [T, n]
    rewards: torch.Tensor        # [T, n] individual rewards
    dones: torch.Tensor          # [T] episode boundary (terminated or truncated)
    values: torch.Tensor         # [T, n]
    actor_h: torch.Tensor        # [T, n, H] hidden state before step t
    critic_h: torch.Tensor
    last_values: torch.Tensor    # [n]
    timeout_values: Optional[torch.Tensor] = None  # [T, n] V(s_timeout) at truncation steps, else 0


def recurrent_minibatches(rollout: Rollout, agent_id: int, sequence_length: int, minibatch_chunks: int,
                          rng: np.random.Generator):
    rollout_steps = rollout.obs.shape[0]
    starts = np.arange(0, rollout_steps, sequence_length, dtype=np.int64)
    rng.shuffle(starts)
    for offset in range(0, len(starts), minibatch_chunks):
        mb = starts[offset: offset + minibatch_chunks]
        cut = lambda x: torch.stack([x[s: s + sequence_length, agent_id] for s in mb], dim=1)
        done_seq = torch.stack([rollout.dones[s: s + sequence_length] for s in mb], dim=1)
        yield (cut(rollout.obs), cut(rollout.actions), cut(rollout.action_masks), cut(rollout.old_log_probs),
               cut(rollout.values), done_seq, rollout.actor_h[mb, agent_id], rollout.critic_h[mb, agent_id], mb)


def ppo_update(agents: Sequence[nn.Module], optimizers, rollout: Rollout, cfg: PPOConfig, device, rng,
               advantages: Optional[torch.Tensor] = None, returns: Optional[torch.Tensor] = None,
               frozen: Sequence[int] = ()) -> Dict[str, float]:
    """``frozen``: agent indices that are not updated (e.g. a pre-trained expert kept fixed); empty = all agents."""
    if (advantages is None) != (returns is None):
        raise ValueError("Pass both advantages and returns, or neither")
    if advantages is None:
        advantages, returns = compute_gae(rollout.rewards, rollout.dones, rollout.values, rollout.last_values,
                                          cfg.gamma, cfg.gae_lambda, rollout.timeout_values)
    names = ("policy_loss", "value_loss", "entropy", "approx_kl", "clip_fraction", "advantage_std",
             "reward_density", "positive_reward_density", "negative_reward_density", "actor_update_skipped")
    metrics = {name: [] for name in names}
    diagnostics: Dict[str, float] = {}
    for agent_id, (net, optimizer) in enumerate(zip(agents, optimizers)):
        if agent_id in frozen:
            continue
        raw_adv = advantages[:, agent_id]
        adv_std = raw_adv.std(unbiased=False)
        if cfg.normalize_advantages:
            centered = raw_adv - raw_adv.mean()
            agent_adv = centered / (adv_std + 1e-8) if float(adv_std.item()) >= cfg.min_advantage_std else centered
        else:
            agent_adv = raw_adv
        agent_rewards = rollout.rewards[:, agent_id]
        reward_density = torch.mean((torch.abs(agent_rewards) > cfg.reward_epsilon).float())
        positive_density = torch.mean((agent_rewards > cfg.reward_epsilon).float())
        negative_density = torch.mean((agent_rewards < -cfg.reward_epsilon).float())
        skip_actor = ((cfg.skip_zero_reward_actor_update
                       and bool(torch.all(torch.abs(agent_rewards) <= cfg.reward_epsilon).item()))
                      or (cfg.skip_no_positive_reward_actor_update
                          and bool(torch.all(agent_rewards <= cfg.reward_epsilon).item())))
        agent_returns = returns[:, agent_id]
        diagnostics[f"agent_{agent_id}/actor_update_skipped"] = float(skip_actor)
        diagnostics[f"agent_{agent_id}/positive_reward_density"] = float(positive_density.item())
        diagnostics[f"agent_{agent_id}/advantage_std"] = float(adv_std.item())
        for _ in range(cfg.ppo_epochs):
            for (obs_seq, actions_seq, mask_seq, old_logprob_seq, old_value_seq, done_seq, actor_h0, critic_h0,
                 mb_starts) in recurrent_minibatches(rollout, agent_id, cfg.sequence_length, cfg.minibatch_chunks, rng):
                adv_seq = torch.stack([agent_adv[s: s + cfg.sequence_length] for s in mb_starts], dim=1).to(device)
                return_seq = torch.stack([agent_returns[s: s + cfg.sequence_length] for s in mb_starts],
                                         dim=1).to(device)
                obs_seq, actions_seq, mask_seq = obs_seq.to(device), actions_seq.to(device), mask_seq.to(device)
                old_logprob_seq, old_value_seq, done_seq = (old_logprob_seq.to(device), old_value_seq.to(device),
                                                            done_seq.to(device))
                new_logprob, entropy, value = net.evaluate_actions(obs_seq, actions_seq, done_seq,
                                                                   actor_h0.to(device), critic_h0.to(device), mask_seq)
                log_ratio = new_logprob - old_logprob_seq
                ratio = log_ratio.exp()
                policy_loss = torch.max(-adv_seq * ratio,
                                        -adv_seq * torch.clamp(ratio, 1.0 - cfg.clip_coef, 1.0 + cfg.clip_coef)).mean()
                value_clipped = old_value_seq + torch.clamp(value - old_value_seq, -cfg.value_clip_coef,
                                                            cfg.value_clip_coef)
                value_loss = 0.5 * torch.max((value - return_seq).pow(2), (value_clipped - return_seq).pow(2)).mean()
                entropy_loss = entropy.mean()
                actor_loss = policy_loss - cfg.entropy_coef * entropy_loss
                if skip_actor:
                    actor_loss = -cfg.entropy_coef * entropy_loss
                loss = actor_loss + cfg.value_loss_coef * value_loss
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(net.parameters(), cfg.max_grad_norm)
                optimizer.step()
                with torch.no_grad():
                    approx_kl = ((ratio - 1.0) - log_ratio).mean()
                    clip_fraction = (torch.abs(ratio - 1.0) > cfg.clip_coef).float().mean()
                for name, value_ in (("policy_loss", policy_loss), ("value_loss", value_loss),
                                     ("entropy", entropy_loss), ("approx_kl", approx_kl),
                                     ("clip_fraction", clip_fraction), ("advantage_std", adv_std),
                                     ("reward_density", reward_density),
                                     ("positive_reward_density", positive_density),
                                     ("negative_reward_density", negative_density)):
                    metrics[name].append(float(value_.item()))
                metrics["actor_update_skipped"].append(float(skip_actor))
    return {**diagnostics, **{k: float(np.mean(v)) if v else float("nan") for k, v in metrics.items()}}
