"""Read-only common-input policy diagnostics and stochastic critic evaluation."""
from __future__ import annotations

import numpy as np
import torch

try:
    from . import train_recurrent_ippo_consensus as T
except ImportError:
    import train_recurrent_ippo_consensus as T

PAIRS = [(0, 2), (1, 3), (0, 1), (0, 3), (1, 2), (2, 3)]
PAIR_LABELS = ["1-3 same", "2-4 same", "1-2 cross", "1-4 cross", "2-3 cross", "3-4 cross"]
ACTIONS = ["NOOP", "FORWARD", "LEFT", "RIGHT", "TOGGLE"]


def js_divergence(p, q):
    """Natural-log JS divergence (not its square root), last axis = action."""
    p, q = np.asarray(p, dtype=np.float64), np.asarray(q, dtype=np.float64)
    if p.shape != q.shape or np.any(p < 0) or np.any(q < 0):
        raise ValueError("Probability arrays must have equal shapes and nonnegative entries.")
    if not np.allclose(p.sum(-1), 1, atol=1e-6) or not np.allclose(q.sum(-1), 1, atol=1e-6):
        raise ValueError("Action probabilities must sum to one.")
    m = (p + q) / 2
    # Zero-probability terms contribute zero, without artificial probability mass.
    def kl(a):
        return np.sum(a * (np.log(np.maximum(a, 1e-300)) - np.log(np.maximum(m, 1e-300))), -1)
    return np.maximum((kl(p) + kl(q)) / 2, 0)


@torch.inference_mode()
def common_probe_response(agents, sequences, device):
    if sequences.ndim != 3:
        raise ValueError("Diagnostics require a shared [probe, time, observation] bank.")
    probabilities = []
    for net in agents:
        h, _ = net.initial_hidden(len(sequences), device)
        steps = []
        for t in range(sequences.shape[1]):
            logits, h = net._actor_step(sequences[:, t].to(device), h)
            steps.append(logits.softmax(-1).cpu().numpy())
        probabilities.append(np.stack(steps, 1))
    probs = np.stack(probabilities)
    per_probe_js = np.stack([js_divergence(probs[i], probs[j]).mean(1) for i, j in PAIRS])
    _, ids = np.unique(sequences.cpu().numpy().reshape(len(sequences), -1), axis=0, return_inverse=True)
    representatives = np.asarray([np.flatnonzero(ids == k)[0] for k in range(ids.max() + 1)])
    mean_js = np.zeros((4, 4))
    unique_js = np.zeros_like(mean_js)
    for k, (i, j) in enumerate(PAIRS):
        mean_js[i, j] = mean_js[j, i] = per_probe_js[k].mean()
        unique_js[i, j] = unique_js[j, i] = per_probe_js[k, representatives].mean()
    return dict(probabilities=probs.tolist(), js_per_probe=per_probe_js.tolist(),
                mean_js=mean_js.tolist(), unique_mean_js=unique_js.tolist(),
                unique_probe_count=len(representatives), input_equivalence_ids=ids.tolist(),
                entropy_per_probe=(-probs * np.log(np.maximum(probs, 1e-300))).sum(-1).mean(2).tolist(),
                action_mask_applied=False, hidden_state="zero_at_each_probe_sequence_start",
                log_base="e", comparison="identical_shared_inputs")


def finite_horizon_returns(rewards, gamma):
    rewards = np.asarray(rewards, dtype=np.float64)
    result = np.zeros_like(rewards)
    future = np.zeros(rewards.shape[1])
    for t in reversed(range(len(rewards))):
        future = rewards[t] + gamma * future
        result[t] = future
    return result


@torch.inference_mode()
def critic_rollouts(cfg, agents, env_kwargs, device, *, episodes, horizon, seed, make_env):
    """Sample the stochastic actor with local NumPy RNG; never consume training RNG.

    MC targets end at the evaluation horizon. TD targets bootstrap at time limits,
    but not genuine terminations. Common-history JS runs all actors on each of the
    first episode's four observed streams, with each actor's own recurrent state.
    """
    rng = np.random.default_rng(seed)
    error_sq, error_abs, td_abs, samples = np.zeros(4), np.zeros(4), np.zeros(4), 0
    history_js_sum, zero_js_sum, history_samples = np.zeros((4, 4)), np.zeros((4, 4)), 0
    returns_by_episode = []
    for episode in range(episodes):
        env = make_env(cfg.env_id, env_kwargs, cfg.observation_format)
        try:
            obs, _ = env.reset(seed=seed + episode)
            actor_h = torch.zeros((4, cfg.recurrent_hidden_dim), device=device)
            critic_h = torch.zeros_like(actor_h)
            # Each policy has four states, one per common observed history stream.
            common_h = [torch.zeros_like(actor_h) for _ in agents]
            rewards_log, values_log = [], []
            terminated = False
            for step in range(horizon):
                x = torch.as_tensor(T.flatten_multi_agent_obs(env, obs), dtype=torch.float32, device=device)
                masks = torch.as_tensor(T.local_rware_action_masks(
                    env, 4, agents[0].action_dim, enabled=cfg.action_mask), device=device)
                if episode == 0:
                    history_probs, zero_probs = [], []
                    for k, net in enumerate(agents):
                        logits, common_h[k] = net._actor_step(x, common_h[k])
                        if step % 10 == 0:
                            history_probs.append(net._masked_logits(logits, masks).softmax(-1).cpu().numpy())
                            zero_probs.append(net.action_probabilities(x, torch.zeros_like(actor_h), masks).cpu().numpy())
                    if history_probs:
                        for i, j in PAIRS:
                            d = js_divergence(history_probs[i], history_probs[j]).sum()
                            z = js_divergence(zero_probs[i], zero_probs[j]).sum()
                            history_js_sum[i, j] += d
                            history_js_sum[j, i] += d
                            zero_js_sum[i, j] += z
                            zero_js_sum[j, i] += z
                        history_samples += 4
                actions, values = [], []
                for i, net in enumerate(agents):
                    p = net.action_probabilities(x[i], actor_h[i], masks[i]).cpu().numpy()[0].astype(np.float64)
                    p /= p.sum()
                    actions.append(int(rng.choice(len(p), p=p)))
                    _, _, value, ah, ch = net.act(x[i], actor_h[i], critic_h[i], deterministic=True,
                                                action_mask=masks[i])
                    values.append(float(value.item()))
                    actor_h[i], critic_h[i] = ah.squeeze(0), ch.squeeze(0)
                obs, rewards, done, truncated, _ = env.step(actions)
                rewards_log.append(rewards)
                values_log.append(values)
                terminated = bool(np.any(done))
                if terminated or bool(np.any(truncated)):
                    break
            rewards, values = np.asarray(rewards_log), np.asarray(values_log)
            target = finite_horizon_returns(rewards, cfg.gamma)
            error_sq += ((values - target) ** 2).sum(0)
            error_abs += np.abs(values - target).sum(0)
            next_x = torch.as_tensor(T.flatten_multi_agent_obs(env, obs), dtype=torch.float32, device=device)
            boundary = np.zeros(4) if terminated else np.asarray([
                net._critic_step(next_x[i].unsqueeze(0), critic_h[i].unsqueeze(0))[0].item()
                for i, net in enumerate(agents)])
            next_values = np.concatenate([values[1:], boundary[None]], axis=0)
            td_abs += np.abs(rewards + cfg.gamma * next_values - values).sum(0)
            samples += len(values)
            returns_by_episode.append(rewards.sum(0).tolist())
        finally:
            env.close()
    metrics = dict(critic_mc_mse=float(error_sq.sum() / (samples * 4)),
                   critic_mc_mae=float(error_abs.sum() / (samples * 4)),
                   critic_td_abs=float(td_abs.sum() / (samples * 4)))
    for i in range(4):
        metrics.update({f"agent_{i+1}_critic_mc_mse": float(error_sq[i] / samples),
                        f"agent_{i+1}_critic_td_abs": float(td_abs[i] / samples)})
    return dict(metrics=metrics, policy="stochastic_masked_actor", episodes=episodes,
                horizon=horizon, mc_target="finite_horizon_discounted_return_zero_tail",
                limitation="MC excludes rewards after the evaluation horizon; continuing-value tail bias remains.",
                td_boundary="bootstrap_on_time_limit_only", samples_per_agent=samples,
                episode_returns=returns_by_episode,
                common_history_masked_js=(history_js_sum / history_samples).tolist(),
                common_observation_zero_hidden_masked_js=(zero_js_sum / history_samples).tolist(),
                common_history_samples=history_samples,
                history_comparison="All actors receive the same empirical observation and mask history; "
                                   "each maintains its own hidden state. These histories are off-policy for other actors.")
