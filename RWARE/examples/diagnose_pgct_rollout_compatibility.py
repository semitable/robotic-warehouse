"""Frozen common-history policy/critic diagnostics; no training or gate updates."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

try:
    from . import train_recurrent_ippo_consensus as T
    from .pgct_probe_diagnostics import PAIRS, PAIR_LABELS, js_divergence
except ImportError:
    import train_recurrent_ippo_consensus as T
    from pgct_probe_diagnostics import PAIRS, PAIR_LABELS, js_divergence

MODES = ["recurrent_masked", "recurrent_unmasked", "zero_hidden_masked", "zero_hidden_unmasked"]


def environment_signature(kwargs):
    """Resolve the E1 generator defaults; communication flags do not change actor input.

    Communication is disabled in IPPO. It does not enter this semantic observation
    or the actor dynamics, so it must not prevent matched on/off history replay.
    All task/reward parameters and map geometry remain part of the comparison.
    """
    result = dict(kwargs)
    for key, value in dict(shelf_rows=1, shelf_columns=3, column_height=8).items():
        result.setdefault(key, value)
    for key in ("communication_enabled", "communication_topology", "communication_range"):
        result.pop(key, None)
    return result


def model_hash(agents):
    digest = hashlib.sha256()
    for agent in agents:
        for key, tensor in agent.state_dict().items():
            digest.update(key.encode())
            digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def load_model(path):
    checkpoint = T._load_checkpoint(str(path), torch.device("cpu"))
    cfg = checkpoint["config"]
    if len(checkpoint["agents"]) != 4 or cfg["observation_format"] != "semantic":
        raise ValueError("This diagnostic requires a four-agent semantic checkpoint.")
    spec = T.get_semantic_observation_spec(SimpleNamespace(sensor_range=cfg["sensor_range"]))
    agents = []
    for state in checkpoint["agents"]:
        net = T.RecurrentActorCritic(spec.obs_dim, 5, cfg["mlp_hidden_dim"], cfg["recurrent_hidden_dim"],
                                    cfg["obs_encoder"], spec.spatial_channels, spec.spatial_size, spec.ego_dim)
        net.load_state_dict(state)
        net.eval()
        agents.append(net)
    metadata = dict(checkpoint=str(Path(path).resolve()), env_steps=checkpoint["env_steps"],
                    config=cfg, env_kwargs=checkpoint["effective_env_kwargs"])
    return agents, metadata


@torch.inference_mode()
def collect_histories(agents, metadata, *, episodes, horizon, seed, deterministic=False):
    """One fixed behavior policy supplies all compared checkpoints' input histories."""
    cfg, env_kwargs = metadata["config"], metadata["env_kwargs"]
    rng = np.random.default_rng(seed)
    observations, masks_log, rewards_log, phases_log, episode_ids, time_ids = [], [], [], [], [], []
    endings = []
    for episode in range(episodes):
        with T.make_env(cfg["env_id"], env_kwargs, cfg["observation_format"]) as env:
            obs, _ = env.reset(seed=seed + episode)
            h = [net.initial_hidden(1, torch.device("cpu"))[0] for net in agents]
            ending = "diagnostic_horizon"
            for step in range(horizon):
                x = np.asarray(T.flatten_multi_agent_obs(env, obs), dtype=np.float32)
                masks = T.local_rware_action_masks(env, 4, 5, enabled=cfg["action_mask"])
                observations.append(x)
                masks_log.append(masks)
                episode_ids.append(episode)
                time_ids.append(step)
                phases_log.append([0 if a.carrying_shelf is None else
                                   (2 if a.carrying_shelf.id in env.unwrapped._shelves_awaiting_return else 1)
                                   for a in env.unwrapped.agents])
                actions = []
                for i, net in enumerate(agents):
                    logits, h[i] = net._actor_step(torch.as_tensor(x[i]).unsqueeze(0), h[i])
                    p = net._masked_logits(logits, torch.as_tensor(masks[i])).softmax(-1).numpy()[0].astype(float)
                    p /= p.sum()
                    actions.append(int(p.argmax()) if deterministic else int(rng.choice(5, p=p)))
                obs, rewards, terminated, truncated, _ = env.step(actions)
                rewards_log.append(rewards)
                if bool(np.any(terminated)):
                    ending = "terminated"
                    break
                if bool(np.any(truncated)):
                    ending = "truncated"
                    break
            endings.append(dict(episode=episode, length=step + 1, ending=ending))
        print(f"Collected episode {episode+1}/{episodes}", flush=True)
    return dict(observations=np.asarray(observations, dtype=np.float32),
                action_masks=np.asarray(masks_log, dtype=np.float32),
                rewards=np.asarray(rewards_log, dtype=np.float32),
                phases=np.asarray(phases_log, dtype=np.int8),
                episode_ids=np.asarray(episode_ids, dtype=np.int32), time_ids=np.asarray(time_ids, dtype=np.int32),
                metadata_json=np.asarray(json.dumps(dict(reference_checkpoint=metadata["checkpoint"],
                    env_kwargs=env_kwargs, env_id=cfg["env_id"], gamma=cfg["gamma"], sensor_range=cfg["sensor_range"],
                    behavior_policy="deterministic" if deterministic else "stochastic",
                    seed=seed, episodes=episodes, horizon=horizon, endings=endings,
                    note="Fixed empirical histories are off-policy for non-reference actors. "
                         "Stored reference-policy rewards are not other policies' on-policy value targets."))))


@torch.inference_mode()
def compare_histories(agents, cache, sample_interval=10):
    """Every actor and critic sees each of the same four observation streams.

    Hidden states are formed by each network from the identical empirical prefix;
    hidden vectors are never copied between networks. Episode boundaries reset all.
    """
    actor_h = [net.initial_hidden(4, torch.device("cpu"))[0] for net in agents]
    critic_h = [net.initial_hidden(4, torch.device("cpu"))[1] for net in agents]
    collected = {name: [] for name in MODES}
    critic_squared, critic_zero_squared, values_log, index_log = [], [], [], []
    for index, observation in enumerate(cache["observations"]):
        if cache["time_ids"][index] == 0:
            actor_h = [net.initial_hidden(4, torch.device("cpu"))[0] for net in agents]
            critic_h = [net.initial_hidden(4, torch.device("cpu"))[1] for net in agents]
        x = torch.as_tensor(observation)
        masks = torch.as_tensor(cache["action_masks"][index])
        sampled = int(cache["time_ids"][index]) % sample_interval == 0
        probabilities = {name: [] for name in MODES}
        values, zero_values = [], []
        for i, net in enumerate(agents):
            logits, actor_h[i] = net._actor_step(x, actor_h[i])
            value, critic_h[i] = net._critic_step(x, critic_h[i])
            if sampled:
                zero_logits, _ = net._actor_step(x, torch.zeros_like(actor_h[i]))
                zero_value, _ = net._critic_step(x, torch.zeros_like(critic_h[i]))
                probabilities[MODES[0]].append(net._masked_logits(logits, masks).softmax(-1).numpy())
                probabilities[MODES[1]].append(logits.softmax(-1).numpy())
                probabilities[MODES[2]].append(net._masked_logits(zero_logits, masks).softmax(-1).numpy())
                probabilities[MODES[3]].append(zero_logits.softmax(-1).numpy())
                values.append(value.numpy().reshape(4))
                zero_values.append(zero_value.numpy().reshape(4))
        if sampled:
            for name in MODES:
                p = np.asarray(probabilities[name])
                collected[name].append(np.stack([js_divergence(p[i], p[j]) for i, j in PAIRS], axis=-1))
            values, zero_values = np.asarray(values), np.asarray(zero_values)
            critic_squared.append(np.stack([(values[i] - values[j]) ** 2 for i, j in PAIRS], axis=-1))
            critic_zero_squared.append(np.stack([(zero_values[i] - zero_values[j]) ** 2 for i, j in PAIRS], axis=-1))
            values_log.append(values)
            index_log.append(index)
    return dict(**{name: np.asarray(rows) for name, rows in collected.items()},
                critic_squared=np.asarray(critic_squared), critic_zero_squared=np.asarray(critic_zero_squared),
                values=np.asarray(values_log), observation_indices=np.asarray(index_log))


def summarize(result, cache):
    policy_means = {name: result[name].mean((0, 1)) for name in MODES}
    value_mean = result["critic_squared"].mean((0, 1))
    value_zero = result["critic_zero_squared"].mean((0, 1))
    pairs = [dict(pair=f"{i+1}-{j+1}", same_team=i % 2 == j % 2,
                  **{name + "_js": float(policy_means[name][k]) for name in MODES},
                  recurrent_critic_mse=float(value_mean[k]), zero_hidden_critic_mse=float(value_zero[k]))
             for k, (i, j) in enumerate(PAIRS)]
    correlation = None
    x = policy_means[MODES[0]]
    if x.std() > 0 and value_mean.std() > 0:
        correlation = float(np.corrcoef(x, value_mean)[0, 1])
    indices = result["observation_indices"]
    phases = cache["phases"][indices]
    by_phase = {}
    for phase, label in enumerate(["empty", "delivery_carry", "return_carry"]):
        mask = phases == phase
        if mask.any():
            by_phase[label] = dict(samples=int(mask.sum()),
                policy_js=result[MODES[0]][mask].mean(0).tolist(), critic_mse=result["critic_squared"][mask].mean(0).tolist())
    return dict(pairs=pairs, sampled_observation_stream_instances=len(indices) * 4,
                policy_js_ranking=[pairs[k]["pair"] for k in np.argsort(x)],
                critic_mse_ranking=[pairs[k]["pair"] for k in np.argsort(value_mean)],
                descriptive_pair_mean_pearson=correlation, pair_count=6,
                correlation_limitation="Only six dependent pairs; not a significance test or evidence of transfer usefulness.",
                phase_groups=by_phase,
                common_history_convention="Same empirical observation/mask history; separate network-specific actor/critic hidden states.",
                value_limitation="Prediction agreement is not value accuracy, reward compatibility, or transfer utility.")


def plot_summary(summaries, path):
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5), dpi=140, constrained_layout=True)
    x = np.arange(6)
    for model_index, model in enumerate(summaries):
        pairs = model["summary"]["pairs"]
        label = model["label"]
        axes[0].plot(x, [p["recurrent_masked_js"] for p in pairs], "o-", label=label + " / recurrent")
        axes[0].plot(x, [p["zero_hidden_masked_js"] for p in pairs], "x--", label=label + " / zero")
        axes[1].plot(x, [p["recurrent_critic_mse"] for p in pairs], "o-", label=label)
        px = [p["recurrent_masked_js"] for p in pairs]
        py = [p["recurrent_critic_mse"] for p in pairs]
        axes[2].scatter(px, py, label=label)
        for pair, a, b in zip(pairs, px, py):
            offset_x = -20 if pair["pair"] == "1-2" else 4
            offset_y = 5 if model_index == 0 else -12
            axes[2].annotate(pair["pair"], (a, b), xytext=(offset_x, offset_y), textcoords="offset points", fontsize=8)
    for ax in axes[:2]:
        ax.set_xticks(x, PAIR_LABELS, rotation=35, ha="right")
    axes[0].set_ylabel("Mean policy JS (natural log)"); axes[0].set_ylim(0, np.log(2))
    axes[0].set_title("Fixed rollout histories / action mask applied")
    axes[1].set_ylabel("Mean squared critic prediction difference")
    axes[1].set_title("Same observation histories / recurrent critics")
    axes[2].set_xlabel("Recurrent masked policy JS"); axes[2].set_ylabel("Critic prediction MSE")
    axes[2].set_title("Six pair means / descriptive association")
    axes[0].legend(fontsize=7); axes[2].legend(fontsize=7)
    fig.savefig(path); plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoints", type=Path, nargs="+", required=True)
    parser.add_argument("--reference-checkpoint", type=Path)
    parser.add_argument("--cache", type=Path, help="Reuse an existing common-history NPZ without new environment rollouts.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--horizon", type=int, default=500)
    parser.add_argument("--seed", type=int, default=5000001)
    parser.add_argument("--sample-interval", type=int, default=10)
    parser.add_argument("--deterministic", action="store_true")
    args = parser.parse_args()
    if min(args.episodes, args.horizon, args.sample_interval) <= 0:
        parser.error("Episode, horizon and sampling interval must be positive.")
    torch.set_num_threads(1)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    if args.cache:
        with np.load(args.cache, allow_pickle=False) as archive:
            cache = {name: archive[name] for name in archive.files}
        cache_path = args.cache.resolve()
    else:
        agents, metadata = load_model(args.reference_checkpoint or args.checkpoints[0])
        before = model_hash(agents)
        cache = collect_histories(agents, metadata, episodes=args.episodes, horizon=args.horizon,
                                  seed=args.seed, deterministic=args.deterministic)
        assert before == model_hash(agents), "Reference model changed during inference."
        cache_path = args.output_dir / "histories.npz"
        np.savez_compressed(cache_path, **cache)
        del agents
    cache_metadata = json.loads(str(cache["metadata_json"]))
    summaries = []
    for index, checkpoint_path in enumerate(args.checkpoints):
        agents, metadata = load_model(checkpoint_path)
        if cache_metadata["sensor_range"] != metadata["config"]["sensor_range"] or \
                cache_metadata["env_id"] != metadata["config"]["env_id"] or \
                environment_signature(cache_metadata["env_kwargs"]) != environment_signature(metadata["env_kwargs"]):
            raise ValueError("Checkpoint environment differs from the fixed history environment.")
        before = model_hash(agents)
        print(f"Evaluating {checkpoint_path}", flush=True)
        result = compare_histories(agents, cache, args.sample_interval)
        assert before == model_hash(agents), "Compared model changed during inference."
        summary = summarize(result, cache)
        np.savez_compressed(args.output_dir / f"model_{index}_samples.npz", **result)
        summaries.append(dict(checkpoint=metadata["checkpoint"], env_steps=metadata["env_steps"],
                              label=f"{checkpoint_path.parent.name} / {metadata['env_steps']}",
                              weights_sha256=before, weights_unchanged=True, summary=summary))
        print(json.dumps(summary["pairs"]), flush=True)
        del agents
    report = dict(cache=str(cache_path.resolve()), cache_metadata=cache_metadata, sample_interval=args.sample_interval,
                  snapshots=summaries, note="Frozen forward-only diagnostics. Gates, optimizer and training are untouched.")
    (args.output_dir / "summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    plot_summary(summaries, args.output_dir / "compatibility.png")
    print(f"Diagnostics: {args.output_dir.resolve()}", flush=True)


if __name__ == "__main__":
    main()
