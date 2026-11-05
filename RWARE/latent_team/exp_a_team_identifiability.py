"""Experiment A: can locally observable behaviour identify latent teams? Frozen models only.

Each checkpoint runs its own stochastic masked policy (weights hashed before and
after; no optimizer, gate or probe update). Every agent pair is scored by:

  canonical_js   current PGCT signal: identical canonical inputs, zero hidden (model level)
  history_js     the two policies on the two agents' own observation histories
  occupancy_js   spatial-role similarity of visited-cell distributions
  goal_evidence  pickups/deliveries that the observer sees inside its own sensor window,
                 classified by the observer's team-relative view (is that shelf requested
                 for me / is that goal one of my goals?)

and the critic-consensus premise is checked on the same histories:

  critic_gap     -E[(V_i - V_j)^2] on each agent's own history
  blend_gain     MSE(V_i, G_i) - MSE((V_i + V_j)/2, G_i) on the early part of i's history

Team labels only score the results; no score is computed from them.
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

try:
    from . import common as C
except ImportError:
    import common as C

from examples import train_recurrent_ippo_consensus as T
from examples.diagnose_pgct_rollout_compatibility import load_model, model_hash
from examples.pgct_probe_diagnostics import common_probe_response, finite_horizon_returns, js_divergence

REF = C.REFERENCE_EXPERIMENT
MODELS = {
    # name: (checkpoint, environment source checkpoint, canonical bank)
    "stage03_init_e1": (C.STAGE03, REF / "e1_ippo_s1/final.pt", REF / "e1_pgct_s1/canonical_probes.pt"),
    "e1_ippo_s1": (REF / "e1_ippo_s1/final.pt", None, REF / "e1_pgct_s1/canonical_probes.pt"),
    "e1_pgct_s1": (REF / "e1_pgct_s1/final.pt", None, REF / "e1_pgct_s1/canonical_probes.pt"),
    "e2_ippo_s1": (REF / "e2_ippo_s1/final.pt", None, REF / "e2_pgct_s1/canonical_probes.pt"),
    "e2_pgct_s1": (REF / "e2_pgct_s1/final.pt", None, REF / "e2_pgct_s1/canonical_probes.pt"),
    "stage04_pgct": (C.STAGE04, None, C.STAGE04.parent / "canonical_probes.pt"),
}
MEASURES = ["canonical_js", "history_js", "occupancy_js", "goal_evidence", "critic_gap", "blend_gain"]
BEHAVIOUR = MEASURES[:4]


def environment_for(checkpoint_path, env_source):
    checkpoint = T._load_checkpoint(str(env_source or checkpoint_path), torch.device("cpu"))
    return checkpoint["config"]["env_id"], dict(checkpoint["effective_env_kwargs"])


def visible(positions, radius):
    """observer x target: target inside observer's (2r+1)^2 semantic window."""
    p = np.asarray(positions)
    seen = np.max(np.abs(p[:, None] - p[None, :]), axis=-1) <= radius
    np.fill_diagonal(seen, False)
    return seen


def record_evidence(evidence, info, seen, team_ids, requested_by_team, goals_by_team):
    """evidence[observer, target, 0/1] += events consistent with same / other team.

    Pickup: the observer's requested_shelf channel shows whether the lifted shelf is
    requested for the observer's own team. Delivery: the observer's relevant_goal
    channel marks its own team goals. Both are inside the observer's local window.
    """
    for event in info.get("pickup_events", []):
        if event.get("return_phase"):
            continue
        j = int(event["agent_id"]) - 1
        for i in np.flatnonzero(seen[:, j]):
            same = int(event["shelf_id"]) in requested_by_team[team_ids[i]]
            evidence[i, j, 0 if same else 1] += 1
    for event in info.get("delivery_events", []):
        j = int(event.get("carrier_id", 0)) - 1
        if j < 0:
            continue
        goal = tuple(int(v) for v in event["goal"])
        for i in np.flatnonzero(seen[:, j]):
            evidence[i, j, 0 if goal in goals_by_team[team_ids[i]] else 1] += 1


@torch.inference_mode()
def run_episode(agents, env_id, env_kwargs, cfg, *, seed, horizon, rng):
    device = torch.device("cpu")
    with T.make_env(env_id, env_kwargs, cfg["observation_format"]) as env:
        obs, _ = env.reset(seed=seed)
        base = env.unwrapped
        team_ids = np.asarray(base.agent_team_ids).copy()
        goals_by_team = [{tuple(int(v) for v in g) for g in goals} for goals in base.goals_by_team]
        height, width = map(int, base.grid_size)
        radius = int(base.sensor_range)
        hidden = [net.initial_hidden(1, device)[0] for net in agents]
        observations, masks_log, rewards_log, sampled_probs = [], [], [], []
        evidence = np.zeros((4, 4, 2))
        occupancy = np.zeros((4, height, width))
        co_visible = np.zeros((4, 4))
        for _ in range(horizon):
            x = np.asarray(T.flatten_multi_agent_obs(env, obs), dtype=np.float32)
            masks = np.asarray(T.local_rware_action_masks(env, 4, 5, enabled=cfg["action_mask"]), dtype=np.float32)
            requested_by_team = [{int(s.id) for s in queue} for queue in base.team_request_queues]
            actions, probs = [], []
            for i, net in enumerate(agents):
                logits, hidden[i] = net._actor_step(torch.as_tensor(x[i]).unsqueeze(0), hidden[i])
                p = net._masked_logits(logits, torch.as_tensor(masks[i])).softmax(-1).numpy()[0].astype(np.float64)
                p /= p.sum()
                probs.append(p)
                actions.append(int(rng.choice(5, p=p)))
            obs, rewards, terminated, truncated, info = env.step(actions)
            positions = [(int(a.x), int(a.y)) for a in base.agents]
            seen = visible(positions, radius)
            co_visible += seen
            record_evidence(evidence, info, seen, team_ids, requested_by_team, goals_by_team)
            for i, (px, py) in enumerate(positions):
                occupancy[i, py, px] += 1
            observations.append(x)
            masks_log.append(masks)
            rewards_log.append(np.asarray(rewards, dtype=np.float64))
            sampled_probs.append(probs)
            if bool(np.any(terminated)) or bool(np.any(truncated)):
                break
    return dict(team_ids=team_ids, observations=np.asarray(observations), masks=np.asarray(masks_log),
                rewards=np.asarray(rewards_log), sampled_probs=np.asarray(sampled_probs),
                evidence=evidence, occupancy=occupancy, co_visible=co_visible,
                ended=bool(np.any(terminated)), events_available="pickup_events" in info)


@torch.inference_mode()
def cross_evaluate(agents, episode):
    """probs[k, t, s], values[k, t, s]: network k on agent s's history with k's own hidden state."""
    x = torch.as_tensor(episode["observations"])
    masks = torch.as_tensor(episode["masks"])
    probs, values = [], []
    for net in agents:
        actor_h, critic_h = net.initial_hidden(4, torch.device("cpu"))
        p_steps, v_steps = [], []
        for t in range(len(x)):
            logits, actor_h = net._actor_step(x[t], actor_h)
            value, critic_h = net._critic_step(x[t], critic_h)
            p_steps.append(net._masked_logits(logits, masks[t]).softmax(-1).numpy())
            v_steps.append(value.numpy().reshape(4))
        probs.append(np.stack(p_steps))
        values.append(np.stack(v_steps))
    probs, values = np.stack(probs).astype(np.float64), np.stack(values).astype(np.float64)
    probs /= probs.sum(-1, keepdims=True)
    own = np.stack([probs[s, :, s] for s in range(4)], axis=1)
    # Replaying a network on its own history must reproduce the sampled policy.
    np.testing.assert_allclose(own, episode["sampled_probs"], atol=1e-5)
    return probs, values


def occupancy_js(occupancy):
    flat = occupancy.reshape(4, -1)
    flat = flat / flat.sum(1, keepdims=True)
    return np.asarray([js_divergence(flat[i], flat[j]) for i, j in C.PAIRS])


def episode_scores(episode, probs, values, gamma, early_steps):
    """Six pair scores per measure; larger = predicted more likely to be teammates."""
    history = []
    for i, j in C.PAIRS:
        # Each pair is compared in both agents' own situations.
        history.append(np.mean([js_divergence(probs[i, :, s], probs[j, :, s]).mean() for s in (i, j)]))
    returns = finite_horizon_returns(episode["rewards"], gamma)
    early = slice(0, min(early_steps, len(returns)))
    own_mse = np.asarray([np.mean((values[i, early, i] - returns[early, i]) ** 2) for i in range(4)])
    gap, blend, donor_excess = [], [], []
    for i, j in C.PAIRS:
        gap.append(np.mean([np.mean((values[r, :, r] - values[d, :, r]) ** 2) for r, d in ((i, j), (j, i))]))
        gains, excess = [], []
        for r, d in ((i, j), (j, i)):
            blended = (values[r, early, r] + values[d, early, r]) / 2
            gains.append(own_mse[r] - np.mean((blended - returns[early, r]) ** 2))
            excess.append(np.mean((values[d, early, r] - returns[early, r]) ** 2) - own_mse[r])
        blend.append(np.mean(gains))
        donor_excess.append(np.mean(excess))
    counts = episode["evidence"]
    same = np.asarray([counts[i, j, 0] + counts[j, i, 0] for i, j in C.PAIRS])
    other = np.asarray([counts[i, j, 1] + counts[j, i, 1] for i, j in C.PAIRS])
    return dict(history_js=-np.asarray(history), occupancy_js=-occupancy_js(episode["occupancy"]),
                goal_evidence=evidence_score(same, other), critic_gap=-np.asarray(gap),
                blend_gain=np.asarray(blend)), dict(evidence_same=same, evidence_other=other,
                                                    donor_excess_mse=np.asarray(donor_excess),
                                                    own_early_mse=own_mse)


def evidence_score(same, other):
    """Beta(1,1) posterior mean of 'consistent with my team' minus 0.5; 0 without evidence."""
    same, other = np.asarray(same, dtype=np.float64), np.asarray(other, dtype=np.float64)
    return (same + 1) / (same + other + 2) - 0.5


def measure_statistics(scores, team_ids, rng, n_boot=2000):
    """scores[episode, pair]; dependent samples, so AUROC/CI are descriptive."""
    positive = np.broadcast_to(C.same_team_mask(team_ids), scores.shape)
    per_episode_gap = scores[:, positive[0]].mean(1) - scores[:, ~positive[0]].mean(1)
    boot = [per_episode_gap[rng.integers(0, len(per_episode_gap), len(per_episode_gap))].mean()
            for _ in range(n_boot)]
    return dict(pair_mean=dict(zip(C.PAIR_LABELS, scores.mean(0).tolist())),
                same_mean=float(scores[positive].mean()), cross_mean=float(scores[~positive].mean()),
                same_minus_cross=float(per_episode_gap.mean()),
                same_minus_cross_ci95=[float(np.quantile(boot, .025)), float(np.quantile(boot, .975))],
                episodes_same_higher=float(np.mean(per_episode_gap > 0)),
                auroc_episode_pairs=C.auroc(scores, positive),
                teammate_recovery=float(np.mean([C.teammate_recovery(row, team_ids) for row in scores])))


def accumulated_evidence(extras, team_ids, checkpoints=(1, 2, 5, 10, 20)):
    same = np.cumsum([e["evidence_same"] for e in extras], axis=0)
    other = np.cumsum([e["evidence_other"] for e in extras], axis=0)
    positive = C.same_team_mask(team_ids)
    rows = []
    for k in checkpoints:
        if k > len(extras):
            break
        score = evidence_score(same[k - 1], other[k - 1])
        rows.append(dict(episodes=k, pairs_with_evidence=int(np.sum(same[k - 1] + other[k - 1] > 0)),
                         auroc_pairs=C.auroc(score, positive),
                         teammate_recovery=C.teammate_recovery(score, team_ids),
                         same_events=same[k - 1].tolist(), other_events=other[k - 1].tolist()))
    return rows


def evaluate_model(name, args, out_dir):
    checkpoint, env_source, bank_path = MODELS[name]
    agents, metadata = load_model(checkpoint)
    env_id, env_kwargs = environment_for(checkpoint, env_source)
    cfg = metadata["config"]
    before = model_hash(agents)
    bank = torch.load(bank_path, map_location="cpu", weights_only=False)["sequences"]
    canonical = np.asarray(common_probe_response(agents, bank, torch.device("cpu"))["mean_js"])
    canonical_pairs = -np.asarray([canonical[i, j] for i, j in C.PAIRS])
    rng = np.random.default_rng(args.seed)
    scores = {m: [] for m in MEASURES}
    extras, team_ids, endings, coverage = [], None, [], []
    for e in range(args.episodes):
        episode = run_episode(agents, env_id, env_kwargs, cfg, seed=args.seed + e, horizon=args.horizon, rng=rng)
        if not episode["events_available"]:
            raise RuntimeError("Environment info has no pickup/delivery events (reveal_team_info is off).")
        if team_ids is None:
            team_ids = episode["team_ids"]
        elif not np.array_equal(team_ids, episode["team_ids"]):
            raise RuntimeError("Team assignment changed across evaluation episodes.")
        probs, values = cross_evaluate(agents, episode)
        row, extra = episode_scores(episode, probs, values, cfg["gamma"], args.early_steps)
        row["canonical_js"] = canonical_pairs
        for m in MEASURES:
            scores[m].append(row[m])
        extras.append(extra)
        endings.append(dict(length=len(episode["rewards"]), terminated=episode["ended"]))
        coverage.append(np.asarray([episode["co_visible"][i, j] for i, j in C.PAIRS]) / len(episode["rewards"]))
        print(f"[{name}] episode {e + 1}/{args.episodes}", flush=True)
    if before != model_hash(agents):
        raise RuntimeError("Model weights changed during frozen evaluation.")
    scores = {m: np.asarray(v) for m, v in scores.items()}
    stats_rng = np.random.default_rng(20261013)
    result = dict(checkpoint=str(Path(checkpoint).resolve()), env_steps=metadata["env_steps"], env_id=env_id,
                  env_source=str(Path(env_source or checkpoint).resolve()), canonical_bank=str(Path(bank_path).resolve()),
                  canonical_bank_size=int(len(bank)), team_ids=team_ids.tolist(),
                  env_kwargs={k: env_kwargs.get(k) for k in ("shelf_rows", "shelf_columns", "column_height",
                                                             "shelf_soft_zone_ratio", "team_reward_mode",
                                                             "goal_team_mode", "request_queue_size_per_team")},
                  weights_sha256=before, weights_unchanged=True, endings=endings,
                  mean_co_visible_fraction=dict(zip(C.PAIR_LABELS, np.mean(coverage, 0).tolist())),
                  measures={m: measure_statistics(scores[m], team_ids, stats_rng) for m in MEASURES},
                  evidence_accumulation=accumulated_evidence(extras, team_ids),
                  donor_excess_mse=dict(zip(C.PAIR_LABELS, np.mean([e["donor_excess_mse"] for e in extras], 0).tolist())),
                  own_early_mse=np.mean([e["own_early_mse"] for e in extras], 0).tolist())
    np.savez_compressed(out_dir / f"{name}_scores.npz", team_ids=team_ids,
                        **{m: scores[m] for m in MEASURES},
                        evidence_same=np.asarray([e["evidence_same"] for e in extras]),
                        evidence_other=np.asarray([e["evidence_other"] for e in extras]))
    return result


def plot(results, path, horizon):
    names = list(results)
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.8), dpi=140, constrained_layout=True)
    width = .8 / len(MEASURES)
    x = np.arange(len(names))
    for k, measure in enumerate(MEASURES):
        axes[0].bar(x + k * width, [results[n]["measures"][measure]["auroc_episode_pairs"] for n in names],
                    width, label=measure)
        axes[1].bar(x + k * width, [results[n]["measures"][measure]["teammate_recovery"] for n in names],
                    width, label=measure)
    for ax, title, chance in ((axes[0], "Same-vs-cross AUROC (episode x pair)", .5),
                              (axes[1], "Teammate is top-scored partner", 1 / 3)):
        ax.axhline(chance, color="k", lw=.8, ls="--")
        ax.set_xticks(x + .4 - width / 2, names, rotation=25, ha="right")
        ax.set_ylim(0, 1.02)
        ax.set_title(title)
    axes[0].legend(fontsize=7, ncol=2)
    for n in names:
        rows = results[n]["evidence_accumulation"]
        axes[2].plot([r["episodes"] * horizon for r in rows], [r["teammate_recovery"] for r in rows], "o-", label=n)
    axes[2].axhline(1 / 3, color="k", lw=.8, ls="--")
    axes[2].set_xscale("log")
    axes[2].set_xlabel("Accumulated environment steps")
    axes[2].set_ylabel("Teammate recovery")
    axes[2].set_title("Local goal evidence accumulated over time")
    axes[2].legend(fontsize=7)
    fig.savefig(path)
    plt.close(fig)


def write_report(summary, path):
    lines = ["# Experiment A — team identifiability and consensus premise", "",
             f"Generated {summary['created']}. Frozen models, stochastic masked own-policy rollouts, "
             f"{summary['episodes']}x{summary['horizon']} steps, reset seeds {summary['seed']}–"
             f"{summary['seed'] + summary['episodes'] - 1}. Early-window MC targets use the first "
             f"{summary['early_steps']} steps (finite horizon, zero tail).", "",
             "AUROC and recovery use dependent episode x pair samples and only 2 same / 4 cross pairs; "
             "they are descriptive, not significance tests. canonical_js is one model-level value per pair.", ""]
    header = "| Model | " + " | ".join(MEASURES) + " |"
    for key, title in (("auroc_episode_pairs", "AUROC (same > cross)"), ("teammate_recovery", "Teammate recovery (chance .333)"),
                       ("same_minus_cross", "Same − cross score (larger = teammates scored as more similar/compatible)")):
        lines += [f"## {title}", "", header, "|---|" + "---:|" * len(MEASURES)]
        for name, result in summary["models"].items():
            cells = []
            for m in MEASURES:
                value = result["measures"][m][key]
                cells.append("—" if value is None else f"{value:.3f}")
            lines.append(f"| {name} | " + " | ".join(cells) + " |")
        lines.append("")
    lines += ["## Goal evidence accumulated over episodes", "", "| Model | episodes | pairs with evidence | recovery | AUROC |",
              "|---|---:|---:|---:|---:|"]
    for name, result in summary["models"].items():
        for row in result["evidence_accumulation"]:
            auc = "—" if row["auroc_pairs"] is None else f"{row['auroc_pairs']:.3f}"
            lines.append(f"| {name} | {row['episodes']} | {row['pairs_with_evidence']}/6 | "
                         f"{row['teammate_recovery']:.3f} | {auc} |")
    lines += ["", "## Critic premise per pair (mean over episodes)", "",
              "| Model | " + " | ".join(C.PAIR_LABELS) + " |", "|---|" + "---:|" * 6]
    for name, result in summary["models"].items():
        lines.append(f"| {name} blend_gain | " + " | ".join(
            f"{v:+.4f}" for v in result["measures"]["blend_gain"]["pair_mean"].values()) + " |")
        lines.append(f"| {name} donor_excess_mse | " + " | ".join(
            f"{v:+.4f}" for v in result["donor_excess_mse"].values()) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--models", nargs="+", choices=list(MODELS), default=list(MODELS))
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--horizon", type=int, default=500)
    parser.add_argument("--early-steps", type=int, default=200)
    parser.add_argument("--seed", type=int, default=5200001)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    if min(args.episodes, args.horizon, args.early_steps, args.threads) <= 0:
        parser.error("Episodes, horizon, early steps and threads must be positive.")
    torch.set_num_threads(args.threads)
    out = (args.output_dir or C.ROOT / "output/latent_team" / ("exp_a_" + datetime.now().strftime("%Y%m%d_%H%M%S"))).resolve()
    out.mkdir(parents=True, exist_ok=False)
    summary = dict(created=datetime.now().astimezone().isoformat(timespec="seconds"), episodes=args.episodes,
                   horizon=args.horizon, early_steps=args.early_steps, seed=args.seed,
                   score_convention="larger = predicted more likely teammates; JS/gap measures are negated",
                   models={})
    for name in args.models:
        summary["models"][name] = evaluate_model(name, args, out)
        (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    plot(summary["models"], out / "identifiability.png", args.horizon)
    write_report(summary, out / "REPORT.md")
    print(f"Experiment A: {out}", flush=True)


if __name__ == "__main__":
    main()
