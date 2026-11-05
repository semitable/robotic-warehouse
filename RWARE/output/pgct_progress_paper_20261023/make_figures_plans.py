"""Env/probe pictures of the running and planned experiments (2026-10-23).  Read-only; no training.

  fig_now_transfer   running main stage 1 (corner/team, penalty .005): who sends a critic to whom under IPPO,
                     Oracle-team and All-peer, on a real mid-episode state of the 1M seed-1 policies
  fig_now_probes     what the two probes see in that cell: reward-coupling responses (agents x events) and the
                     policy-probe JS matrix (canonical bank), 1M seed-1 policies
  fig_next_gates     planned stage 2 on the same state: donor arcs chosen by PGCT-reward, PGCT-policy (D < tau)
                     and Random-gated (same number of arcs as PGCT-policy)
  fig_next_probes    planned corner/individual cell (E1-F role): gate probes with the shortest-path first action
                     for a left-goal and a right-goal agent, and how many gate probes separate the teams under
                     corner vs adjacent goals (BFS on the map, loaded agents cannot enter shelf cells)
  fig_next_lbf       planned second environment: concept sketch of a two-team Level-Based Foraging variant
                     (not implemented)

    python output/pgct_progress_paper_20261023/make_figures_plans.py
"""
import json
import sys
from collections import deque
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.patches import Circle, FancyArrowPatch, Rectangle

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "latent_team"))
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "output/pen005_analysis_20261023"))  # dominance_diag.js_matrix
from examples import train_recurrent_ippo_consensus as T  # noqa: E402
from reward_correlation_baseline import load_model  # noqa: E402
import make_figures as MF  # noqa: E402  (decode / draw_probe helpers)

OUT = HERE / "figures"
RED, BLUE = MF.RED, MF.BLUE
PEN = ROOT / "runs/pilot_20261022_112521_b_e1f6_comm3_pen005"
RUN1 = PEN / "b_e1f6_comm3_pen005_ippo_s1"
COMM, TAU = 3, 0.12
DIRV = {0: (0, -1), 1: (0, 1), 2: (-1, 0), 3: (1, 0)}  # up, down, left, right
WRAP = [0, 3, 1, 2]  # up, right, down, left (clockwise)


def save(fig, name):
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(OUT / f"{name}.png", dpi=130, bbox_inches="tight")
    plt.close(fig)


# ------------------------------------------------------------------ a real mid-episode state
@torch.inference_mode()
def mid_episode_state(steps=180, seed=5000001):
    agents, _ = load_model(RUN1 / "final.pt")
    c = T._load_checkpoint(str(RUN1 / "final.pt"), torch.device("cpu"))
    cfg, kw = c["config"], dict(c["effective_env_kwargs"])
    rng = np.random.default_rng(7)
    with T.make_env(cfg["env_id"], kw, cfg["observation_format"]) as env:
        obs, _ = env.reset(seed=seed)
        u = env.unwrapped
        n = len(u.agents)
        hid = [net.initial_hidden(1, torch.device("cpu"))[0] for net in agents]
        for _ in range(steps):
            x = np.asarray(T.flatten_multi_agent_obs(env, obs), dtype=np.float32)
            m = np.asarray(T.local_rware_action_masks(env, n, 5, enabled=True), dtype=np.float32)
            acts = []
            for i, net in enumerate(agents):
                lg, hid[i] = net._actor_step(torch.as_tensor(x[i]).unsqueeze(0), hid[i])
                p = net._masked_logits(lg, torch.as_tensor(m[i])).softmax(-1).numpy()[0].astype(np.float64)
                acts.append(int(rng.choice(5, p=p / p.sum())))
            obs, *_ = env.step(acts)
        H, W = map(int, u.grid_size)
        req = {int(s.id): t for t, q in enumerate(u.team_request_queues) for s in q}
        carried = {a.carrying_shelf.id for a in u.agents if a.carrying_shelf is not None}
        st = dict(H=H, W=W, shelves=[(s.x, s.y, s.id) for s in u.shelfs if s.id not in carried], req=req,
                  agents=[(a.x, a.y) for a in u.agents], dirs=[int(a.dir.value) for a in u.agents],
                  loaded=[a.carrying_shelf is not None for a in u.agents],
                  teams=[int(t) for t in np.asarray(u.agent_team_ids)],
                  goals=[tuple(g[0]) for g in kw["team_goal_positions"]], sensor=int(kw.get("sensor_range", 5)))
    return agents, st


def neighbours(st):
    n = len(st["agents"])
    P = np.array(st["agents"])
    return [(i, j) for i in range(n) for j in range(n) if i != j and np.abs(P[i] - P[j]).max() <= COMM]


def draw_state(ax, st, title, arcs=(), dominant=None, show_comm=None):
    MF.draw_map(ax, {**st, "shelves": st["shelves"]}, title)
    for k, ((x, y), d, ld) in enumerate(zip(st["agents"], st["dirs"], st["loaded"])):
        dx, dy = DIRV[d]
        ax.add_patch(FancyArrowPatch((x, y), (x + 0.55 * dx, y + 0.55 * dy), arrowstyle="-|>", mutation_scale=7, lw=0.8, color="0.2"))
        if ld:
            ax.add_patch(Rectangle((x + 0.12, y + 0.12), 0.32, 0.32, fc="#ffbf00", ec="0.3", lw=0.5, zorder=4))
    if dominant is not None:
        x, y = st["agents"][dominant]
        ax.plot([x], [y - 0.55], marker="*", color="#ffbf00", ms=9, mec="0.2", mew=0.5, zorder=6)
    if show_comm is not None:
        x, y = st["agents"][show_comm]
        ax.add_patch(Rectangle((x - COMM - 0.5, y - COMM - 0.5), 2 * COMM + 1, 2 * COMM + 1, fc="none", ec="0.4", ls=":", lw=0.9))
    for j, i, col in arcs:  # donor j -> receiver i
        (xj, yj), (xi, yi) = st["agents"][j], st["agents"][i]
        ax.add_patch(FancyArrowPatch((xj, yj), (xi, yi), arrowstyle="-|>", mutation_scale=8, lw=1.3, color=col,
                                     connectionstyle="arc3,rad=0.18", shrinkA=6, shrinkB=6, zorder=5, alpha=0.85))


# ------------------------------------------------------------------ 1. running: transfer graphs
def fig_now_transfer(st):
    nb = neighbours(st)
    teams = st["teams"]
    graphs = [("IPPO: no transfer", []),
              ("Oracle-team: same-team neighbours", [(j, i, "#2ca02c") for i, j in nb if teams[i] == teams[j]]),
              ("All-peer: every neighbour", [(j, i, "#2ca02c" if teams[i] == teams[j] else "#d62728") for i, j in nb])]
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 4.3))
    for ax, (title, arcs) in zip(axes, graphs):
        draw_state(ax, st, title, arcs, dominant=1, show_comm=1 if not arcs else None)
    fig.text(0.5, 0.0, "Running main stage 1: corner goals, team reward, penalty .005, IPPO / Oracle-team / All-peer x seeds 1-3, 5M.  "
             "State: 1M seed-1 policies after 180 steps.  Arrow = critic distilled from donor to receiver (green: same team, red: other team);\n"
             "star = the agent that learned first (agent 2); dotted box = its communication range (3); yellow square = carried shelf.",
             ha="center", fontsize=7.5)
    save(fig, "fig_now_transfer")


# ------------------------------------------------------------------ 2. running: what the probes see
@torch.inference_mode()
def fig_now_probes(agents, st):
    res = json.loads((ROOT / "output/pen005_analysis_20261023/results.json").read_text(encoding="utf-8"))["auroc"]["s1 final.pt"]
    R = np.array(res["responses"], dtype=float)[:, :2]
    C = np.array(res["seen"], dtype=float)[:, :2]
    bank = torch.load(PEN / "canonical_bank_b_e1f6_comm3_pen005_s1.pt", map_location="cpu", weights_only=False)["sequences"][:, 0].float()
    import dominance_diag as DD  # noqa: E402
    D, _ = DD.js_matrix(agents, bank)
    teams = st["teams"]
    labels = [f"{k + 1}{'R' if t == 0 else 'B'}" for k, t in enumerate(teams)]
    order = [k for k in range(6) if teams[k] == 0] + [k for k in range(6) if teams[k] == 1]
    fig, (a, b, c) = plt.subplots(1, 3, figsize=(10.5, 3.3), gridspec_kw=dict(width_ratios=[0.8, 1.1, 1.1]))
    im = a.imshow(np.where(C[order] >= 1, R[order], np.nan), cmap="Greens", vmin=-0.1, vmax=1.1, aspect="auto")
    for r, k in enumerate(order):
        for e in range(2):
            a.text(e, r, "-" if C[k, e] < 1 else f"{R[k, e]:.2f}\n(n={int(C[k, e])})", ha="center", va="center", fontsize=6.5)
    a.set_xticks([0, 1]); a.set_xticklabels([f"delivery at\n{st['goals'][0]}", f"delivery at\n{st['goals'][1]}"], fontsize=7)
    a.set_yticks(range(6)); a.set_yticklabels([labels[k] for k in order], fontsize=7)
    a.set_title("(a) reward-coupling response rho_i(e)\n(team AUROC 1.00)", fontsize=8)
    Do = D[np.ix_(order, order)]
    im2 = b.imshow(Do, cmap="Purples", vmin=0, vmax=max(0.3, float(Do.max())))
    for r in range(6):
        for q in range(6):
            b.text(q, r, f"{Do[r, q]:.2f}", ha="center", va="center", fontsize=6, color="white" if Do[r, q] > 0.2 else "black")
    b.axhline(2.5, color="0.2", lw=0.8); b.axvline(2.5, color="0.2", lw=0.8)
    b.set_xticks(range(6)); b.set_xticklabels([labels[k] for k in order], fontsize=7)
    b.set_yticks(range(6)); b.set_yticklabels([labels[k] for k in order], fontsize=7)
    b.set_title("(b) policy probe: mean JS (canonical bank)\nblocks = same team; gate opens if < 0.12", fontsize=8)
    pairs = [(i, j) for i in range(6) for j in range(i + 1, 6)]
    same = [D[i, j] for i, j in pairs if teams[i] == teams[j]]
    cross = [D[i, j] for i, j in pairs if teams[i] != teams[j]]
    c.scatter(np.zeros(len(same)) + np.random.default_rng(0).uniform(-0.08, 0.08, len(same)), same, color="#2ca02c", label="same team")
    c.scatter(np.ones(len(cross)) + np.random.default_rng(1).uniform(-0.08, 0.08, len(cross)), cross, color="#d62728", label="other team")
    c.axhline(TAU, color="0.3", ls=":", lw=1); c.text(1.25, TAU, "tau", fontsize=7, va="center")
    c.set_xticks([0, 1]); c.set_xticklabels(["same-team pairs", "cross-team pairs"], fontsize=7.5); c.set_xlim(-0.5, 1.5)
    c.set_ylabel("mean JS"); c.set_title("(c) policy-probe distances by pair type", fontsize=8)
    fig.text(0.5, -0.04, "Running cell (corner goals, team reward, .005), 1M seed-1 policies.  Agent labels: number + team (R/B).  "
             "The reward-coupling probe separates the teams; the policy probe does not (yet).", ha="center", fontsize=7.5)
    fig.tight_layout(); save(fig, "fig_now_probes")
    return D


# ------------------------------------------------------------------ 3. planned stage 2: gate choices
def fig_next_gates(st, D):
    res = json.loads((ROOT / "output/pen005_analysis_20261023/results.json").read_text(encoding="utf-8"))["auroc"]["s1 final.pt"]
    R = np.array(res["responses"], dtype=float)[:, :2]
    C = np.array(res["seen"], dtype=float)[:, :2]
    nb = neighbours(st)
    teams = st["teams"]

    def reward_same(i, j):
        ok = [e for e in range(2) if C[i, e] >= 1 and C[j, e] >= 1]
        if not ok or all(abs(R[i, e]) < 0.25 and abs(R[j, e]) < 0.25 for e in ok):
            return None  # abstain: no common event, or both responses near zero
        return float(np.mean([abs(R[i, e] - R[j, e]) for e in ok])) < 0.5

    col = lambda i, j: "#2ca02c" if teams[i] == teams[j] else "#d62728"
    rew = [(j, i, col(i, j)) for i, j in nb if reward_same(i, j)]
    pol = [(j, i, col(i, j)) for i, j in nb if D[i, j] < TAU]
    rng = np.random.default_rng(3)
    pick = rng.choice(len(nb), size=min(len(pol), len(nb)), replace=False) if nb else []
    rnd = [(nb[k][1], nb[k][0], col(nb[k][0], nb[k][1])) for k in pick]
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 4.3))
    for ax, (title, arcs) in zip(axes, [("PGCT-reward (planned)", rew), ("PGCT-policy: D < tau", pol), ("Random-gated: same #arcs", rnd)]):
        draw_state(ax, st, f"{title}  [{sum(a[2] == '#2ca02c' for a in arcs)} same / {sum(a[2] == '#d62728' for a in arcs)} cross]", arcs, dominant=1)
    fig.text(0.5, 0.0, "Planned stage 2 on the same state (illustration with 1M seed-1 policies; the real gates use running statistics).  "
             "PGCT-reward abstains when both responses are near zero.\nGreen arcs = same-team donors, red arcs = other-team donors.",
             ha="center", fontsize=7.5)
    save(fig, "fig_next_gates")


# ------------------------------------------------------------------ 4. planned corner/indiv: discriminative gate probes
def blocked_map(st):
    B = np.zeros((st["H"], st["W"]), dtype=bool)
    for x, y, _ in st["shelves"]:
        B[y, x] = True
    return B


def first_actions(B, start, d0, goal):
    H, W = B.shape
    def nxt(s):
        x, y, d = s
        out = []
        dx, dy = DIRV[d]
        nx, ny = x + dx, y + dy
        if 0 <= nx < W and 0 <= ny < H and not B[ny, nx]:
            out.append(("FWD", (nx, ny, d)))
        out.append(("LEFT", (x, y, WRAP[(WRAP.index(d) - 1) % 4])))
        out.append(("RIGHT", (x, y, WRAP[(WRAP.index(d) + 1) % 4])))
        return out
    def dist(s):
        seen, q = {s}, deque([(s, 0)])
        while q:
            cur, k = q.popleft()
            if (cur[0], cur[1]) == goal:
                return k
            for _, n2 in nxt(cur):
                if n2 not in seen:
                    seen.add(n2); q.append((n2, k + 1))
        return 10 ** 9
    opts = [(a, dist(s)) for a, s in nxt((start[0], start[1], d0))]
    best = min(k for _, k in opts)
    return {a for a, k in opts if k == best}


def fig_next_probes():
    bank = torch.load(ROOT / "output/env_view_20261015/probes_v2/probe_bank_v2_e1f_s1.pt", map_location="cpu", weights_only=False)
    gate, gsc = bank["sequences"][:, 0], bank["gate_scenarios"]
    st = MF.env_state(Path(str(MF.RUNS["corner/indiv, .005"]).format(1)))
    B = blocked_map(st)
    H, W = B.shape
    layouts = {"corner": [(0, H - 1), (W - 1, H - 1)], "adjacent": [(W // 2 - 1, H - 1), (W // 2, H - 1)]}
    disc = {}
    table = []
    for name, (gl, gr) in layouts.items():
        n = 0
        for s in gsc:
            B2 = B.copy(); B2[s["y"], s["x"]] = False
            al = first_actions(B2, (s["x"], s["y"]), s["direction"], gl)
            ar = first_actions(B2, (s["x"], s["y"]), s["direction"], gr)
            differ = al.isdisjoint(ar)
            n += differ
            if name == "corner":
                table.append((al, ar, differ))
        disc[name] = n
    pick, used = [], set()
    for k, (_, _, dif) in enumerate(table):
        pos = (gsc[k]["x"], gsc[k]["y"])
        if dif and pos not in used:
            pick.append(k); used.add(pos)
        if len(pick) == 4:
            break
    fig = plt.figure(figsize=(11, 4.8))
    gs = fig.add_gridspec(2, 5, width_ratios=[1.6, 1, 1, 0.25, 1.1])
    for k, i in enumerate(pick):
        s = gsc[i]
        ax = fig.add_subplot(gs[k // 2, 1 + k % 2])
        MF.draw_probe(ax, gate[i], f"gate #{i} at ({s['x']},{s['y']}) dir {s['direction']}")
        al, ar, _ = table[i]
        ax.set_xlabel(f"goal left: {'/'.join(sorted(al))}\ngoal right: {'/'.join(sorted(ar))}", fontsize=7)
    ax = fig.add_subplot(gs[:, 4])
    ax.bar([0, 1], [disc["corner"], disc["adjacent"]], color=[RED, "0.6"], width=0.6)
    for x, v in enumerate([disc["corner"], disc["adjacent"]]):
        ax.text(x, v + 0.6, f"{v}/48", ha="center", fontsize=8)
    ax.set_xticks([0, 1]); ax.set_xticklabels(["corner goals\n(0,10)/(9,10)", "adjacent goals\n(4,10)/(5,10)"], fontsize=7.5)
    ax.set_ylabel("gate probes whose optimal first\naction differs by goal side"); ax.set_ylim(0, 50)
    ax.set_title("condition (a) in principle", fontsize=8)
    axm = fig.add_subplot(gs[:, 0])
    MF.draw_map(axm, st, "planned corner/individual cell (E1-F role), 5M\norange = positions of the probes shown")
    for i in pick:
        s = gsc[i]
        axm.plot([s["x"]], [s["y"]], marker="s", ms=9, mfc="none", mec="#ff7f0e", mew=1.6)
        axm.text(s["x"] + 0.35, s["y"] - 0.3, f"#{i}", fontsize=6.5, color="#ff7f0e")
    fig.text(0.5, -0.02, "Shortest-path first action of a loaded agent (BFS; loaded agents cannot enter shelf cells).  "
             "A learned left-goal and right-goal agent should answer these probes differently if condition (a) holds; "
             "with zero memory and the goal out of view,\nthe only cue is the agent's absolute position and what it has learned about its own goal side.",
             ha="center", fontsize=7.5)
    fig.tight_layout(); save(fig, "fig_next_probes")
    return disc


# ------------------------------------------------------------------ 5. planned second environment (concept)
def fig_next_lbf():
    rng = np.random.default_rng(11)
    fig, axes = plt.subplots(1, 2, figsize=(8.5, 3.9))
    for ax, (title, team_reward) in zip(axes, [("individual reward: only the collector gains", False),
                                                ("team reward: the collector's team gains", True)]):
        n = 8
        ax.add_patch(Rectangle((-0.5, -0.5), n, n, fc="#f7f7f7", ec="0.6"))
        cells = rng.permutation(n * n)[:16]
        foods = [(int(c % n), int(c // n)) for c in cells[:10]]
        for k, (x, y) in enumerate(foods):
            t = k % 2
            ax.add_patch(Rectangle((x - 0.32, y - 0.32), 0.64, 0.64, fc=RED if t == 0 else BLUE, alpha=0.35, ec="0.4"))
            ax.text(x, y, str(int(rng.integers(1, 4))), ha="center", va="center", fontsize=7)
        for k, c in enumerate(cells[10:]):
            x, y = int(c % n), int(c // n)
            ax.add_patch(Circle((x, y), 0.3, fc="white", ec=RED if k % 2 == 0 else BLUE, lw=1.6))
            ax.text(x, y, str(int(rng.integers(1, 3))), ha="center", va="center", fontsize=6.5)
        ax.set_xlim(-0.6, n - 0.4); ax.set_ylim(n - 0.4, -0.6); ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(title, fontsize=8)
    fig.text(0.5, -0.03, "Concept only (not implemented): Level-Based Foraging with two latent teams.  Squares = food of the red / blue team's type "
             "(number = level), circles = agents (number = level).\nEach agent sees 'my food type' team-relatively; teammates are unknown. "
             "Varying food placement (mixed vs team zones) and the reward mode gives the same 2x2 as in RWARE.", ha="center", fontsize=7.5)
    save(fig, "fig_next_lbf")


def main():
    torch.set_num_threads(4)
    agents, st = mid_episode_state()
    fig_now_transfer(st); print("ok now_transfer", flush=True)
    D = fig_now_probes(agents, st); print("ok now_probes", flush=True)
    fig_next_gates(st, D); print("ok next_gates", flush=True)
    disc = fig_next_probes(); print("ok next_probes", disc, flush=True)
    fig_next_lbf(); print("ok next_lbf", flush=True)


if __name__ == "__main__":
    main()
