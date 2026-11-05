"""T185 convergence rule of latent_team/main_check.py (read-only progress check). No training."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from latent_team import main_check as M


def rows(team, agents, start=4.0e6, end=5.1e6, step=51_200):  # a verdict needs an evaluation past the milestone
    out, s = [], start + step
    while s <= end:
        out.append((s, team(s), agents(s)))
        s += step
    return out


def test_flat_curves_converge_and_rising_ones_do_not():
    flat = rows(lambda s: 20.0, lambda s: [0.0, 5.0, 5.0, 4.0, 3.0, 3.0])
    v = M.convergence(flat, 5_000_000)
    assert v["converged"] and v["team_a"] == v["team_b"] == 20.0
    rising = rows(lambda s: 10.0 + 4e-6 * (s - 4e6), lambda s: [0.0] + [2.0 + 8e-7 * (s - 4e6)] * 5)
    assert not M.convergence(rising, 5_000_000)["converged"]


def test_one_agent_starting_late_keeps_the_job_running():
    late = rows(lambda s: 20.0, lambda s: [0.0 if s <= 4.5e6 else 1.0, 5.0, 5.0, 4.0, 3.0, 3.0])
    v = M.convergence(late, 5_000_000)
    assert not v["converged"] and v["worst_agent"] == 1 and v["worst_agent_change"] == 1.0


def test_unreached_milestone_or_sparse_evaluations_give_no_verdict():
    short = rows(lambda s: 20.0, lambda s: [1.0] * 6, end=4.9e6)
    assert M.convergence(short, 5_000_000) is None
    sparse = rows(lambda s: 20.0, lambda s: [1.0] * 6, step=200_000)
    assert M.convergence(sparse, 5_000_000) is None


def test_small_noise_within_tolerance_converges():
    noisy = rows(lambda s: 20.0 + (0.6 if int(s / 51_200) % 2 else -0.6), lambda s: [0.1, 5, 5, 4, 3, 3])
    assert M.convergence(noisy, 5_000_000)["converged"]
