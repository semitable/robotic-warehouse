"""Minimal environment interface for PGCT.

PGCT needs six separate kinds of information from an environment. Each adapter must keep them apart:

1. Local actor observation   -- what one agent's policy sees (``EnvAdapter.reset/step`` observations).
2. Local critic input        -- in both RWARE and Overcooked this equals the actor observation. Max-GAE
                                 runs a donor critic on the receiver's critic inputs, so every agent's
                                 critic input must live in one shared space (same size and semantics).
3. Private reward            -- one scalar per agent (``rewards[i]``). Never summed or shared.
4. Public probe view         -- ``ProbeProjection.public_view``: the part of a local observation any agent
                                 could also see (physical scene, own pose/held item). It must not contain
                                 team identity, private task information or absolute coordinates.
5. Communication topology    -- ``EnvAdapter.contact_matrix``: who can currently exchange messages.
6. Evaluation-only info      -- ``EnvAdapter.oracle_team_ids``: ground-truth teams. Only the Oracle
                                 baseline and evaluation/audit code may call it; the PGCT gate never does.

The interface is deliberately small: it covers what the RWARE and Overcooked experiments use.
"""
from __future__ import annotations

from typing import Any, Dict, Optional, Protocol, Tuple, runtime_checkable

import numpy as np


@runtime_checkable
class ProbeProjection(Protocol):
    """Maps one agent's local observation to its public reward-probe view (a flat vector)."""

    view_size: int          # length of a public view
    n_actions: int          # number of discrete actions (one-hot in the predictor input)
    view_dtype: Any         # storage dtype of views in the predictor buffers (RWARE: np.uint8)

    def public_view(self, observation: np.ndarray) -> np.ndarray:
        """Public part of one agent's local observation."""

    def mirror(self, before: np.ndarray, actions: np.ndarray,
               after: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Optional symmetry augmentation of probe events (only used when the gate config enables it)."""

    # Optional (not part of the runtime check): ``public_transition(before_obs, action, after_obs)
    # -> (view_before, action, view_after)``. When a projection defines it, the gate uses it instead of two
    # independent ``public_view`` calls, so both views and the action can be expressed in one frame
    # (e.g. relative to the agent's heading before the step). Projections without it behave as before.


class NoMirror:
    """Mixin for projections without a symmetry augmentation."""

    def mirror(self, before, actions, after):
        raise ValueError(f"{type(self).__name__} defines no mirror augmentation; set mirror=False")


@runtime_checkable
class EnvAdapter(Protocol):
    """What the generic recurrent IPPO/PGCT trainer (pgct_core.trainer) needs from an environment."""

    n_agents: int
    obs_dim: int            # local actor observation size (= local critic input size)
    n_actions: int
    projection: ProbeProjection

    def reset(self, seed: Optional[int] = None) -> np.ndarray:
        """Start an episode; returns local observations [n_agents, obs_dim] (float32)."""

    def step(self, actions) -> Tuple[np.ndarray, np.ndarray, bool, bool, Dict[str, Any]]:
        """(observations, individual rewards [n_agents], terminated, truncated, info).

        ``info`` must not contain team labels; ``info['events']`` may hold per-agent counters for logs."""

    def contact_matrix(self) -> np.ndarray:
        """Current communication availability [n_agents, n_agents] (bool, symmetric)."""

    def oracle_team_ids(self) -> np.ndarray:
        """Ground-truth team per agent. Oracle baseline and evaluation code only."""
