import itertools

from gymnasium import register

from rware.warehouse import RewardType, ObservationType
from rware.multi_team_warehouse import MultiTeamWarehouse, TeamRewardMode
from rware.multi_team_grid import MultiTeamGrid

_sizes = {
    "tiny": (1, 3),
    "small": (2, 3),
    "medium": (2, 5),
    "large": (3, 5),
}

_difficulty = {"-easy": 2, "": 1, "-hard": 0.5}

_perms = itertools.product(
    _sizes.keys(),
    _difficulty,
    range(1, 20),
)

for size, diff, agents in _perms:
    # normal tasks
    register(
        id=f"rware-{size}-{agents}ag{diff}-v2",
        entry_point="rware.warehouse:Warehouse",
        kwargs={
            "column_height": 8,
            "shelf_rows": _sizes[size][0],
            "shelf_columns": _sizes[size][1],
            "n_agents": agents,
            "msg_bits": 0,
            "sensor_range": 1,
            "request_queue_size": int(agents * _difficulty[diff]),
            "max_inactivity_steps": None,
            "max_steps": 500,
            "reward_type": RewardType.INDIVIDUAL,
        },
    )


_multi_team_rware_setups = [
    ("tiny", 4, 2),
    ("small", 6, 2),
    ("small", 6, 3),
    ("medium", 8, 2),
    ("medium", 10, 3),
]

for size, agents, teams in _multi_team_rware_setups:
    per_team_requests = max(1, agents // teams)
    register(
        id=f"rware-multiteam-{size}-{agents}ag-{teams}teams-v0",
        entry_point="rware.multi_team_warehouse:MultiTeamWarehouse",
        kwargs={
            "column_height": 8,
            "shelf_rows": _sizes[size][0],
            "shelf_columns": _sizes[size][1],
            "n_agents": agents,
            "msg_bits": 0,
            "sensor_range": 1,
            "request_queue_size": per_team_requests * teams,
            "request_queue_size_per_team": per_team_requests,
            "max_inactivity_steps": None,
            "max_steps": 500,
            "reward_type": RewardType.INDIVIDUAL,
            "n_teams": teams,
            "team_reward_mode": TeamRewardMode.TEAM,
        },
    )


_multi_team_grid_setups = [
    ("small", (12, 12), 4, 2, 2, 0.03, 100),
    ("main", (15, 15), 6, 2, 3, 0.05, 200),
    ("main", (15, 15), 6, 3, 2, 0.05, 200),
    ("large", (20, 20), 8, 2, 4, 0.07, 300),
    ("large", (24, 24), 10, 3, 3, 0.07, 300),
]

for label, grid_size, agents, teams, targets, obstacle_density, max_steps in _multi_team_grid_setups:
    register(
        id=f"mtgrid-{label}-{agents}ag-{teams}teams-v0",
        entry_point="rware.multi_team_grid:MultiTeamGrid",
        kwargs={
            "grid_size": grid_size,
            "n_agents": agents,
            "n_teams": teams,
            "msg_bits": 0,
            "sensor_range": 2,
            "targets_per_team": targets,
            "obstacle_density": obstacle_density,
            "target_layout": "zones",
            "team_reward_mode": TeamRewardMode.TEAM,
            "max_steps": max_steps,
        },
    )


def image_registration():
    _observation_type = {
        "": ObservationType.FLATTENED,
        "-img": ObservationType.IMAGE,
        "-imgdict": ObservationType.IMAGE_DICT,
    }
    _image_directional = {"": True, "-Nd": False}
    _perms = itertools.product(
        _sizes.keys(),
        _difficulty,
        _observation_type,
        _image_directional,
        range(1, 20),
    )
    for size, diff, obs_type, directional, agents in _perms:
        if obs_type == "" and directional == "":
            # already registered before
            continue
        if directional != "" and obs_type == "":
            # directional values should only be used with image observations
            continue
        register(
            id=f"rware{obs_type}{directional}-{size}-{agents}ag{diff}-v2",
            entry_point="rware.warehouse:Warehouse",
            kwargs={
                "column_height": 8,
                "shelf_rows": _sizes[size][0],
                "shelf_columns": _sizes[size][1],
                "n_agents": agents,
                "msg_bits": 0,
                "sensor_range": 1,
                "request_queue_size": int(agents * _difficulty[diff]),
                "max_inactivity_steps": None,
                "max_steps": 500,
                "reward_type": RewardType.INDIVIDUAL,
                "observation_type": _observation_type[obs_type],
                "image_observation_directional": _image_directional[directional],
            },
        )


def full_registration():
    _observation_type = {"": ObservationType.FLATTENED, "-img": ObservationType.IMAGE}
    _sensor_ranges = {f"-{sight}s": sight for sight in range(2, 6)}
    _sensor_ranges[""] = 1
    _image_directional = {"": True, "-Nd": False}
    _perms = itertools.product(
        _sizes.keys(),
        _difficulty,
        _observation_type,
        _sensor_ranges,
        _image_directional,
        range(1, 20),
        range(1, 16),
    )
    for (
        size,
        diff,
        obs_type,
        sensor_range,
        directional,
        agents,
        column_height,
    ) in _perms:
        # normal tasks with modified column height
        if directional != "" and obs_type == "":
            # directional should only be used with image observations
            continue
        register(
            id=f"rware{obs_type}{directional}{sensor_range}-{size}-{column_height}h-{agents}ag{diff}-v2",
            entry_point="rware.warehouse:Warehouse",
            kwargs={
                "column_height": column_height,
                "shelf_rows": _sizes[size][0],
                "shelf_columns": _sizes[size][1],
                "n_agents": agents,
                "msg_bits": 0,
                "sensor_range": _sensor_ranges[sensor_range],
                "request_queue_size": int(agents * _difficulty[diff]),
                "max_inactivity_steps": None,
                "max_steps": 500,
                "reward_type": RewardType.INDIVIDUAL,
                "observation_type": _observation_type[obs_type],
                "image_observation_directional": _image_directional[directional],
            },
        )

    _rewards = {
        "indiv": RewardType.INDIVIDUAL,
        "global": RewardType.GLOBAL,
        "twostage": RewardType.TWO_STAGE,
    }

    _perms = itertools.product(
        range(1, 5),
        range(3, 10, 2),
        range(1, 16),
        range(1, 20),
        range(1, 20),
        _rewards,
        _observation_type,
        _sensor_ranges,
        _image_directional,
    )

    for (
        rows,
        cols,
        column_height,
        agents,
        req,
        rew,
        obs_type,
        sensor_range,
        directional,
    ) in _perms:
        register(
            id=f"rware{obs_type}{directional}{sensor_range}-{rows}x{cols}-{column_height}h-{agents}ag-{req}req-{rew}-v2",
            entry_point="rware.warehouse:Warehouse",
            kwargs={
                "column_height": column_height,
                "shelf_rows": rows,
                "shelf_columns": cols,
                "n_agents": agents,
                "msg_bits": 0,
                "sensor_range": _sensor_ranges[sensor_range],
                "request_queue_size": req,
                "max_inactivity_steps": None,
                "max_steps": 500,
                "reward_type": _rewards[rew],
                "observation_type": _observation_type[obs_type],
                "image_observation_directional": _image_directional[directional],
            },
        )
