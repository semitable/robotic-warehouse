import gym
from .warehouse import Warehouse, RewardType, Action
import itertools

_sizes = {
    "tiny": (1, 3),
    "small": (2, 3),
    "medium": (2, 5),
    "large": (3, 5),
}

_difficulty = {"-easy": 2, "": 1, "-hard": 0.5}

_perms = itertools.product(_sizes.keys(), _difficulty, range(1, 20),)

for size, diff, agents in _perms:
    # normal tasks
    gym.register(
        id=f"rware-{size}-{agents}ag{diff}-v1",
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

def full_registration():
    _perms = itertools.product(_sizes.keys(), _difficulty, range(1, 20), range(2, 11),)
    for size, diff, agents, column_height in _perms:
        # normal tasks with modified column height
        gym.register(
            id=f"rware-{size}-{column_height}h-{agents}ag{diff}-v1",
            entry_point="rware.warehouse:Warehouse",
            kwargs={
                "column_height": column_height,
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

    _perms = itertools.product(
        range(1, 5),
        range(3, 10, 2),
        range(2, 11),
        range(1, 20),
        range(1, 20),
        ["indiv", "global", "twostage"],
    )
    _rewards = {
        "indiv": RewardType.INDIVIDUAL,
        "global": RewardType.GLOBAL,
        "twostage": RewardType.TWO_STAGE,
    }

    for rows, cols, column_height, agents, req, rew in _perms:
        gym.register(
            id=f"rware-{rows}x{cols}-{column_height}h-{agents}ag-{req}req-{rew}-v1",
            entry_point="rware.warehouse:Warehouse",
            kwargs={
                "column_height": column_height,
                "shelf_rows": rows,
                "shelf_columns": cols,
                "n_agents": agents,
                "msg_bits": 0,
                "sensor_range": 1,
                "request_queue_size": req,
                "max_inactivity_steps": None,
                "max_steps": 500,
                "reward_type": _rewards[rew],
            },
        )
