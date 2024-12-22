from typing import Optional
import importlib
import pytest

from rware.warehouse import Warehouse, RewardType, ObservationType

_has_pettingzoo = importlib.util.find_spec("pettingzoo") is not None
if _has_pettingzoo:
    from pettingzoo.test import parallel_api_test
    from rware.pettingzoo import PettingZooWrapper


@pytest.mark.parametrize("n_agents", [1, 3])
@pytest.mark.parametrize("msg_bits", [0, 1])
@pytest.mark.parametrize("sensor_range", [1, 3])
@pytest.mark.parametrize("max_inactivity_steps", [None, 10])
@pytest.mark.parametrize("reward_type", [RewardType.GLOBAL, RewardType.INDIVIDUAL])
@pytest.mark.parametrize(
    "observation_type",
    [
        ObservationType.DICT,
        ObservationType.IMAGE,
        ObservationType.IMAGE_DICT,
        ObservationType.FLATTENED,
    ],
)
def test_pettingzoo_wrapper(
    n_agents: int,
    msg_bits: int,
    sensor_range: int,
    max_inactivity_steps: Optional[int],
    reward_type: RewardType,
    observation_type: ObservationType,
):
    if not _has_pettingzoo:
        pytest.skip("PettingZoo not available.")
        return

    env = Warehouse(
        shelf_columns=1,
        column_height=5,
        shelf_rows=3,
        n_agents=n_agents,
        msg_bits=msg_bits,
        sensor_range=sensor_range,
        request_queue_size=5,
        max_inactivity_steps=max_inactivity_steps,
        max_steps=None,
        reward_type=reward_type,
        observation_type=observation_type,
    )
    env = PettingZooWrapper(env)
    parallel_api_test(env)
