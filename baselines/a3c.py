import gym, ray
import pprint
from ray.rllib.agents import a3c
from ray import tune
from ray.tune.registry import register_env

from utils import ENVIRONMENT, parse, env_creator

register_env(f"ray-{ENVIRONMENT}", env_creator)

if __name__ == "__main__":
    args = parse()

    ray.init()
    tune.run(
        "A3C",
        checkpoint_freq=100,
        checkpoint_at_end=True,
        stop={
            "episode_reward_mean": 20,
            "training_iteration": 1000,
        },
        config={
            "env": f"ray-{ENVIRONMENT}",
            "eager": False,
            # Size of rollout batch
            "sample_batch_size": 10,
            # Learning rate
            "lr": tune.grid_search([0.01, 0.001, 0.0001, 1e-5]),
            # Entropy coefficient
            "entropy_coeff": tune.grid_search([0.1, 0.01, 0.001]),
            # Workers sample async. Note that this increases the effective
            # sample_batch_size by up to 5x due to async buffering of batches.
            "sample_async": True,
        },
    )
