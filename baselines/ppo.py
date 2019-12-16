import gym, ray
import pprint
from ray.rllib.agents import ppo
from ray import tune
from ray.tune.registry import register_env

from utils import ENVIRONMENT, parse, env_creator

register_env(f"ray-{ENVIRONMENT}", env_creator)

if __name__ == "__main__":
    args = parse()

    ray.init()
    tune.run(
        "PPO",
        checkpoint_freq=100,
        checkpoint_at_end=True,
        stop={
            "episode_reward_mean": args.stop_reward,
            "training_iteration": args.num_iters,
        },
        config={
            "env": f"ray-{ENVIRONMENT}",
            "num_gpus": 0,
            "num_workers": args.num_workers,
            "num_envs_per_worker": 5,
            "lr": tune.grid_search([0.01, 0.001, 0.0001, 1e-5]),
            "eager": False,
        },
    )
