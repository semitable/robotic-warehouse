import gym, ray
import pprint
from ray.rllib.agents import impala
from ray import tune
from ray.tune.registry import register_env

from utils import ENVIRONMENT, parse, env_creator

register_env(f"ray-{ENVIRONMENT}", env_creator)

if __name__ == "__main__":
    args = parse()

    ray.init()
    tune.run(
        "IMPALA",
        checkpoint_freq=100,
        checkpoint_at_end=True,
        stop={
            "episode_reward_mean": 20,
            "training_iteration": 1000,
        },
        config={
            "env": f"ray-{ENVIRONMENT}",
            "eager": False,
            "sample_batch_size": 50,
            "train_batch_size": 500,
            "min_iter_time_s": 10,
            "num_workers": 2,
            # number of GPUs the learner should use.
            "num_gpus": 0,
            # number of sample batches to store for replay. The number of transitions
            # saved total will be (replay_buffer_num_slots * sample_batch_size).
            "replay_buffer_num_slots": 1e5,
            "lr": tune.grid_search([0.01, 0.001, 0.0005, 0.0001]),
        },
    )
