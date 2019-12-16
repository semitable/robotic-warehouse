import gym, ray
from robotic_warehouse import Warehouse, RewardType
from robotic_warehouse.utils.wrappers import DictAgents
from ray.rllib.agents import ppo
from ray.rllib.env import MultiAgentEnv
from ray import tune
from ray.tune.registry import register_env
import pprint

ENVIRONMENT = "rware-1x3-2ag-10req-indiv-v0"


class RayWarehouseEnv(MultiAgentEnv):
    def __init__(self, env):
        super().__init__()
        self.__env = env
        self.observation_space = env.observation_space[0]
        self.action_space = env.action_space[0]

    def reset(self):
        return self.__env.reset()

    def step(self, action_dict):
        return self.__env.step(action_dict)

    def render(self, *args, **kwargs):
        return self.__env.render(*args, **kwargs)


def env_creator(env_config):
    env = gym.make(ENVIRONMENT)
    env = DictAgents(env)
    env = RayWarehouseEnv(env)
    return env


register_env(f"ray-{ENVIRONMENT}", env_creator)

if __name__ == "__main__":
    ray.init()
    tune.run(
        "PPO",
        checkpoint_freq=100,
        checkpoint_at_end=True,
        stop={"episode_reward_mean": 300},
        config={
            "env": f"ray-{ENVIRONMENT}",
            "num_gpus": 0,
            "num_workers": 11,
            "num_envs_per_worker": 5,
            # "lr": tune.grid_search([0.01, 0.001, 0.0001]),
            "eager": False,
        },
    )
