import argparse
import gym, ray
from robotic_warehouse import Warehouse, RewardType
from robotic_warehouse.utils.wrappers import DictAgents
from ray.rllib.env import MultiAgentEnv

ENVIRONMENT = "rware-tiny-2ag-v0"

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
    env = gym.make(env_config)
    env = DictAgents(env)
    env = RayWarehouseEnv(env)
    return env

# setup parser
def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-iters", type=int, default=1000, help="maximum number of iterations")
    parser.add_argument(
            "--stop-reward",
            type=int,
            default=20,
            help="number of episodic reward mean to stop at",
    )
    parser.add_argument("--num-workers", type=int, default=1)
    args = parser.parse_args()
    return args
