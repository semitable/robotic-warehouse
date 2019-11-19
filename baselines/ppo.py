import gym, ray
from robotic_warehouse import Warehouse, RewardType
from robotic_warehouse.utils.wrappers import DictAgents
from ray.rllib.agents import ppo
from ray.rllib.env import MultiAgentEnv

from ray.tune.registry import register_env
import pprint

ray.init()


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


def env_creator(env_config):
    env = Warehouse(3, 8, 1, 2, 0, 1, 10, 200, RewardType.GLOBAL)
    env = DictAgents(env)
    env = RayWarehouseEnv(env)
    return env


register_env("rware-v0", env_creator)

config = ppo.DEFAULT_CONFIG.copy()
config["num_workers"] = 10


trainer = ppo.PPOTrainer(env="rware-v0", config=config)
pp = pprint.PrettyPrinter(depth=6)

for _ in range(1000):
    pp.pprint(trainer.train())
