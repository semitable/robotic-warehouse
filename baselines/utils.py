import argparse
import gym, ray
from ray.rllib.env import MultiAgentEnv
import re
from robotic_warehouse import Warehouse, RewardType
from robotic_warehouse.utils.wrappers import DictAgents

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

def extract_num_agents(env_config):
    regex = r"([0-9]+)ag"
    match = re.search(regex, env_config)
    assert match
    num = int(match.group(1))
    return num

def extract_spaces(env_config):
    env = gym.make(env_config)
    obs_space = env.observation_space[0]
    act_space = env.action_space[0]
    return obs_space, act_space

def env_creator(env_config):
    env = gym.make(ENVIRONMENT)
    env = DictAgents(env)
    env = RayWarehouseEnv(env)
    return env

# setup parser
def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-iters", type=int, default=1000, help="maximum number of iterations")
    parser.add_argument("--ip-head", type=str, default=None, help="IP head of training")
    parser.add_argument("--redis-pwd", type=str, default=None)
    parser.add_argument(
            "--stop-reward",
            type=int,
            default=20,
            help="number of episodic reward mean to stop at",
    )
    parser.add_argument("--num-workers", type=int, default=1)
    args = parser.parse_args()
    return args
