import gym
import numpy as np
from robotic_warehouse import Action
from gym import spaces
import math


class FlattenAgents(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        sa_action_space = [len(Action), *env.msg_bits * (2,)]
        if len(sa_action_space) == 1 and self.n_agents == 1:
            sa_action_space = spaces.Discrete(sa_action_space[0])
        else:
            sa_action_space = spaces.MultiDiscrete(self.n_agents * sa_action_space)
        self.action_space = sa_action_space

        obs_length = sum(
            [self.observation_space[i].shape[0] for i in range(self.n_agents)]
        )
        self.observation_space = spaces.Box(low=0, high=1, shape=(obs_length,))

    def reset(self, **kwargs):
        observation = super().reset(**kwargs)
        return np.concatenate(observation)

    def step(self, action):
        try:
            action = np.split(action, self.n_agents)
        except (AttributeError, IndexError):
            action = [action]

        observation, reward, done, info = super().step(action)
        observation = np.concatenate(observation)
        reward = np.sum(reward)
        done = all(done)
        return observation, reward, done, info


class DictAgents(gym.Wrapper):
    def reset(self, **kwargs):
        observation = super().reset(**kwargs)
        digits = int(math.log10(self.n_agents)) + 1

        return {f"agent_{i:{digits}}": obs_i for i, obs_i in enumerate(observation)}

    def step(self, action):
        digits = int(math.log10(self.n_agents)) + 1
        keys = [f"agent_{i:{digits}}" for i in range(self.n_agents)]
        assert keys == sorted(action.keys())

        # unwrap actions
        action = [action[key] for key in sorted(action.keys())]

        # step
        observation, reward, done, info = super().step(action)

        # wrap observations, rewards and dones
        observation = {
            f"agent_{i:{digits}}": obs_i for i, obs_i in enumerate(observation)
        }
        reward = {f"agent_{i:{digits}}": rew_i for i, rew_i in enumerate(reward)}
        done = {f"agent_{i:{digits}}": done_i for i, done_i in enumerate(done)}
        done["__all__"] = all(done.values())

        return observation, reward, done, info
