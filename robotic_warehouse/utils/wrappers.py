import gym
import numpy as np
from robotic_warehouse import Action
from gym import spaces


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
        if type(action) is int:
            action = [action]
        else:
            action = np.split(action, self.n_agents)

        observation, reward, done, info = super().step(action)
        observation = np.concatenate(observation)
        reward = np.sum(reward)
        done = all(done)
        return observation, reward, done, info
