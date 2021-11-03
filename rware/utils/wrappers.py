import gym
import numpy as np
from rware import Action
from gym import spaces
from gym import ObservationWrapper

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

        self.observation_space = spaces.Tuple(
            tuple(space for space in self.observation_space)
        )

    def reset(self, **kwargs):
        observation = super().reset(**kwargs)
        return np.concatenate(
            [spaces.flatten(s, o) for s, o in zip(self.observation_space, observation)]
        )

    def step(self, action):
        try:
            action = np.split(action, self.n_agents)
        except (AttributeError, IndexError):
            action = [action]

        observation, reward, done, info = super().step(action)
        observation = np.concatenate(
            [spaces.flatten(s, o) for s, o in zip(self.observation_space, observation)]
        )
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


class FlattenSAObservation(ObservationWrapper):
    r"""Observation wrapper that flattens the observation."""
    def __init__(self, env):
        super(FlattenSAObservation, self).__init__(env)
        
        ma_spaces = []

        for sa_obs in env.observation_space:
            flatdim = spaces.flatdim(sa_obs)
            ma_spaces += [spaces.Box(low=-float('inf'), high=float('inf'), shape=(flatdim,), dtype=np.float32)]
        
        self.observation_space = spaces.Tuple(tuple(ma_spaces))

    def observation(self, observation):
        return [spaces.flatten(obs_space, obs) for obs_space, obs in zip(self.env.observation_space, observation)]

class SquashDones(gym.Wrapper):

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        return observation, reward, all(done), info
