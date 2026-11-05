import math

import gymnasium as gym
import numpy as np

from rware.warehouse import Action


class FlattenAgents(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        sa_action_space = [len(Action), *env.msg_bits * (2,)]
        if len(sa_action_space) == 1 and self.unwrapped.n_agents == 1:
            sa_action_space = gym.spaces.Discrete(sa_action_space[0])
        else:
            sa_action_space = gym.spaces.MultiDiscrete(
                self.unwrapped.n_agents * sa_action_space
            )
        self.action_space = sa_action_space

        self.observation_space = gym.spaces.Tuple(
            tuple(space for space in self.observation_space)
        )

    def reset(self, **kwargs):
        observation, info = super().reset(**kwargs)
        return np.concatenate(
            [
                gym.spaces.flatten(s, o)
                for s, o in zip(self.observation_space, observation)
            ]
        ), info

    def step(self, action):
        try:
            action = np.split(action, self.unwrapped.n_agents)
        except (AttributeError, IndexError):
            action = [action]

        observation, reward, done, truncated, info = super().step(action)
        observation = np.concatenate(
            [
                gym.spaces.flatten(s, o)
                for s, o in zip(self.observation_space, observation)
            ]
        )
        reward = np.sum(reward)
        return observation, reward, done, truncated, info


class DictAgents(gym.Wrapper):
    def reset(self, **kwargs):
        observation = super().reset(**kwargs)
        digits = int(math.log10(self.unwrapped.n_agents)) + 1
        return {f"agent_{i:{digits}}": obs_i for i, obs_i in enumerate(observation)}

    def step(self, action):
        digits = int(math.log10(self.unwrapped.n_agents)) + 1
        keys = [f"agent_{i:{digits}}" for i in range(self.unwrapped.n_agents)]
        assert keys == sorted(action.keys())

        # unwrap actions
        action = [action[key] for key in sorted(action.keys())]

        # step
        observation, reward, done, truncated, info = super().step(action)

        # wrap observations, rewards and dones
        observation = {
            f"agent_{i:{digits}}": obs_i for i, obs_i in enumerate(observation)
        }
        reward = {f"agent_{i:{digits}}": rew_i for i, rew_i in enumerate(reward)}
        done_dict = {
            f"agent_{i:{digits}}": done for i in range(self.unwrapped.n_agents)
        }
        truncated_dict = {
            f"agent_{i:{digits}}": truncated for i in range(self.unwrapped.n_agents)
        }

        return observation, reward, done_dict, truncated_dict, info


class FlattenSAObservation(gym.ObservationWrapper):
    r"""Observation wrapper that flattens the observation."""

    def __init__(self, env):
        super(FlattenSAObservation, self).__init__(env)

        ma_spaces = []

        for sa_obs in env.observation_space:
            flatdim = gym.spaces.flatdim(sa_obs)
            ma_spaces += [
                gym.spaces.Box(
                    low=-float("inf"),
                    high=float("inf"),
                    shape=(flatdim,),
                    dtype=np.float32,
                )
            ]

        self.observation_space = gym.spaces.Tuple(tuple(ma_spaces))

    def observation(self, observation):
        return [
            gym.spaces.flatten(obs_space, obs)
            for obs_space, obs in zip(self.env.observation_space, observation)
        ]
