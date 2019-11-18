import gym


class MultiAgentObservationSpace(list):
    def __init__(self, ma_space):
        for x in ma_space:
            assert isinstance(x, gym.spaces.space.Space)

        super().__init__(ma_space)

    def sample(self):
        """ Samples from each element of the list """
        return [sa_space.sample() for sa_space in self]

    def contains(self, obs):
        """ Checks if each obs is contained in respective agent """
        for space, ob in zip(self, obs):
            if not space.contains(ob):
                return False
        else:
            return True


class MultiAgentActionSpace(list):
    def __init__(self, ma_space):
        for x in ma_space:
            assert isinstance(x, gym.spaces.space.Space)

        super(MultiAgentActionSpace, self).__init__(ma_space)

    def sample(self):
        """ Samples action from each element of the list"""
        return [sa_space.sample() for sa_space in self]
