import gym
import numpy as np

class OneHotWrapper(gym.ObservationWrapper):
    """One Hot encodes the observation"""

    def __init__(self, env):
        super().__init__(env)
        shape = self.observation_space.shape
        self.observation_space = gym.spaces.Box(low=0, high=4, shape=(5, *shape), dtype=np.int16)

    
    def observation(self, observation):
        out = np.zeros((observation.size, 5), dtype=np.uint8)
        out[np.arange(observation.size), observation.ravel()] = 1
        out.shape = (5,) + observation.shape
        return out