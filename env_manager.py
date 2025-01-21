import gymnasium
import pygame
import numpy as np

class EnvironmentManager:
    def __init__(self, render_mode, seed):
        self.env = gymnasium.make("CartPole-v1", render_mode=render_mode)
        self.env.action_space.seed(seed)
        np.random.seed(seed)
        self.seed = seed

    def reset(self):
        return self.env.reset(seed=self.seed)

    def step(self, action):
        return self.env.step(action)

    def close(self):
        self.env.close()