import gymnasium
import pygame
import numpy as np
import random

class EnvironmentManager:
    def __init__(self, render_mode, environment="CartPole-v1"):
        self.env = gymnasium.make(environment, render_mode=render_mode)
        np.random.seed()

    def reset(self):
        random_seed = random.randint(0, 2**32 - 1)
        self.env.action_space.seed(random_seed)
        np.random.seed(random_seed)
        return self.env.reset(seed=random_seed)

    def step(self, action):
        return self.env.step(action)

    def close(self):
        self.env.close()