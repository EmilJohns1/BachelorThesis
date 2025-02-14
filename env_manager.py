import gymnasium
import pygame

import numpy as np


class EnvironmentManager:
    def __init__(self, render_mode, seed, environment="CartPole-v1"):
        self.env = gymnasium.make(environment, render_mode=render_mode)
        self.env.reset(seed=seed)

    def reset(self):
        return self.env.reset()

    def step(self, action):
        return self.env.step(action)

    def close(self):
        self.env.close()
