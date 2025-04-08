import gymnasium
import pygame

import numpy as np

from util.positional_encoder import PositionalEncoder

class EnvironmentManager:
    def __init__(self, render_mode, seed, environment="CartPole-v1"):
        self.env = gymnasium.make(environment, render_mode=render_mode)
        self.env.reset(seed=seed)
        self.action_space_n = self.env.action_space.n
        self.observation_space = self.env.observation_space
        self.encoder = PositionalEncoder()

    def reset(self):
        state, info = self.env.reset()
        return (self.encoder.encode(state), info)

    def step(self, action):
        state, reward, terminated, truncated, info = self.env.step(action)
        return (self.encoder.encode(state), reward, terminated, truncated, info)

    def close(self):
        self.env.close()
