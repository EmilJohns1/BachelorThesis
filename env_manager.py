import gymnasium



class EnvironmentManager:
    def __init__(self, render_mode, seed, environment="CartPole-v1"):
        self.env = gymnasium.make(environment, render_mode=render_mode)
        self.env.reset(seed=seed)
        self.action_space_n = self.env.action_space.n
        self.observation_space = self.env.observation_space

    def reset(self):
        return self.env.reset()

    def step(self, action):
        return self.env.step(action)

    def close(self):
        self.env.close()
