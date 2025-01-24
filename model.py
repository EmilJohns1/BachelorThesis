import numpy as np
from scipy.cluster.vq import vq, kmeans, whiten
import matplotlib.pyplot as plt

class Model:
    def __init__(self, action_space_n, _discount_factor):
        self.states: list[np.ndarray] = []  # States are stored here
        self.rewards: list[float] = []  # Value for each state index
        self.state_action_transitions: list[list[tuple[int, int]]] = [
            [] for _ in range(action_space_n)
        ]  # A list for each action
        self.discount_factor: float = _discount_factor # Low discount factor penalizes longer episodes
        self.states_mean = np.array([0., 0., 0., 0.]) 
        self.M2 = np.array([0., 0., 0., 0.])
        self.states_std = np.array([1., 1., 1., 1.])

    def update_model(self, states, actions, rewards):
        for i, state in enumerate(states):
            self.add_state(state)
            self.rewards.append(np.power(self.discount_factor, len(states) - 1 - i) * rewards)
            if i > 0:
                self.state_action_transitions[actions[i - 1]].append(
                    (len(self.states) - 2, len(self.states) - 1)
                )
    
    def add_state(self, new_state):
        self.states.append(new_state)
        n = len(self.states)

        for i in range(4):
            delta = new_state[i] - self.states_mean[i]

            self.states_mean[i] += delta/n

            self.M2[i] += delta*(new_state[i] - self.states_mean[i])

            self.states_std[i] = np.sqrt(self.M2[i]/n)
        