import gymnasium as gym

import numpy as np

import time


class Agent:
    def __init__(
        self, model, gaussian_width=0.3, exploration_rate=0.1
    ):
        self.model = model
        self.gaussian_width = gaussian_width
        self.exploration_rate = exploration_rate
        self.testing = False

    def normalize_states(self):
        states_mean = np.array([0.0, 0.0, 0.0, 0.0])
        states_std = np.array([1.0, 1.0, 1.0, 1.0])

        if len(self.model.states) > 0:
            states_mean = self.model.states_mean
            states_std = self.model.states_std

        for i, _ in enumerate(states_std):
            if states_std[i] == 0.0:
                states_std[i] = 1.0
        return states_mean, states_std

    def compute_action_rewards(self, state, states_mean, states_std):
        action_rewards = np.zeros(len(self.model.actions))
        action_weights = np.zeros(len(self.model.actions))

        for action in self.model.actions:
            if len(self.model.state_action_transitions_from[action]) > 0:
                states = self.model.state_action_transitions_from[action]
                predicted_delta = np.zeros(self.model.state_dimensions) # Same dimension as states
                if len(self.model.delta_splines) > 0:
                    predicted_delta = self.model.delta_splines[action](state.reshape(1, -1))[0]

                norm_state = (state + predicted_delta - states_mean) / states_std
                norm_states = (self.model.states[states] - states_mean) / states_std

                dist = norm_state - norm_states
                dist_sq = np.square(dist)

                weight = np.exp(-np.sum(dist_sq, axis=1) / self.gaussian_width)

                sum_weight = np.sum(weight)
                if sum_weight > 1e-8:
                    action_weights[action] = sum_weight
                    action_rewards[action] = np.sum(weight * self.model.rewards[states]) / sum_weight

        return action_rewards, action_weights

    def get_action(self, action_rewards, action_weights):
        actions_array = np.array(self.model.actions)
        if isinstance(self.model.actions, list):
            if self.testing:
                return np.argmax(action_rewards)
            if np.any(action_weights == 0): 
                return np.random.choice(actions_array[np.where(action_weights == 0)[0]])
            if np.random.rand() < self.exploration_rate:
                return np.random.choice(actions_array)
            return np.argmax(action_rewards)

        if isinstance(actions_array, gym.spaces.Box):
            action_dim = actions_array.shape[0]
            return np.random.uniform(
                actions_array.low, actions_array.high, size=action_dim
            )
