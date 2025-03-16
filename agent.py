import gymnasium as gym

import numpy as np


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
        action_rewards = [0.0 for _ in self.model.actions]
        action_weights = [0.0 for _ in self.model.actions]

        for action in self.model.actions:
            if len(self.model.state_action_transitions[action]) > 0:
                states, deltas = zip(*self.model.state_action_transitions[action])
                states = np.array(states)
                deltas = np.array(deltas)
                dist = (state - states_mean) / states_std - (
                    self.model.states[states]
                    - states_mean
                ) / states_std
                weight = np.exp(-np.sum(np.square(dist), axis=1) / self.gaussian_width)
                
                estimated_delta = ((
                    np.sum(weight[:, None] * deltas, axis=0)
                    / np.sum(weight)
                ) - states_mean) / states_std

                predicted_state = state + estimated_delta
                dist = (predicted_state - states_mean)/states_std - (self.model.states[states] - states_mean)/states_std
                weight = np.exp(-np.sum(np.square(dist), axis=1) / self.gaussian_width)
                action_weights[action] = np.sum(weight)
                action_rewards[action] = np.sum(weight*self.model.rewards[states])/action_weights[action]
                print(f"act rew: {action_rewards[action]}")

                #action_rewards[action] = np.exp(-np.sum(np.square(estimated_delta)) / self.gaussian_width) * estimated_reward # Vil alltid føre til at rewarden blir predikert å bli lavere
                
        return action_rewards, action_weights

    def get_action(self, action_rewards, action_weights):
        if isinstance(self.model.actions, list):
            if self.testing:
                return np.argmax(action_rewards)
            if np.any(action_weights == 0): 
                return np.random.choice(self.model.actions[np.where(action_weights == 0)[0]])
            if np.random.rand() < self.exploration_rate:
                return np.random.choice(self.model.actions)
            return np.argmax(action_rewards)

        if isinstance(self.model.actions, gym.spaces.Box):
            action_dim = self.model.actions.shape[0]
            return np.random.uniform(
                self.model.actions.low, self.model.actions.high, size=action_dim
            )
