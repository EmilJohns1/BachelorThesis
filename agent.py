import numpy as np

class Agent:
    def __init__(self, model, gaussian_width=0.3, exploration_rate=0.1):
        self.model = model
        self.gaussian_width = gaussian_width
        self.exploration_rate = exploration_rate

    def normalize_states(self):
        states_mean = np.array([0., 0., 0., 0.])
        states_std = np.array([1., 1., 1., 1.])

        if len(self.model.states) > 0:
            states_mean = self.model.states_mean
            states_std = self.model.states_std

        for i, _ in enumerate(states_std):
            if states_std[i] == 0.:
                states_std[i] = 1.
        return states_mean, states_std

    def compute_action_rewards(self, state, states_mean, states_std):
        action_rewards = [0. for _ in self.model.state_action_transitions]
        weight_sums = [0. for _ in self.model.state_action_transitions]
        for action, _ in enumerate(self.model.state_action_transitions):
            for state_from, state_to in self.model.state_action_transitions[action]:
                dist = (state - states_mean) / states_std - (self.model.states[state_from] - states_mean) / states_std
                weight = np.exp(-np.sum(np.square(dist)) / self.gaussian_width)
                weight_sums[action] += weight
                action_rewards[action] += weight * self.model.rewards[state_to]
            if weight_sums[action] > 0.:
                action_rewards[action] /= weight_sums[action]

        return action_rewards, weight_sums

    def get_action(self, action_rewards, weight_sums):
        for action, _ in enumerate(self.model.state_action_transitions):# Hvorfor har vi state_action_transactions i stedet for weight_sums?
            if weight_sums[action] == 0:
                return action  # Return action that has never been chosen before
            if weight_sums[action] / np.max(weight_sums) < self.exploration_rate:
                return action  # Return action that has little data for the current state
        return np.argmax(action_rewards)
