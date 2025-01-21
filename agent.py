import numpy as np

class Agent:
    def __init__(self, model, gaussian_width=0.3, exploration_rate=0.1):
        self.model = model
        self.gaussian_width = gaussian_width
        self.exploration_rate = exploration_rate

    def normalize_states(self):
        states_mean = np.array([0.])
        states_std = np.array([1.])
        if len(self.model.states) > 0:
            states_mean = np.mean(self.model.states, axis=0)
            states_std = np.std(self.model.states, axis=0)
        for i, _ in enumerate(states_std):
            if states_std[i] == 0.:
                states_std[i] = 1.
        return states_mean, states_std

    def compute_action_rewards(self, state, states_mean, states_std):
        action_rewards = [0. for _ in self.model.state_action_transitions]
        weight_sums = [0. for _ in self.model.state_action_transitions]
        for i, transitions in enumerate(self.model.state_action_transitions):
            for state_from, state_to in transitions:
                dist = (
                    (state - states_mean) / states_std -
                    (self.model.states[state_from] - states_mean) / states_std
                )
                weight = np.exp(-np.sum(np.square(dist)) / self.gaussian_width)
                weight_sums[i] += weight
                action_rewards[i] += weight * self.model.rewards[state_to]
            if weight_sums[i] > 0.:
                action_rewards[i] /= weight_sums[i]
        return action_rewards, weight_sums

    def get_action(self, action_rewards, weight_sums):
        for i, weight_sum in enumerate(weight_sums):
            if weight_sum == 0:
                return i  # Return action that has never been chosen before
            if weight_sum / np.max(weight_sums) < self.exploration_rate:
                return i  # Return action that has little data for the current state
        return np.argmax(action_rewards)
