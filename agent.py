import numpy as np


class Agent:
    def __init__(
        self, model, gaussian_width=0.3, exploration_rate=0.1, use_clusters=False
    ):
        self.model = model
        self.gaussian_width = gaussian_width
        self.exploration_rate = exploration_rate
        self.use_clusters = use_clusters

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

        # Ensure transitions refer to valid cluster indices
        for action in self.model.actions:
            if len(self.model.state_action_transitions_from[action]) > 0:
                dist = (state - states_mean) / states_std - (
                    self.model.states[self.model.state_action_transitions_from[action]]
                    - states_mean
                ) / states_std
                weight = np.exp(-np.sum(np.square(dist), axis=1) / self.gaussian_width)
                action_weights[action] = np.sum(weight)
                action_rewards[action] = (
                    np.sum(
                        weight
                        * self.model.rewards[
                            np.array(
                                self.model.state_action_transitions_to[action],
                                dtype=int,
                            )  # Ensure it's an integer array
                        ]
                    )
                    / action_weights[action]
                )
        return action_rewards, action_weights

    def get_action(self, action_rewards, action_weights):
        for action in self.model.actions:
            if self.use_clusters:
                return np.argmax(action_rewards)
            if action_weights[action] == 0:
                return action  # Return action that has never been chosen before
            if action_weights[action] / np.max(action_weights) < self.exploration_rate:
                return (
                    action  # Return action that has little data for the current state
                )
        return np.argmax(action_rewards)
