import gymnasium as gym

import numpy as np

from time import perf_counter


class Agent:
    def __init__(
        self, model, gaussian_width=3.0, exploration_rate=0.1
    ):
        self.model = model
        self.gaussian_width = gaussian_width
        self.exploration_rate = exploration_rate
        self.testing = False
        self.predicted_deltas = {} # Used to adjust splines in the model

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
        self.predicted_deltas = {}

        for action in self.model.actions:
            if self.model.transition_model.has_transitions(action):
                (center, query_points, query_point_rewards) = self.model.get_transition_data(state, action)
                from_states = self.model.state_action_transitions_from[action]
                to_states = self.model.states[self.model.state_action_transitions_to[action]]
                predicted_delta = np.zeros(self.model.state_dimensions) # Same dimension as states
                if len(self.model.delta_predictor) > 0:
                    predicted_delta = predicted_delta = (
                        self.model.delta_predictor[action].predict(state.reshape(1, -1))[0]
                        if self.model.delta_predictor[action] is not None
                        else np.zeros(self.model.state_dimensions)
                    )
                self.predicted_deltas[action] = predicted_delta

                weight = np.exp(-np.sum(np.square((state + predicted_delta - states_mean) / states_std - (to_states - states_mean) / states_std), axis=1) / self.gaussian_width)
                weight_2 = np.exp(-np.sum(np.square((center - states_mean) / states_std - (query_points - states_mean) / states_std), axis=1) / self.gaussian_width)

                assert np.array_equal(weight, weight_2)
                assert np.array_equal(self.model.rewards[from_states], query_point_rewards)

                sum_weight = np.sum(weight)
                if sum_weight > 0:
                    action_weights[action] = sum_weight
                    action_rewards[action] = np.sum(weight * self.model.rewards[from_states]) / sum_weight
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
        
    def update_approximation(self, action, actual_delta, error_threshold=0.01):
        if not self.testing:
            self.model.check_transition_error(action, actual_delta, error_threshold)
