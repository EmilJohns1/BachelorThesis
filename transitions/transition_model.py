from collections import defaultdict
from transitions.transition_method import Transition_Method

import numpy as np

from sklearn.linear_model import LinearRegression


def factory(method: Transition_Method, action_space_n):
    match method:
        case Transition_Method.Direct_Transition_Mapping:
            return Direct_Transition_Model(action_space_n)
        case Transition_Method.Delta_Transition_Variant:
            return Delta_Transition_Model(action_space_n)


class Direct_Transition_Model:
    def __init__(self, action_space_n):
        self.action_space_n = action_space_n
        self.state_action_transitions_from = [[] for _ in range(action_space_n)]
        self.state_action_transitions_to = [[] for _ in range(action_space_n)]

    def update_transitions(self, action, from_state, to_state):
        from_index, from_vector = from_state
        to_index, to_vector = to_state

        self.state_action_transitions_from[action].append(from_index)
        self.state_action_transitions_to[action].append(to_index)

    def has_transitions(self, action):
        return len(self.state_action_transitions_from[action]) > 0

    def get_transition_center(self, state, _):
        return state

    def update_predictions(self, _action, _actual_delta, _error_threshold, _states):
        return

    def cluster_transitions(
        self,
        states_array,
        clustered_states,
        cluster_labels,
        new_states,
        k,
        gaussian_width,
    ):
        state_to_cluster = {i: cluster_labels[i] for i in range(len(states_array))}

        transition_counts = defaultdict(lambda: defaultdict(int))

        for action in range(self.action_space_n):
            for from_state, to_state in zip(
                self.state_action_transitions_from[action],
                self.state_action_transitions_to[action],
            ):
                from_cluster = state_to_cluster[from_state]
                to_cluster = state_to_cluster[to_state]

                transition_counts[(from_cluster, action)][to_cluster] += 1

        clustered_transitions_from = [[] for _ in range(self.action_space_n)]
        clustered_transitions_to = [[] for _ in range(self.action_space_n)]
        for (from_cluster, action), to_clusters in transition_counts.items():

            for to_cluster, _ in to_clusters.items():
                clustered_transitions_from[action].append(from_cluster)
                clustered_transitions_to[action].append(to_cluster)

        self.state_action_transitions_from = clustered_transitions_from
        self.state_action_transitions_to = clustered_transitions_to
        return new_states

    def get_query_points(self, action, states):
        return states[self.state_action_transitions_from[action]]

    def get_query_point_rewards(self, action, rewards):
        return rewards[self.state_action_transitions_to[action]]


class Delta_Transition_Model:
    def __init__(self, action_space_n):
        self.action_space_n = action_space_n
        self.state_action_transitions_from = [[] for _ in range(action_space_n)]
        self.state_action_transitions_to = [[] for _ in range(action_space_n)]
        self.transition_delta = [[] for _ in range(action_space_n)]
        self.delta_predictor = [None] * action_space_n
        self.predicted_deltas = {}

    def update_transitions(self, action, from_state, to_state):
        from_index, from_vector = from_state
        to_index, to_vector = to_state

        self.state_action_transitions_from[action].append(from_index)
        self.state_action_transitions_to[action].append(to_index)
        self.transition_delta[action].append(to_vector - from_vector)

    def has_transitions(self, action):
        return len(self.state_action_transitions_from[action]) > 0

    def get_transition_center(self, state, action):
        predicted_delta = np.zeros_like(state)  # Same dimension as states
        if len(self.delta_predictor) > 0:
            predicted_delta = (
                self.delta_predictor[action].predict(state.reshape(1, -1))[0]
                if self.delta_predictor[action] is not None
                else np.zeros_like(state)
            )
        self.predicted_deltas[action] = predicted_delta
        return state + predicted_delta

    def get_query_points(self, action, states):
        return states[self.state_action_transitions_to[action]]

    def get_query_point_rewards(self, action, rewards):
        return rewards[self.state_action_transitions_from[action]]

    def update_predictions(self, action, actual_delta, error_threshold, states):
        if action not in self.predicted_deltas:
            return False  # No prediction was made

        predicted_delta = self.predicted_deltas[action]

        # Compute error (Mean Squared Error)
        error = np.mean(np.square(predicted_delta - actual_delta))

        # If the error is high, update splines
        if error > error_threshold:
            self.update_delta_predictors(states)
            return True

    def update_delta_predictors(self, states):
        for action in range(self.action_space_n):
            if len(self.state_action_transitions_from[action]) < 2:
                continue

            X = states[self.state_action_transitions_from[action]]
            y = np.array(self.transition_delta[action])

            model = LinearRegression()
            model.fit(X, y)
            self.delta_predictor[action] = model

    def cluster_transitions(
        self,
        states_array,
        clustered_states,
        cluster_labels,
        new_states,
        k,
        gaussian_width,
    ):
        cluster_transitions_from = [
            [i for i in range(k)] for _ in range(self.action_space_n)
        ]
        cluster_transitions_to = [
            [i for i in range(k)] for _ in range(self.action_space_n)
        ]
        cluster_deltas = [
            [np.zeros_like(states_array[0]) for _ in range(k)]
            for _ in range(self.action_space_n)
        ]

        for i, centroid in enumerate(clustered_states):
            cluster_indices = np.where(cluster_labels == i)[0]
            for action in range(self.action_space_n):
                from_indices = np.array(self.state_action_transitions_from[action])
                deltas = np.array(self.transition_delta[action])

                valid_mask = np.isin(from_indices, cluster_indices)

                cluster_states = states_array[from_indices[valid_mask]]
                selected_deltas = deltas[valid_mask]

                weights = np.exp(
                    -np.sum(np.square(cluster_states - centroid), axis=1)
                    / gaussian_width
                )
                weighted_deltas = weights[:, np.newaxis] * selected_deltas
                weight_sum = np.sum(weights)
                if weight_sum == 0:
                    continue
                cluster_deltas[action][i] = np.sum(weighted_deltas, axis=0) / weight_sum

                to_state = centroid + cluster_deltas[action][i]
                to_state = to_state.reshape(1, -1)
                new_states = np.vstack([new_states, to_state])
                cluster_transitions_to[action][i] = len(new_states) - 1

        self.state_action_transitions_from = cluster_transitions_from
        self.state_action_transitions_to = cluster_transitions_to
        self.transition_delta = cluster_deltas
        return new_states
