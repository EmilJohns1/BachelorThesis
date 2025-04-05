from transitions.transition_method import Transition_Method
from sklearn.linear_model import LinearRegression

import numpy as np

def factory(method: Transition_Method, action_space_n):
    match method:
        case Transition_Method.Direct_Transition_Mapping:
            return Direct_Transition_Model(action_space_n)
        case Transition_Method.Delta_Transition_Variant:
            return Delta_Transition_Model(action_space_n)


class Direct_Transition_Model:
    def __init__(self, action_space_n):
        self.state_action_transitions_from = [[] for _ in range(action_space_n)]
        self.state_action_transitions_to = [[] for _ in range(action_space_n)]

    def update_transitions(self, action, from_state, to_state):
        from_index, from_vector = from_state
        to_index, to_vector = to_state

        self.state_action_transitions_from[action].append(from_index)
        self.state_action_transitions_to[action].append(to_index)

    def has_transitions(self, action):
        len(self.state_action_transitions_from[action]) > 0

    def get_transition_center(self, state, _):
        return state
    
    def update_predictions(self, _action, _actual_delta, _error_threshold, _states):
        return

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
        len(self.state_action_transitions_from[action]) > 0

    def get_transition_center(self, state, action):
        predicted_delta = np.zeros(self.model.state_dimensions) # Same dimension as states
        if len(self.delta_predictor) > 0:
            predicted_delta = (
                self.model.delta_predictor[action].predict(state.reshape(1, -1))[0]
                if self.model.delta_predictor[action] is not None
                else np.zeros(self.model.state_dimensions)
            )
        self.predicted_deltas[action] = predicted_delta
        return state + predicted_delta
    
    def get_query_points(self, states):
        

    def update_predictions(self, action, actual_delta, error_threshold, states):
        if action not in self.predicted_deltas:
            return  # No prediction was made

        predicted_delta = self.predicted_deltas[action]

        # Compute error (Mean Squared Error)
        error = np.mean(np.square(predicted_delta - actual_delta))

        # If the error is high, update splines
        if error > error_threshold:
            print(f"Updating splines for action {action}, error: {error:.10f}")
            self.update_delta_predictors(states)

    def update_delta_predictors(self, states):
        for action in range(self.action_space_n):
            if len(self.state_action_transitions_from[action]) < 2:
                continue  # Not enough data

            X = states[self.state_action_transitions_from[action]]
            y = np.array(self.transition_delta[action])

            model = LinearRegression()
            model.fit(X, y)
            self.delta_predictor[action] = model
