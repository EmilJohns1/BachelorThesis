from transitions.transition_method import Transition_Method

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

class Delta_Transition_Model:
    def __init__(self, action_space_n):
        self.state_action_transitions_from = [[] for _ in range(action_space_n)]
        self.state_action_transitions_to = [[] for _ in range(action_space_n)]
        self.transition_delta = [[] for _ in range(action_space_n)]
        self.delta_predictor = [None] * action_space_n

    def update_transitions(self, action, from_state, to_state):
        from_index, from_vector = from_state
        to_index, to_vector = to_state

        self.state_action_transitions_from[action].append(from_index)
        self.state_action_transitions_to[action].append(to_index)
        self.transition_delta[action].append(to_vector - from_vector)