class Model:
    def __init__(self, action_space_n):
        self.states: list[np.ndarray] = []  # States are stored here
        self.rewards: list[float] = []  # Value for each state index
        self.state_action_transitions: list[list[tuple[int, int]]] = [
            [] for _ in range(action_space_n)
        ]  # A list for each action

    def update_model(self, states, actions, rewards):
        for i, state in enumerate(states):
            self.states.append(state)
            self.rewards.append(rewards)
            if i > 0:
                self.state_action_transitions[actions[i - 1]].append(
                    (len(self.states) - 2, len(self.states) - 1)
                )