import numpy as np
from scipy.cluster.vq import vq, kmeans, whiten
import matplotlib.pyplot as plt

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
    
    def run_k_means(self):
        print("Running k-means...")
        
    def show_states(self):
        states = np.array(self.states)  # Convert list of ndarrays to a 2D array
        plt.figure()
        for i in range(states.shape[1]):  # Loop through the 4 dimensions
            plt.plot(states[:, i], label=f"State Dimension {i+1}")
        plt.legend()
        plt.xlabel("Time Step")
        plt.ylabel("State Value")
        plt.title("State Evolution Over Time")
        plt.show()