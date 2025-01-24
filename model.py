import numpy as np
from scipy.cluster.vq import vq, kmeans, whiten
from scipy.cluster.vq import kmeans2
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
    
    def run_k_means(self, k):
        print("Running k-means...")
        
        states_array = np.array(self.states)
        
        centroids, labels = kmeans2(states_array, k, minit="points")
        
        self.clustered_states = centroids
        self.cluster_labels = labels
    
    def update_transitions_and_rewards_for_clusters(self):
        
        # Initialize clustered transitions and rewards
        num_clusters = len(self.clustered_states)
        clustered_transitions = [[] for _ in range(len(self.state_action_transitions))]
        clustered_rewards = [0.0 for _ in range(num_clusters)]
        cluster_counts = [0 for _ in range(num_clusters)]

        # Map original states and transitions to clusters
        for action, transitions in enumerate(self.state_action_transitions):
            for state_from, state_to in transitions:
                cluster_from = self.cluster_labels[state_from]
                cluster_to = self.cluster_labels[state_to]
                clustered_transitions[action].append((cluster_from, cluster_to))
        
        # Aggregate rewards for clusters
        for i, cluster_label in enumerate(self.cluster_labels):
            clustered_rewards[cluster_label] += self.rewards[i]
            cluster_counts[cluster_label] += 1

        # Normalize rewards by the number of states in each cluster
        for i in range(num_clusters):
            if cluster_counts[i] > 0:
                clustered_rewards[i] /= cluster_counts[i]

        # Update the model with clustered transitions and rewards
        self.state_action_transitions = clustered_transitions
        self.rewards = clustered_rewards
        
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