import numpy as np
from scipy.cluster.vq import vq, kmeans, whiten
from scipy.cluster.vq import kmeans2
import matplotlib.pyplot as plt

class Model:
    def __init__(self, action_space_n, _discount_factor, _observation_space):
        self.states: np.ndarray = np.empty((0, _observation_space.shape[0]))  # States are stored here
        self.rewards: np.ndarray = np.empty(0)  # Value for each state index

        self.actions: list[int] = list(range(action_space_n))
        # Lists for each action containing from and to state indices, i.e.
        # in which state the action was performed and the resulting state of that action
        self.state_action_transitions_from: list[list[int]] = [[] for _ in self.actions]
        self.state_action_transitions_to: list[list[int]] = [[] for _ in self.actions]

        self.discount_factor: float = _discount_factor # Low discount factor penalizes longer episodes
        self.states_mean = np.array([0., 0., 0., 0.]) 
        self.M2 = np.array([0., 0., 0., 0.])
        self.states_std = np.array([1., 1., 1., 1.])

    def update_model(self, states, actions, rewards):
        for i, state in enumerate(states):
            self.add_state(state)
            self.rewards = np.hstack((self.rewards, np.power(self.discount_factor, len(states) - 1 - i) * rewards))
            if i > 0:
                self.state_action_transitions_from[actions[i - 1]].append(len(self.states) - 2)
                self.state_action_transitions_to[actions[i - 1]].append(len(self.states) - 1)
    
    def add_state(self, new_state):
        self.states = np.vstack((self.states, new_state))
        n = len(self.states)

        for i in range(4):
            delta = new_state[i] - self.states_mean[i]

            self.states_mean[i] += delta/n

            self.M2[i] += delta*(new_state[i] - self.states_mean[i])

            self.states_std[i] = np.sqrt(self.M2[i]/n)

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
        
