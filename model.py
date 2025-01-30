import numpy as np
from scipy.cluster.vq import vq, whiten
from scipy.cluster.vq import kmeans2
from scipy.special import softmax
from sklearn.cluster import k_means
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
        gaussian_width = 0.2
        print("Running k-means...")
        
        states_array = np.array(self.states)
        weights = softmax(self.rewards)

        # Compute distances from mean state (or an alternative reference point)
        mean_state = np.mean(states_array, axis=0)  # Could also use median or a specific state
        dist = states_array - mean_state  # Compute distance from mean
        squared_dist = np.sum(np.square(dist), axis=1)  # Squared Euclidean distance

        # Apply Gaussian weighting function
        gaussian_weights = np.exp(-squared_dist / gaussian_width)
        
        centroids, labels, inertia = k_means(X=states_array, n_clusters=k, sample_weight=gaussian_weights)
        
        self.clustered_states = centroids
        self.cluster_labels = labels
    
    def update_transitions_and_rewards_for_clusters(self):
        # Map states to clusters
        state_to_cluster = {i: self.cluster_labels[i] for i in range(len(self.states))}
        
        # Create new lists for clustered transitions
        clustered_transitions_from = [[] for _ in self.actions]
        clustered_transitions_to = [[] for _ in self.actions]
        
        for action in self.actions:
            for from_state, to_state in zip(self.state_action_transitions_from[action], self.state_action_transitions_to[action]):
                from_cluster = state_to_cluster[from_state]
                to_cluster = state_to_cluster[to_state]

                clustered_transitions_from[action].append(from_cluster)
                clustered_transitions_to[action].append(to_cluster)

        # Update model transitions
        self.state_action_transitions_from = clustered_transitions_from
        self.state_action_transitions_to = clustered_transitions_to