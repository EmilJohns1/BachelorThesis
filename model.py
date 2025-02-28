from collections import defaultdict
from scipy.cluster.vq import kmeans2
from scipy.cluster.vq import vq
from scipy.cluster.vq import whiten
from scipy.special import log_softmax


import numpy as np
import copy
import matplotlib.pyplot as plt

from sklearn.cluster import k_means
from clusterer import Clusterer
from util.cluster_visualizer import ClusterVisualizer

class Model:
    def __init__(self, action_space_n, discount_factor, observation_space, K, sigma):
        obs_dim = observation_space.shape[0]
        self.states: np.ndarray = np.empty((0, obs_dim))  # States are stored here
        self.clusterer = Clusterer(K=K, D=obs_dim, sigma=sigma, lambda_=0.5, learning_rate=0.02, action_space_n=action_space_n)
        self.original_states: np.ndarray = np.empty(
            (0, obs_dim)
        )  # States are stored here
        self.rewards: np.ndarray = np.empty(0)  # Value for each state index
        self.original_rewards = np.empty(0)
        self.reward_weights = np.ones(0)

        self.actions: list[int] = list(range(action_space_n))
        # Lists for each action containing from and to state indices, i.e.
        # in which state the action was performed and the resulting state of that action
        self.state_action_transitions_from: list[list[int]] = [[] for _ in self.actions]
        self.state_action_transitions_to: list[list[int]] = [[] for _ in self.actions]
        self.new_transitions_index = np.zeros(len(self.actions), dtype=int)

        self.discount_factor: float = (
            discount_factor  # Low discount factor penalizes longer episodes
        )
        self.states_mean = np.zeros(obs_dim)
        self.M2 = np.zeros(obs_dim)
        self.states_std = np.ones(obs_dim)

        self.using_clusters = False

    def update_model(self, states, actions, rewards):
        for i, state in enumerate(states):
            self.add_state(state)
            self.rewards = np.hstack(
                (
                    self.rewards,
                    np.power(self.discount_factor, len(states) - 1 - i) * rewards,
                )
            )
            if i > 0:
                self.state_action_transitions_from[actions[i - 1]].append(
                    len(self.states) - 2
                )
                self.state_action_transitions_to[actions[i - 1]].append(
                    len(self.states) - 1
                )

    def add_state(self, new_state):
        self.states = np.vstack((self.states, new_state))
        n = len(self.states)

        delta = new_state - self.states_mean  # Element-wise difference
        self.states_mean += delta / n  # Update mean

        self.M2 += delta * (new_state - self.states_mean)  # Update variance accumulator
        self.states_std = np.sqrt(self.M2 / n)  # Compute standard deviation

    def scale_rewards(self, log_softmaxed_rewards, new_min=0.01, new_max=100.0):
        print("Shifting rewards...")

        rewards = np.array(log_softmaxed_rewards)
        max_reward = np.max(rewards)

        if np.all(rewards == max_reward):  
            print("Rewards have no variation, shifting skipped.")
            return rewards

        # Shift so that the maximum value is new_max while keeping differences
        shifted_rewards = rewards + (new_max - max_reward)

        return shifted_rewards
    
    def scaled_log_softmax(self):
        log_softmax_rewards = log_softmax(self.rewards)
        scaled_rewards = self.scale_rewards(log_softmaxed_rewards=log_softmax_rewards, new_max=3)
        return np.exp(scaled_rewards)

    def run_k_means(self, k):
        print("Running k-means...")

        self.original_states = self.states
        self.original_rewards = self.rewards

        states_array = np.array(self.states)

        new_rewards = self.scaled_log_softmax()
        
        if np.any(new_rewards == 0):
            print("Warning: Zero values detected in new_rewards!")
            print("Indices with zero values:", np.where(new_rewards == 0))

        centroids, labels, inertia = k_means(
            X=states_array, n_clusters=k, sample_weight=new_rewards
        )

        self.clustered_states = centroids
        self.cluster_labels = labels
    
    def update_transitions_and_rewards_for_clusters(self, gaussian_width=0.2):
        state_to_cluster = {i: self.cluster_labels[i] for i in range(len(self.states))}

        transition_counts = defaultdict(lambda: defaultdict(int))

        for action in self.actions:
            for from_state, to_state in zip(
                self.state_action_transitions_from[action],
                self.state_action_transitions_to[action],
            ):
                from_cluster = state_to_cluster[from_state]
                to_cluster = state_to_cluster[to_state]

                transition_counts[(from_cluster, action)][to_cluster] += 1

        clustered_transitions_from = [[] for _ in self.actions]
        clustered_transitions_to = [[] for _ in self.actions]
        clustered_transition_probs = [{} for _ in self.actions]

        for (from_cluster, action), to_clusters in transition_counts.items():
            total_transitions = sum(to_clusters.values())

            for to_cluster, count in to_clusters.items():
                clustered_transitions_from[action].append(from_cluster)
                clustered_transitions_to[action].append(to_cluster)
                clustered_transition_probs[action][(from_cluster, to_cluster)] = (
                    count / total_transitions
                )

        self.state_action_transitions_from = clustered_transitions_from
        self.state_action_transitions_to = clustered_transitions_to
        self.transition_probs = clustered_transition_probs

         # Initialize rewards for clusters
        num_clusters = len(self.clustered_states)
        cluster_rewards = np.zeros(num_clusters)
        cluster_weights = np.zeros(num_clusters)  # Sum of weights for normalization

        # Compute new rewards for clusters
        states_array = np.array(self.states)

        for i, centroid in enumerate(self.clustered_states):
            # Compute distances between centroid and all states in the cluster
            cluster_indices = np.where(self.cluster_labels == i)[
                0
            ]  # Get state indices in this cluster
            cluster_states = states_array[cluster_indices]
            cluster_rewards_raw = self.rewards[cluster_indices]

            if len(cluster_states) > 0:
                dist = np.sum(
                    np.square(cluster_states - centroid), axis=1
                )  # Squared Euclidean distance
                weights = np.exp(-dist / gaussian_width)  # Apply Gaussian weighting

                # Weighted sum of rewards
                weighted_rewards = np.sum(weights * cluster_rewards_raw)
                total_weight = np.sum(weights)  # Normalization factor

                cluster_rewards[i] = (
                    weighted_rewards / total_weight if total_weight > 0 else 0
                )
                cluster_weights[i] = (
                    total_weight  # Keep track of the total weight for debugging
                )

        # Store the computed cluster rewards
        self.rewards = cluster_rewards
        self.states = self.clustered_states
        self.states_mean = np.mean(self.states, axis=0)
        self.states_std = np.std(self.states, axis=0)

    def cluster_states(self, k, gaussian_width):
        self.original_states = self.states
        self.original_rewards = self.rewards
        print(f"Total states: {len(self.states)}")
        
        X = self.states
        X_rewards = self.rewards
        if self.using_clusters: # Exclude the centroids from calculations
            X = self.states[k:]  
            X_rewards = self.rewards[k:]
        
        print("Running online clustering")
        for i in range(1):
            self.clusterer.update(X=X, X_rewards=X_rewards)
        
        print("Updating transitions")
        # For each action, slice from the stored new_transitions_index for that action,
        # and re-index by subtracting k.
        transitions_from_new = []
        transitions_to_new = []
        for i, _ in enumerate(self.actions):
            transitions_from_new.append(
                [x - k for x in self.state_action_transitions_from[i][self.new_transitions_index[i]:]]
            )
            transitions_to_new.append(
                [x - k for x in self.state_action_transitions_to[i][self.new_transitions_index[i]:]]
            )
        
        self.clusterer.update_transitions(
            x=X, 
            state_action_transitions_from=transitions_from_new,
            state_action_transitions_to=transitions_to_new,
            threshold=5e-1
        )
        
        # Get the updated centroids, rewards, and transitions from the clusterer.
        new_states, new_rewards, new_transitions_from, new_transitions_to = self.clusterer.get_model_attributes()
        
        self.states = copy.deepcopy(new_states)
        self.rewards = copy.deepcopy(new_rewards)
        self.state_action_transitions_from = copy.deepcopy(new_transitions_from)
        self.state_action_transitions_to = copy.deepcopy(new_transitions_to)
        
        # Update the new_transitions_index array for each action.
        # Each element now holds the length of the transitions list for that action.
        new_indices = []
        for i, _ in enumerate(self.actions):
            new_indices.append(len(new_transitions_from[i]))
        self.new_transitions_index = np.array(new_indices)
        
        self.states_mean = np.mean(self.states, axis=0)
        self.states_std = np.std(self.states, axis=0)
        self.using_clusters = True

        #visualizer = ClusterVisualizer(model=self)
        #visualizer.plot_clusters()