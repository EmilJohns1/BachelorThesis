import numpy as np
import time
from collections import defaultdict

from scipy.special import softmax


class Clusterer:
    def __init__(self, K: int, D: int, sigma: float, lambda_: float, learning_rate: float, action_space_n: int):
        self.K = K
        self.D = D
        self.mu = None
        self.is_initialized = False
        self.sigma = sigma
        self.lambda_ = lambda_
        self.learning_rate = learning_rate

        self.i: list[int] = []
        self.j: list[int] = []
        for i in range(0, K):
            for j in range(0, K):
                if i != j:
                    self.i.append(i)
                    self.j.append(j)


        #Rewards for each centroid i
        self.cluster_rewards = np.zeros(K)

        #Total weight for each centroid, used for normalizing rewards
        self.cluster_weights = np.zeros(K)

        #Cluster transitions
        self.actions: list[int] = list(range(action_space_n))
        self.transitions_from: list[list[int]] = [[] for _ in self.actions]
        self.transitions_to: list[list[int]] = [[] for _ in self.actions]

    def f(self, x: np.ndarray, i: None | int | list[int] = None) -> np.ndarray:
            if i == None:
                return np.exp(-np.square(x - self.mu).sum(axis=1) / self.sigma)
            if isinstance(i, int):
                return np.exp(-np.square(x - self.mu[i]).sum() / self.sigma)
            return np.exp(-np.square(x - self.mu[i]).sum(axis=1) / self.sigma)

    def update(self, X: np.ndarray, X_rewards: np.ndarray):
        """
        X: Array of shape (num_states, D)
        X_rewards: Array of shape (num_states,)
        """
        if not self.is_initialized:
            min_vals = np.min(X, axis=0)
            max_vals = np.max(X, axis=0)
            self.mu = np.random.uniform(low=min_vals, high=max_vals, size=(self.K, self.D))
            self.is_initialized = True

        n_iter = 5  # Number of Gauss-Seidel sweeps; adjust as needed
        reward_weights = softmax(X_rewards)
        for _ in range(n_iter):
            print(f"Iteration {_}")
            for i in range(self.K):
                # Data-driven update for centroid i
                # Compute F for all states relative to centroid i
                F_i = np.exp(-np.sum((X - self.mu[i])**2, axis=1) / self.sigma)  # Shape: (num_states,)
                diff_i = np.sum((F_i*reward_weights)[:, None] * (X - self.mu[i]), axis=0)  # Shape: (D,)

                # Compute neighboring influence for centroid i
                # Use a mask to exclude centroid i
                mask = np.arange(self.K) != i
                neighbors = self.mu[mask]  # (K-1, D)
                # Compute weights between centroid i and all other centroids
                weights = np.exp(-np.sum((neighbors - self.mu[i])**2, axis=1) / self.sigma)  # (K-1,)
                # Sum influence over neighbors
                influence = 2.0 * self.lambda_ * np.sum(weights[:, None] * (neighbors - self.mu[i]), axis=0)

                # Gauss-Seidel update for centroid i
                self.mu[i] += (self.learning_rate / self.sigma) * (diff_i - influence)

        # Update rewards using the final centroids
        F = np.exp(-np.square(X[:, np.newaxis, :] - self.mu).sum(axis=2) / self.sigma)  # Shape: (num_states, K)
        weights = F.sum(axis=0)
        mask = weights != 0
        rewards = (F * X_rewards[:, np.newaxis]).sum(axis=0)
        self.cluster_weights[mask] += weights[mask]
        self.cluster_rewards[mask] *= 1 / (1 + weights[mask] / self.cluster_weights[mask])
        self.cluster_rewards[mask] += rewards[mask] / self.cluster_weights[mask]
        
    def update_transitions(self, x: np.ndarray, state_action_transitions_from, state_action_transitions_to, threshold: float):
        gaussian_weights = np.exp(-np.square(x[:, np.newaxis, :] - self.mu).sum(axis=2) / self.sigma)  # Shape: (num_states, K)

        state_to_cluster = np.argmax(gaussian_weights, axis=1)  # Shape: (num_states,)
        
        max_weights = np.max(gaussian_weights, axis=1)

        transition_counts = defaultdict(lambda: defaultdict(int))

        for action in self.actions:
            for from_state, to_state in zip(
                state_action_transitions_from[action],
                state_action_transitions_to[action],
            ):
                # Skip transition if either state has a maximum weight below the threshold
                if max_weights[from_state] < threshold or max_weights[to_state] < threshold:
                    continue

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
                clustered_transition_probs[action][(from_cluster, to_cluster)] = count / total_transitions

        self.transitions_from = clustered_transitions_from
        self.transitions_to = clustered_transitions_to

    def get_model_attributes(self):
        return (self.mu, self.cluster_rewards, self.transitions_from, self.transitions_to)
