import numpy as np
import time
from collections import defaultdict


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

        # Attributes for identifying labels for centroids and states
        self.Q = np.zeros((K, K))
        self.f_sum = np.zeros(K)
        self.N = 0.0
        self.k, self.l = np.indices((K, K))

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
        # Compute weights for all states at once
        F = np.exp(-np.square(X[:, np.newaxis, :] - self.mu).sum(axis=2) / self.sigma)  # Shape: (num_states, K)

        # Compute weighted differences for centroids
        diff = (F[..., np.newaxis] * (X[:, np.newaxis, :] - self.mu)).sum(axis=0)  # Shape: (K, D)

        # Compute competitive interactions between centroids (as before)
        # Compute interactions between centroids (Fixing shape issue)
        mu_diff = self.mu[self.j] - self.mu[self.i]  # Shape: (K*(K-1), D)

        # Compute pairwise influence
        centroid_weights = self.f(self.mu[self.j], self.i)  # Shape: (K*(K-1),)
        centroid_weights = centroid_weights.reshape(-1, 1)  # Reshape for broadcasting

        # Sum contributions correctly (Fix: Use sum over the correct axis)
        neighboring_influence = (2.0 * self.lambda_ * mu_diff * centroid_weights).reshape(self.K, self.K - 1, self.D).sum(axis=1)  # Shape: (K, D)

        
        # Update centroids in a single step
        self.mu += (self.learning_rate / self.sigma) * (diff - neighboring_influence)

        # Update rewards
        weights = F.sum(axis=0)  # Aggregate weights per centroid
        mask = weights != 0

        # Aggregate rewards per centroid
        rewards = (F * X_rewards[:, np.newaxis]).sum(axis=0)  

        # Apply updates only where mask is True
        self.cluster_weights[mask] += weights[mask]
        self.cluster_rewards[mask] *= 1 / (1 + weights[mask] / self.cluster_weights[mask])
        self.cluster_rewards[mask] += rewards[mask] / self.cluster_weights[mask]

        f_all = np.exp(-np.square(X[:, np.newaxis, :] - self.mu).sum(axis=2) / self.sigma)

        # Update the cumulative sum of f values
        self.f_sum += np.sum(f_all, axis=0)

        # Normalize each row of f_all by its L-inf norm
        norms = np.max(f_all, axis=1, keepdims=True)
        normalized_f = f_all / norms

        # Update Q by summing the outer products for all states
        self.Q += normalized_f.T @ normalized_f

        # Increment N by the number of states processed
        self.N += X.shape[0]

        
    def f_min(self) -> float:
        # sqrt(a/2) is the inflection point and equal to solution of d^2/dx^2 e^(-x^2/a) = 0 given x>0, a>0
        return np.exp(-np.square(3. * np.sqrt(self.sigma / 2.) - 0.).sum() / self.sigma)
    
    def get_centroid_labels(self, R_min=0.99999, use_f_min: bool = False) -> np.ndarray:
        # Compute R safely, converting any NaNs to 0
        with np.errstate(divide='ignore', invalid='ignore'):
            R = self.Q[self.k, self.l] / (np.sqrt(self.Q[self.k, self.k]) * np.sqrt(self.Q[self.l, self.l]))
            R = np.nan_to_num(R)
            
        if use_f_min:
            f_min = self.f_min()
            for k in range(self.K):
                if self.f_sum[k] / self.N < f_min:
                    R[k, :] = 0.0
                    R[:, k] = 0.0

        labels = np.zeros(self.K, dtype=int)
        L = 0

        # Iterative connectivity assignment using a stack
        for k in range(self.K):
            if labels[k] == 0 and R[k, k] > 0.0:
                L += 1
                stack = [k]
                while stack:
                    current = stack.pop()
                    if labels[current] == 0:
                        labels[current] = L
                        for l in range(self.K):
                            if labels[l] == 0 and R[current, l] > R_min:
                                stack.append(l)
        return labels

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

        
        
        