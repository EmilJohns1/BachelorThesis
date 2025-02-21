import numpy as np
import time
from collections import defaultdict


class Clusterer:
    def __init__(self, K: int, D: int, sigma: float, lambda_: float, learning_rate: float, action_space_n: int):
        self.K = K
        self.D = D
        self.mu = np.random.rand(K, D)
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

    def update(self, x: np.ndarray, x_reward):
        # weighting_function goes here
        f = self.f(x)

        #Update centroids
        diff = f.reshape(-1, 1) * (x - self.mu) - (2.0 * self.lambda_ *
                                                        (self.mu[self.j] - self.mu[self.i]) * self.f(self.mu[self.j], self.i).reshape(-1, 1)).reshape(
                                                            -1, self.K - 1, self.mu.shape[1]).sum(axis=1)
        
        self.mu += self.learning_rate * diff / self.sigma

        #Update Q for labeling
        p = np.inf
        self.f_sum += f
        if p != None:
            f /= np.linalg.norm(f, p)

        self.Q[self.k, self.l] += f[self.k] * f[self.l]

        self.N += 1

        #Update rewards
        weights = f

        mask = weights != 0

        # Apply the mask
        rewards = weights[mask] * x_reward
        self.cluster_weights[mask] += weights[mask]
        self.cluster_rewards[mask] *= 1 / (1 + weights[mask] / self.cluster_weights[mask])
        self.cluster_rewards[mask] += rewards / self.cluster_weights[mask]
        
    def f_min(self) -> float:
        # sqrt(a/2) is the inflection point and equal to solution of d^2/dx^2 e^(-x^2/a) = 0 given x>0, a>0
        return np.exp(-np.square(3. * np.sqrt(self.sigma / 2.) - 0.).sum() / self.sigma)
    
    def get_centroid_labels(self, R_min: float, use_f_min: bool = False) -> np.ndarray:
        """
        `use_f_min = True` (default `False`): the Gaussian functions with negligible average output are disregarded.
        """
        R = np.empty((self.K, self.K))
        R[self.k, self.l] = self.Q[self.k, self.l] / (np.sqrt(self.Q[self.k, self.k]) * np.sqrt(self.Q[self.l, self.l]))

        if use_f_min:
            f_min = self.f_min()
            for k in range(self.model.K):
                if self.f_sum[k] / self.N < f_min:
                    for l in range(self.K):
                        R[k, l] = 0.0
                        R[l, k] = 0.0

        labels = np.zeros(self.K, dtype=int)
        L: int = 0

        def assign(k):
            labels[k] = L
            for l in range(self.K):
                if labels[l] == 0 and R[k, l] > R_min:
                    assign(l)

        for k in range(self.K):
            if labels[k] == 0 and R[k, k] > 0.0:
                L += 1
                assign(k)

        return labels

    def update_transitions(self, x: np.ndarray):
        # Get the labels of the clusters and states
        centroid_labels = self.get_centroid_labels()
        state_to_cluster = np.array([])

        for state in x:
            gaussian_weights = self.f(state)
            closest_centroid_index = np.argmax(gaussian_weights)
            state_label = centroid_labels[closest_centroid_index]
            state_to_cluster.append(state_label)

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

        self.transitions_from = clustered_transitions_from
        self.transitions_to = clustered_transitions_to

    def get_model_attributes(self):
        return (self.mu, self.cluster_rewards, self.transitions_from, self.transitions_to)


        # for all states:
        #   find gaussian weight to all centroids
        #   argmax to get the index of the closest centroid
        #   use index on centroid_labels to get the corresponding label. 
        #   store in list for updating transitions

        
        
        