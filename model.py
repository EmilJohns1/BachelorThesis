from collections import defaultdict
from scipy.cluster.vq import kmeans2
from scipy.cluster.vq import vq
from scipy.cluster.vq import whiten
from scipy.special import softmax

import numpy as np

import matplotlib.pyplot as plt

from sklearn.cluster import k_means


class Model:
    def __init__(self, action_space_n, _discount_factor, _observation_space):
        obs_dim = _observation_space.shape[0]
        self.states: np.ndarray = np.empty((0, obs_dim))  # States are stored here
        self.original_states: np.ndarray = np.empty(
            (0, obs_dim)
        )  # States are stored here
        self.rewards: np.ndarray = np.empty(0)  # Value for each state index
        self.reward_weights = np.ones(0)

        self.actions: list[int] = list(range(action_space_n))
        # Lists for each action containing from and to state indices, i.e.
        # in which state the action was performed and the resulting state of that action
        self.state_action_transitions_from: list[list[int]] = [[] for _ in self.actions]
        self.state_action_transitions_to: list[list[int]] = [[] for _ in self.actions]

        self.discount_factor: float = (
            _discount_factor  # Low discount factor penalizes longer episodes
        )
        self.states_mean = np.zeros(obs_dim)
        self.M2 = np.zeros(obs_dim)
        self.states_std = np.ones(obs_dim)

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

    def scale_rewards(self, new_min=0.01, new_max=100.0):
        print("Scaling rewards...")
        print(
            f"Rewards before scaling: {self.rewards}"
        )  # Debug: Check the values of rewards
        rewards = np.array(self.rewards)
        min_reward = np.min(self.rewards)
        max_reward = np.max(self.rewards)

        if max_reward == min_reward:
            print("Rewards have no variation, scaling skipped.")
            return rewards

        scaled_rewards = ((rewards - min_reward) / (max_reward - min_reward)) * (
            new_max - new_min
        ) + new_min
        return scaled_rewards

    def run_k_means(self, k):
        print("Running k-means...")

        self.original_states = self.states

        states_array = np.array(self.states)

        shifted_rewards = self.scale_rewards()

        new_rewards = np.power(shifted_rewards, 3)

        print(f"New rewards: {new_rewards}")

        centroids, labels, inertia = k_means(
            X=states_array, n_clusters=k, sample_weight=new_rewards
        )

        self.clustered_states = centroids
        self.cluster_labels = labels

    def update_transitions_and_rewards_for_clusters(self):
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
        gaussian_width = 0.2
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
        print(
            "Rewards length:{}\nStates length: {}".format(
                len(self.rewards), len(self.clustered_states)
            )
        )
