import copy
import gymnasium as gym
from scipy.special import log_softmax
from scipy.special import softmax
from clusterer import Clusterer
from transitions.transition_method import Transition_Method
from transitions.transition_model import factory

import numpy as np

import matplotlib.pyplot as plt

from sklearn.cluster import MiniBatchKMeans
from sklearn.cluster import k_means
from sklearn.mixture import GaussianMixture

from util.clustering_alg import Clustering_Type


class Model:
    def __init__(
        self,
        action_space_n,
        discount_factor,
        observation_space,
        k,
        sigma,
        find_k=False,
        lower_k=None,
        upper_k=None,
        step=500,
        transition_method=Transition_Method.Delta_Transition_Variant,
    ):
        if isinstance(observation_space, gym.spaces.box.Box):
            obs_dim = observation_space.shape[0]
        elif isinstance(observation_space, gym.spaces.discrete.Discrete):
            obs_dim = 1
        else:
            raise ValueError("Unsupported observation space type!")
        self.state_dimensions = obs_dim
        self.states: np.ndarray = np.empty((0, obs_dim))
        self.clusterer = Clusterer(
            K=k,
            D=obs_dim,
            sigma=sigma,
            lambda_=0.5,
            learning_rate=0.02,
            action_space_n=action_space_n,
        )
        self.original_states: np.ndarray = np.empty((0, obs_dim))
        self.rewards: np.ndarray = np.empty(0)
        self.original_rewards = np.empty(0)
        self.reward_weights = np.ones(0)

        self.actions: list[int] = list(range(action_space_n))

        self.transition_model = factory(
            method=transition_method, action_space_n=action_space_n
        )

        self.new_transitions_index = np.zeros(len(self.actions), dtype=int)

        self.discount_factor: float = discount_factor
        self.states_mean = np.zeros(obs_dim)
        self.M2 = np.zeros(obs_dim)
        self.states_std = np.ones(obs_dim)
        self.k = k
        self.find_k = find_k
        self.lower_k = lower_k
        self.upper_k = upper_k
        self.step = step

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
                prev_state_index = len(self.states) - 2
                current_state_index = len(self.states) - 1
                self.transition_model.update_transitions(
                    actions[i - 1],
                    (prev_state_index, self.states[prev_state_index]),
                    (current_state_index, self.states[current_state_index]),
                )

    def get_transition_data(self, state, action):
        return (
            self.transition_model.get_transition_query_point(state, action),
            self.transition_model.get_transition_centres(action, self.states),
            self.transition_model.get_centre_rewards(action, self.rewards),
        )

    def check_transition_error(self, action, actual_delta, error_threshold):
        self.transition_model.update_predictions(
            action, actual_delta, error_threshold, self.states
        )

    def add_state(self, new_state):
        self.states = np.vstack((self.states, new_state))
        n = len(self.states)

        delta = new_state - self.states_mean
        self.states_mean += delta / n

        self.M2 += delta * (new_state - self.states_mean)
        self.states_std = np.sqrt(self.M2 / n)

    def scale_rewards(self, log_softmaxed_rewards, new_max=0):
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
        scaled_rewards = self.scale_rewards(
            log_softmaxed_rewards=log_softmax_rewards, new_max=3
        )
        return np.exp(scaled_rewards)

    def find_optimal_k(self, states, rewards):
        """
        Uses the elbow method and second derivative to find the optimal k.

        Parameters:
            states (np.array): Array of states.
            rewards (np.array): Array of rewards (used as weights).
            k_range (tuple): Range of k values to try.

        Returns:
            optimal_k (int): Best value of k based on the elbow method.
        """

        k_min = max(1, int(self.lower_k))
        k_max = max(k_min, int(self.upper_k))
        step = self.step

        k_values = range(k_min, k_max + 1, step)
        print(k_values)

        print(f"Trying k values: {k_values}...")
        inertia_values = []

        for k in k_values:
            print(k)
            kmeans = MiniBatchKMeans(
                n_clusters=k, random_state=42, n_init=10, batch_size=1000
            )
            kmeans.fit(states, sample_weight=rewards)
            inertia = kmeans.inertia_
            inertia_values.append(inertia)

        first_derivative = np.diff(inertia_values)

        second_derivative = np.diff(first_derivative)

        k_temp = k_values[np.argmin(second_derivative) + 1]
        optimal_k = np.max([k_temp, 1])

        plt.figure(figsize=(10, 5))
        plt.plot(k_values, inertia_values, marker="o", label="Inertia")
        plt.xlabel("Number of Clusters (k)")
        plt.ylabel("Inertia")
        plt.title("Elbow Method for Optimal k")
        plt.axvline(
            optimal_k, color="r", linestyle="--", label=f"Optimal k = {optimal_k}"
        )
        plt.legend()
        plt.show()

        return optimal_k

    def run_k_means(self):
        print("Running k-means...")
        self.original_states = self.states
        self.original_rewards = self.rewards

        states_array = self.states
        temperature = 50
        new_rewards = softmax(self.rewards / temperature)

        if np.any(new_rewards == 0):
            print("Warning: Zero values detected in new_rewards!")
            print("Indices with zero values:", np.where(new_rewards == 0))

        if self.find_k:
            print("Running elbow method")
            self.k = self.find_optimal_k(states_array, new_rewards)

        centroids, labels, inertia = k_means(
            X=states_array, n_clusters=self.k, sample_weight=new_rewards
        )

        self.clustered_states = centroids
        self.cluster_labels = labels

        # Kan bruke det nedenfor også, men bruker mye lenger tid grunnet n_init, og presterer ikke merkbart bedre.
        # Må testes mer på bedre PC med forskjellige verdier for n_init.

        # k_means = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
        # k_means.fit(states_array, sample_weight=new_rewards)

        # self.clustered_states = k_means.cluster_centers_
        # self.cluster_labels = k_means.labels_

        print("K-Means clustering completed")

    def update_transitions_and_rewards_for_clusters(self, gaussian_width):
        print("Updating transitions")
        assert self.k == len(self.clustered_states)
        states_array = np.array(self.states)
        new_states = self.transition_model.cluster_transitions(
            states_array=states_array,
            clustered_states=self.clustered_states,
            cluster_labels=self.cluster_labels,
            new_states=np.copy(self.clustered_states),
            k=self.k,
            gaussian_width=gaussian_width,
        )

        num_clusters = len(self.clustered_states)
        cluster_rewards = np.zeros(num_clusters)
        cluster_weights = np.zeros(num_clusters)

        for i, centroid in enumerate(self.clustered_states):
            cluster_indices = np.where(self.cluster_labels == i)[0]
            cluster_states = states_array[cluster_indices]
            cluster_rewards_raw = self.rewards[cluster_indices]

            if len(cluster_states) > 0:
                dist = np.sum(np.square(cluster_states - centroid), axis=1)
                weights = np.exp(-dist / gaussian_width)

                weighted_rewards = np.sum(weights * cluster_rewards_raw)
                total_weight = np.sum(weights)

                cluster_rewards[i] = (
                    weighted_rewards / total_weight if total_weight > 0 else 0
                )
                cluster_weights[i] = total_weight

        self.rewards = cluster_rewards
        self.states = new_states
        self.states_mean = np.mean(self.states, axis=0)
        self.states_std = np.std(self.states, axis=0)

    def run_online_clustering(self, k, gaussian_width):
        self.original_states = self.states
        self.original_rewards = self.rewards
        print(f"Total states: {len(self.states)}")

        X = self.states
        X_rewards = self.rewards
        if self.using_clusters:  # Exclude the centroids from calculations
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
                [
                    x - k
                    for x in self.transition_model.state_action_transitions_from[i][
                        self.new_transitions_index[i] :
                    ]
                ]
            )
            transitions_to_new.append(
                [
                    x - k
                    for x in self.transition_model.state_action_transitions_to[i][
                        self.new_transitions_index[i] :
                    ]
                ]
            )

        self.clusterer.update_transitions(
            x=X,
            state_action_transitions_from=transitions_from_new,
            state_action_transitions_to=transitions_to_new,
            threshold=5e-1,
        )

        # Get the updated centroids, rewards, and transitions from the clusterer.
        new_states, new_rewards, new_transitions_from, new_transitions_to = (
            self.clusterer.get_model_attributes()
        )

        self.states = copy.deepcopy(new_states)
        self.rewards = copy.deepcopy(new_rewards)
        self.transition_model.state_action_transitions_from = copy.deepcopy(
            new_transitions_from
        )
        self.transition_model.state_action_transitions_to = copy.deepcopy(
            new_transitions_to
        )

        # Update the new_transitions_index array for each action.
        # Each element now holds the length of the transitions list for that action.
        new_indices = []
        for i, _ in enumerate(self.actions):
            new_indices.append(len(new_transitions_from[i]))
        self.new_transitions_index = np.array(new_indices)

        self.states_mean = np.mean(self.states, axis=0)
        self.states_std = np.std(self.states, axis=0)
        self.using_clusters = True

    def run_gaussian_mixture(self):
        print("Running gaussian mixture clustering...")
        self.original_states = self.states
        self.original_rewards = self.rewards

        states_array = self.states

        gmm = GaussianMixture(
            n_components=self.k,
            covariance_type="full",
            random_state=42,
            init_params="kmeans",
        )
        gmm.fit(states_array)

        self.clustered_states = gmm.means_
        self.cluster_labels = gmm.predict(states_array)
        print("Finished gaussian mixture.")

    def cluster_states(self, k, gaussian_width, cluster_type):
        if cluster_type is Clustering_Type.K_Means:
            self.run_k_means()
            self.update_transitions_and_rewards_for_clusters(
                gaussian_width=gaussian_width
            )
        elif cluster_type is Clustering_Type.Online_Clustering:
            self.run_online_clustering(k=k, gaussian_width=gaussian_width)
        elif cluster_type is Clustering_Type.Gaussian_Mixture:
            self.run_gaussian_mixture()
            self.update_transitions_and_rewards_for_clusters(
                gaussian_width=gaussian_width
            )

        self.using_clusters = True
        # self.update_splines()
