import numpy as np
from scipy.cluster.vq import vq, whiten
from scipy.cluster.vq import kmeans2
from scipy.special import softmax
from sklearn.cluster import k_means
import matplotlib.pyplot as plt
from collections import defaultdict

class Model:
    def __init__(self, action_space_n, _discount_factor, _observation_space):
        self.states: np.ndarray = np.empty((0, _observation_space.shape[0]))  # States are stored here
        self.rewards: np.ndarray = np.empty(0)  # Value for each state index
        self.reward_weights = np.ones(0)

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
    
    def scale_rewards(self, new_min=0.01, new_max=100.0):
        print("Scaling rewards...")
        print(f"Rewards before scaling: {self.rewards}")  # Debug: Check the values of rewards
        rewards = np.array(self.rewards)
        min_reward = np.min(self.rewards)
        max_reward = np.max(self.rewards)

        if max_reward == min_reward:
            print("Rewards have no variation, scaling skipped.")
            return rewards

        scaled_rewards = ((rewards - min_reward) / (max_reward - min_reward)) * (new_max - new_min) + new_min
        return scaled_rewards

    def run_k_means(self, k):
        print("Running k-means...")

        states_array = np.array(self.states)

        shifted_rewards = self.scale_rewards()

        new_rewards = np.power(shifted_rewards, 3)
        
        print(f"New rewards: {new_rewards}")

        centroids, labels, inertia = k_means(X=states_array, n_clusters=k, sample_weight=new_rewards)

        self.clustered_states = centroids
        self.cluster_labels = labels

    
    def update_transitions_and_rewards_for_clusters(self):
      state_to_cluster = {i: self.cluster_labels[i] for i in range(len(self.states))}

      transition_counts = defaultdict(lambda: defaultdict(int))
      
      cluster_reward_sums = np.zeros(len(self.clustered_states))
      cluster_reward_counts = np.zeros(len(self.clustered_states))

      for action in self.actions:
          for from_state, to_state in zip(self.state_action_transitions_from[action], self.state_action_transitions_to[action]):
              from_cluster = state_to_cluster[from_state]
              to_cluster = state_to_cluster[to_state]

              transition_counts[(from_cluster, action)][to_cluster] += 1

              cluster_reward_sums[to_cluster] += self.rewards[to_state]
              cluster_reward_counts[to_cluster] += 1

      self.cluster_rewards = np.zeros(len(self.clustered_states))
      for cluster in range(len(self.clustered_states)):
          if cluster_reward_counts[cluster] > 0:
              self.cluster_rewards[cluster] = cluster_reward_sums[cluster] / cluster_reward_counts[cluster]
          else:
              self.cluster_rewards[cluster] = 0 

      clustered_transitions_from = [[] for _ in self.actions]
      clustered_transitions_to = [[] for _ in self.actions]
      clustered_transition_probs = [{} for _ in self.actions]

      for (from_cluster, action), to_clusters in transition_counts.items():
          total_transitions = sum(to_clusters.values())
          
          for to_cluster, count in to_clusters.items():
              clustered_transitions_from[action].append(from_cluster)
              clustered_transitions_to[action].append(to_cluster)
              clustered_transition_probs[action][(from_cluster, to_cluster)] = count / total_transitions

      self.state_action_transitions_from = clustered_transitions_from
      self.state_action_transitions_to = clustered_transitions_to
      self.transition_probs = clustered_transition_probs

      self.rewards = self.cluster_rewards
      self.states = self.clustered_states

      print(f"Total number of actions: {len(self.state_action_transitions_from)}")
      for action in range(len(self.state_action_transitions_from)):
          print(f"Action {action}: Unique from clusters = {len(set(clustered_transitions_from[action]))}, "
                f"Unique to clusters = {len(set(clustered_transitions_to[action]))}")
      
      print(f"Expected cluster count: {len(self.states)} (should be {len(self.clustered_states)})")
      print(f"Expected reward array size: {len(self.rewards)} (should be {len(self.clustered_states)})")