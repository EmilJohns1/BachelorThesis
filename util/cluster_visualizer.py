import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

class ClusterVisualizer:
    def __init__(self, model):
        """
        Initializes the visualizer with the trained model.
        
        :param model: The trained model containing clustered states and rewards.
        """
        self.model = model

    def plot_clusters(self, save_path=None):
        """
        Visualizes clusters using t-SNE.
        
        :param save_path: If provided, saves the plot instead of displaying it.
        """
        if len(self.model.states) == 0:
            print("No states available for visualization.")
            return

        tsne = TSNE(n_components=2, random_state=42)
        states_2d = tsne.fit_transform(self.model.states)

        labels = np.array([self.model.cluster_labels[i] for i in range(len(self.model.states))])

        plt.figure(figsize=(8, 6))
        scatter = plt.scatter(states_2d[:, 0], states_2d[:, 1], c=labels, cmap="tab10", alpha=0.7)
        plt.colorbar(scatter, label="Cluster ID")
        plt.title("t-SNE Visualization of State Clusters")
        plt.xlabel("t-SNE Component 1")
        plt.ylabel("t-SNE Component 2")

        if save_path:
            plt.savefig(save_path)
        else:
            plt.show()

    def plot_rewards(self, save_path=None):
        """
        Visualizes rewards using t-SNE, where color represents reward magnitude.
        
        :param save_path: If provided, saves the plot instead of displaying it.
        """
        if len(self.model.states) == 0:
            print("No states available for visualization.")
            return

        tsne = TSNE(n_components=2, random_state=42)
        states_2d = tsne.fit_transform(self.model.states)

        print(f"States_2D shape: {states_2d.shape}")

        # Normalize rewards for better visualization
        rewards = np.array(self.model.rewards).flatten()

        rewards_normalized = (rewards - np.min(rewards)) / (np.max(rewards) - np.min(rewards) + 1e-5)
        rewards_normalized = rewards_normalized[:states_2d.shape[0]]

        plt.figure(figsize=(8, 6))
        scatter = plt.scatter(states_2d[:, 0], states_2d[:, 1], 
                            c=rewards_normalized, cmap="plasma", alpha=0.7)
        plt.colorbar(scatter, label="Normalized Reward")
        plt.title("t-SNE Visualization of Rewards")
        plt.xlabel("t-SNE Component 1")
        plt.ylabel("t-SNE Component 2")

        if save_path:
            plt.savefig(save_path)
        else:
            plt.show()