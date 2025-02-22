from mpl_toolkits.mplot3d import Axes3D

import numpy as np

import matplotlib.pyplot as plt
from matplotlib import colormaps

from sklearn.cluster import AgglomerativeClustering
from sklearn.manifold import TSNE


class ClusterVisualizer:
    def __init__(self, model):
        self.model = model

    def plot_clusters(self):
        fig, axes = plt.subplots(1, 2, figsize=(16, 8))  # Two side-by-side plots

        ### PLOT 1: Original States - Colored by Original Rewards ###
        if self.model.original_states.shape[1] > 2:
            tsne = TSNE(n_components=2, random_state=42)
            original_states_2d = tsne.fit_transform(self.model.original_states)
        else:
            original_states_2d = self.model.original_states

        original_rewards = self.model.original_rewards  # Get original rewards
        scatter1 = axes[0].scatter(
            original_states_2d[:, 0],
            original_states_2d[:, 1],
            c=original_rewards,  # Color by original rewards
            cmap="viridis",
            s=25,
            alpha=0.8,
            edgecolors="k"
        )

        cbar1 = plt.colorbar(scatter1, ax=axes[0])
        cbar1.set_label("Original Reward Value")

        axes[0].set_title("Original States (Colored by Original Rewards)")
        axes[0].set_xlabel("TSNE Component 1")
        axes[0].set_ylabel("TSNE Component 2")

        ### PLOT 2: Clustered States - Colored by Clustered Rewards ###
        if self.model.states.shape[1] > 2:
            tsne = TSNE(n_components=2, random_state=42)
            clustered_states_2d = tsne.fit_transform(self.model.states)
        else:
            clustered_states_2d = self.model.states

        clustered_rewards = self.model.rewards  # Get clustered rewards
        scatter2 = axes[1].scatter(
            clustered_states_2d[:, 0],
            clustered_states_2d[:, 1],
            c=clustered_rewards,  # Color by clustered rewards
            cmap="viridis",
            s=25,
            alpha=0.8,
            edgecolors="k"
        )

        cbar2 = plt.colorbar(scatter2, ax=axes[1])
        cbar2.set_label("Clustered Reward Value")

        axes[1].set_title("Clustered States (Colored by Clustered Rewards)")
        axes[1].set_xlabel("TSNE Component 1")
        axes[1].set_ylabel("TSNE Component 2")

        ### SET SAME LIMITS FOR BOTH PLOTS ###
        all_x = np.concatenate((original_states_2d[:, 0], clustered_states_2d[:, 0]))
        all_y = np.concatenate((original_states_2d[:, 1], clustered_states_2d[:, 1]))

        x_min, x_max = all_x.min(), all_x.max()
        y_min, y_max = all_y.min(), all_y.max()

        for ax in axes:
            ax.set_xlim(x_min, x_max)
            ax.set_ylim(y_min, y_max)

        plt.show()

    def plot_rewards(self):
        if self.model.states.shape[1] < 2:
            print("Insufficient dimensions to plot rewards.")
            return

        tsne = TSNE(n_components=3, random_state=42)
        reduced_states = tsne.fit_transform(self.model.states)
        rewards = np.array(self.model.rewards)

        fig = plt.figure(figsize=(10, 7))
        ax = fig.add_subplot(111, projection="3d")

        scatter = ax.scatter(
            reduced_states[:, 0],
            reduced_states[:, 1],
            reduced_states[:, 2],
            c=rewards,
            cmap="coolwarm",
            alpha=0.8,
            s=10,
        )

        ax.set_xlabel("TSNE Feature 1")
        ax.set_ylabel("TSNE Feature 2")
        ax.set_zlabel("TSNE Feature 3")
        ax.set_title("3D Rewards Visualization using TSNE")
        fig.colorbar(scatter, label="Reward Value")

        plt.show()
