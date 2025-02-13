import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import colormaps
from sklearn.cluster import AgglomerativeClustering

class ClusterVisualizer:
    def __init__(self, model):
        self.model = model
    
    def plot_clusters(self):
        if self.model.original_states.shape[1] > 2:
            tsne = TSNE(n_components=2, random_state=42)
            states_2d = tsne.fit_transform(self.model.original_states)
        else:
            states_2d = self.model.original_states

        fig, ax = plt.subplots(figsize=(10, 8))

        num_groups = 15
        agglomerative = AgglomerativeClustering(n_clusters=num_groups, metric='euclidean', linkage='ward')
        centroids = self.model.clustered_states
        group_labels = agglomerative.fit_predict(centroids)

        colormap = plt.cm.get_cmap('rainbow_r', num_groups)
        colors = [colormap(label) for label in group_labels]

        label_to_color = {}
        for i, label in enumerate(self.model.cluster_labels):
            group = group_labels[label]
            label_to_color[label] = colors[group]

        scatter = ax.scatter(states_2d[:, 0], states_2d[:, 1], c=[label_to_color[label] for label in self.model.cluster_labels], s=25, alpha=0.8)

        cbar = plt.colorbar(scatter)
        cbar.set_label('Cluster Group')

        ax.set_xlabel('TSNE Component 1')
        ax.set_ylabel('TSNE Component 2')
        ax.set_title('2D Visualization of States and Clusters (Grouped by Proximity)')

        plt.show()

    
    def plot_rewards(self):
      if self.model.states.shape[1] < 2:
          print("Insufficient dimensions to plot rewards.")
          return

      tsne = TSNE(n_components=3, random_state=42)
      reduced_states = tsne.fit_transform(self.model.states)
      rewards = np.array(self.model.rewards)

      fig = plt.figure(figsize=(10, 7))
      ax = fig.add_subplot(111, projection='3d')

      scatter = ax.scatter(
          reduced_states[:, 0], reduced_states[:, 1], reduced_states[:, 2],
          c=rewards, cmap='coolwarm', alpha=0.8, s=10
      )

      ax.set_xlabel("TSNE Feature 1")
      ax.set_ylabel("TSNE Feature 2")
      ax.set_zlabel("TSNE Feature 3")
      ax.set_title("3D Rewards Visualization using TSNE")
      fig.colorbar(scatter, label="Reward Value")

      plt.show()