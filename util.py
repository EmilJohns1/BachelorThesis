import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import json

def plot_rewards(episode_rewards):
    # Calculate running mean and std
    running_means = np.cumsum(episode_rewards) / np.arange(1, len(episode_rewards) + 1)
    running_stds = [np.std(episode_rewards[:i + 1]) for i in range(len(episode_rewards))]
    
    # Plot rewards, mean, and standard deviation
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(episode_rewards) + 1), episode_rewards, label="Rewards", alpha=0.5)
    plt.plot(range(1, len(running_means) + 1), running_means, label="Running Mean", color="orange")
    plt.fill_between(range(1, len(running_means) + 1),
                        np.array(running_means) - np.array(running_stds),
                        np.array(running_means) + np.array(running_stds),
                        color="orange", alpha=0.3, label="Mean ± Std")
    plt.xlabel("Episode")
    plt.ylabel("Rewards")
    plt.title("Episode Rewards with Running Mean and Std")
    plt.ylim(0, 500)
    plt.legend()
    plt.show()

def write_to_json(data):
    timestamp = datetime.now().strftime("%Y_%m_%d_%H-%M-%S")
    filename = f"{timestamp}.json"
    with open(filename, "w") as f:
        json.dump(data, f, indent=4)
