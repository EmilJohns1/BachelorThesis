import itertools
import os

import numpy as np

import matplotlib.pyplot as plt

import json


def plot_rewards(episode_rewards):
    running_means = np.cumsum(episode_rewards) / np.arange(1, len(episode_rewards) + 1)
    running_stds = [
        np.std(episode_rewards[: i + 1]) for i in range(len(episode_rewards))
    ]

    plt.figure(figsize=(10, 6))
    plt.plot(
        range(1, len(episode_rewards) + 1), episode_rewards, label="Rewards", alpha=0.5
    )
    plt.plot(
        range(1, len(running_means) + 1),
        running_means,
        label="Running Mean",
        color="orange",
    )
    plt.fill_between(
        range(1, len(running_means) + 1),
        np.array(running_means) - np.array(running_stds),
        np.array(running_means) + np.array(running_stds),
        color="orange",
        alpha=0.3,
        label="Mean ± Std",
    )
    plt.xlabel("Episode")
    plt.ylabel("Rewards")
    plt.title("Episode Rewards with Running Mean and Std")
    plt.ylim(0, 500)
    plt.legend()
    plt.show()


def plot_multiple_runs(folder_name, title, field, block=True):
    all_rewards = []

    for filename in os.listdir(folder_name):
        if filename.endswith(".json"):
            with open(os.path.join(folder_name, filename), "r") as f:
                data = json.load(f)
                all_rewards.append(data[field])

    if not all_rewards:
        print("No valid data found in the folder.")
        return

    num_episodes = min(len(rewards) for rewards in all_rewards)
    all_rewards = [
        rewards[:num_episodes] for rewards in all_rewards
    ]
    all_rewards = np.array(all_rewards)

    mean_rewards = np.mean(all_rewards, axis=0)
    std_rewards = np.std(all_rewards, axis=0)
    running_means = np.cumsum(mean_rewards) / np.arange(1, len(mean_rewards) + 1)
    running_stds = [np.std(mean_rewards[: i + 1]) for i in range(len(mean_rewards))]

    plt.figure(figsize=(10, 6))
    for i, rewards in enumerate(all_rewards):
        plt.plot(range(1, num_episodes + 1), rewards, alpha=0.5, label=f"Run {i+1}")

    # Plot mean and standard deviation
    # plt.plot(range(1, num_episodes + 1), mean_rewards, color="black", linewidth=1, label="Mean")
    # plt.fill_between(range(1, num_episodes + 1), mean_rewards - std_rewards, mean_rewards + std_rewards, color="gray", alpha=0.3, label="Mean ± Std")

    # Plot running mean and standard deviation
    plt.plot(
        range(1, len(running_means) + 1),
        running_means,
        label="Running Mean",
        color="black",
    )
    plt.fill_between(
        range(1, len(running_means) + 1),
        np.array(running_means) - np.array(running_stds),
        np.array(running_means) + np.array(running_stds),
        color="black",
        alpha=0.3,
        label="Running Mean ± Std",
    )

    plt.xlabel("Episode")
    plt.ylabel("Rewards")
    plt.title(title)
    plt.ylim(0, 500)
    plt.legend()
    plt.show(block=block)


def plot_avg_rewards_recursive(root_folder, field="testing_rewards", block=True):
    import os

    import numpy as np

    import matplotlib.pyplot as plt

    import json

    folders = []
    for dirpath, _, filenames in os.walk(root_folder):
        if any(f.endswith(".json") for f in filenames):
            folders.append(dirpath)

    for i, dirpath in enumerate(folders):
        all_rewards = []
        for filename in os.listdir(dirpath):
            if filename.endswith(".json"):
                with open(os.path.join(dirpath, filename), "r") as f:
                    data = json.load(f)
                    if field in data:
                        all_rewards.append(data[field])

        if not all_rewards:
            continue

        num_episodes = min(len(r) for r in all_rewards)
        all_rewards = [r[:num_episodes] for r in all_rewards]
        all_rewards = np.array(all_rewards)

        mean_rewards = np.mean(all_rewards, axis=0)
        running_means = np.cumsum(mean_rewards) / np.arange(1, len(mean_rewards) + 1)
        running_stds = [np.std(mean_rewards[: i + 1]) for i in range(len(mean_rewards))]

        plt.figure(figsize=(10, 6))
        plt.plot(
            range(1, len(running_means) + 1),
            running_means,
            label="Running Mean",
            color="black",
        )
        plt.fill_between(
            range(1, len(running_means) + 1),
            np.array(running_means) - np.array(running_stds),
            np.array(running_means) + np.array(running_stds),
            color="black",
            alpha=0.3,
            label="Running Mean ± Std",
        )

        plt.xlabel("Episode")
        plt.ylabel("Rewards")
        plt.title(os.path.relpath(dirpath, root_folder))
        plt.ylim(0, 500)
        plt.legend()
        plt.show(block=(block if i == len(folders) - 1 else False))


def compare_experiments(
    *folders, title="", field="testing_rewards", block=True, labels=None
):
    plt.figure(figsize=(10, 6))

    line_styles = ["-", "--", "-.", ":"]
    markers = ["o", "s", "D", "^", "v", "<", ">", "x", "*", "+", "."]

    style_cycle = itertools.cycle([(ls, m) for ls in line_styles for m in markers])

    for idx, folder in enumerate(folders):
        all_rewards = []
        for filename in os.listdir(folder):
            if filename.endswith(".json"):
                with open(os.path.join(folder, filename), "r") as f:
                    data = json.load(f)
                    all_rewards.append(data[field])

        if not all_rewards:
            print(f"No valid data in {folder}")
            continue

        num_episodes = min(len(r) for r in all_rewards)
        all_rewards = [r[:num_episodes] for r in all_rewards]
        all_rewards = np.array(all_rewards)

        mean_rewards = np.mean(all_rewards, axis=0)
        running_means = np.cumsum(mean_rewards) / np.arange(1, len(mean_rewards) + 1)
        running_stds = [np.std(mean_rewards[: i + 1]) for i in range(len(mean_rewards))]

        linestyle, marker = next(style_cycle)
        label = (
            labels[idx] if labels and idx < len(labels) else os.path.basename(folder)
        )
        plt.plot(
            running_means, label=label, linestyle=linestyle, marker=marker, markevery=50
        )
        plt.fill_between(
            range(len(running_means)),
            np.array(running_means) - np.array(running_stds),
            np.array(running_means) + np.array(running_stds),
            alpha=0.3,
        )

    plt.xlabel("Episode")
    plt.ylabel("Running Mean Reward")
    plt.title(title)
    plt.legend()
    plt.ylim(0, 500)
    plt.show(block=block)
