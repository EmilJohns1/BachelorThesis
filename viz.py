import os
import json
import numpy as np
import matplotlib.pyplot as plt


def compare_phase_runs(folders, phase, title, field="rewards", labels=None, block=True):
    """
    Compare the running mean rewards for a given phase ('training' or 'testing')
    across one or more experiment folders.
    """
    plt.figure(figsize=(10, 6))
    for idx, folder in enumerate(folders):
        all_rewards = []
        for filename in os.listdir(folder):
            if filename.endswith(".json"):
                with open(os.path.join(folder, filename), "r") as f:
                    data = json.load(f)
                if data.get("phase") != phase:
                    continue
                all_rewards.append(data[field])

        if not all_rewards:
            print(f"No '{phase}' data found in {folder}")
            continue

        # Trim to shortest run
        num_eps = min(len(r) for r in all_rewards)
        arr = np.array([r[:num_eps] for r in all_rewards])

        # Compute running mean and std of the mean rewards
        mean_rewards = arr.mean(axis=0)
        running = np.cumsum(mean_rewards) / np.arange(1, len(mean_rewards) + 1)
        running_std = [np.std(mean_rewards[:i+1]) for i in range(len(mean_rewards))]

        label = labels[idx] if labels and idx < len(labels) else os.path.basename(folder)
        plt.plot(range(1, len(running) + 1), running, label=label)
        plt.fill_between(
            range(1, len(running) + 1),
            np.array(running) - np.array(running_std),
            np.array(running) + np.array(running_std),
            alpha=0.3,
        )

    plt.xlabel("Episode")
    plt.ylabel("Running Mean Reward")
    plt.title(title)
    plt.legend()
    plt.ylim(0, 500)
    plt.show(block=block)


if __name__ == "__main__":
    # Path to your logs directory
    logs_folder = os.path.join(os.getcwd(), "logs")
    dqn_folder = os.path.join(logs_folder, "dqn")
    enc_folder = os.path.join(logs_folder, "dqn-encoder")

    # 1. DQN Training
    compare_phase_runs(
        [dqn_folder],
        phase="training",
        title="DQN Training",
        labels=["DQN"],
        block=False
    )

    # 2. DQN Testing
    compare_phase_runs(
        [dqn_folder],
        phase="testing",
        title="DQN Testing",
        labels=["DQN"],
        block=False
    )

    # 3. Encoder Training
    compare_phase_runs(
        [enc_folder],
        phase="training",
        title="DQN-Encoder Training",
        labels=["DQN-Encoder"],
        block=False
    )

    # 4. Encoder Testing
    compare_phase_runs(
        [enc_folder],
        phase="testing",
        title="DQN-Encoder Testing",
        labels=["DQN-Encoder"],
        block=False
    )

    # 5. Compare Testing DQN vs Encoder
    compare_phase_runs(
        [dqn_folder, enc_folder],
        phase="testing",
        title="DQN vs DQN-Encoder Testing",
        labels=["DQN", "DQN-Encoder"],
        block=False
    )

    # 6. Compare Training DQN vs Encoder
    compare_phase_runs(
        [dqn_folder, enc_folder],
        phase="training",
        title="DQN vs DQN-Encoder Training",
        labels=["DQN", "DQN-Encoder"],
        block=True
    )

    print("Visualization complete.")
