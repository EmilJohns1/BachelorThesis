import random
import time
import gymnasium as gym

import numpy as np

from util.logger import write_to_json


def create_bins():
    bins = [
        np.linspace(-2.4, 2.4, 10)[1:-1],
        np.linspace(-5.0, 5.0, 10)[1:-1],
        np.linspace(-0.418, 0.418, 10)[1:-1],
        np.linspace(-3.5, 3.5, 10)[1:-1],
    ]
    return bins


def discretize_state(state, bins):
    state_idx = []
    for i, val in enumerate(state):
        state_idx.append(np.digitize(val, bins[i]))
    return tuple(state_idx)


class QLearningAgent:
    def __init__(
        self,
        action_space,
        state_bins,
        alpha=1.0,
        gamma=1.0,
        epsilon=1.0,
        epsilon_min=0.1,
        epsilon_decay=0.99,
    ):
        self.action_space = action_space
        self.state_bins = state_bins
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay

        dims = tuple(len(b) + 1 for b in state_bins) + (action_space.n,)
        self.q_table = np.zeros(dims)

    def choose_action(self, state):
        if np.random.random() < self.epsilon:
            return self.action_space.sample()
        return np.argmax(self.q_table[state])

    def learn(self, state, action, reward, next_state, done):
        old = self.q_table[state][action]
        nxt = np.max(self.q_table[next_state])
        update = (1 - self.alpha) * old + self.alpha * (
            reward + self.gamma * nxt * (not done)
        )
        self.q_table[state][action] = update
        if done:
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)


def train_q_learning(
    env_name: str = "CartPole-v1",
    episodes: int = 100,
    testing_episodes: int = 100,
    runs: int = 100,
    seed: int = 42,
):
    # set global seed
    random.seed(seed)
    np.random.seed(seed)

    bins = create_bins()

    for run in range(1, runs + 1):

        seed_train = random.randint(0, 2**32 - 1)
        seed_test = random.randint(0, 2**32 - 1)

        # Training phase
        random.seed(seed_train)
        np.random.seed(seed_train)
        env_train = gym.make(env_name)
        env_train.reset(seed=seed_train)
        agent = QLearningAgent(env_train.action_space, bins)
        train_rewards = []

        for ep in range(1, episodes + 1):
            state, _ = env_train.reset()
            dstate = discretize_state(state, bins)
            done = False
            total = 0.0
            while not done:
                action = agent.choose_action(dstate)
                next_s, r, done, truncated, _ = env_train.step(action)
                total += r
                nd = discretize_state(next_s, bins)
                agent.learn(dstate, action, r, nd, done or truncated)
                dstate = nd
            train_rewards.append(total)
            print(
                f"[Run {run} Train] Ep {ep}/{episodes} → Reward: {total:.2f}, Epsilon: {agent.epsilon:.3f}"
            )
        env_train.close()

        write_to_json(
            {
                "title": f"train{run}",
                "phase": "training",
                "environment": env_name,
                "seed": seed_train,
                "episodes": episodes,
                "rewards": train_rewards,
                "alpha": agent.alpha,
                "gamma": agent.gamma,
                "epsilon_min": agent.epsilon_min,
                "epsilon_decay": agent.epsilon_decay,
            }
        )

        # short delay before testing
        time.sleep(3)

        # Testing phase (greedy)
        random.seed(seed_test)
        np.random.seed(seed_test)
        env_test = gym.make(env_name)
        env_test.reset(seed=seed_test)
        agent.epsilon = 0.0
        test_rewards = []

        for ep in range(1, testing_episodes + 1):
            state, _ = env_test.reset()
            dstate = discretize_state(state, bins)
            done = False
            total = 0.0
            while not done:
                action = np.argmax(agent.q_table[dstate])
                next_s, r, done, truncated, _ = env_test.step(action)
                total += r
                dstate = discretize_state(next_s, bins)
                if done or truncated:
                    break
            test_rewards.append(total)
            print(f"[Run {run} Test ] Ep {ep}/{testing_episodes} → Reward: {total:.2f}")
        env_test.close()

        write_to_json(
            {
                "title": f"test{run}",
                "phase": "testing",
                "environment": env_name,
                "seed": seed_test,
                "episodes": testing_episodes,
                "rewards": test_rewards,
                "alpha": agent.alpha,
                "gamma": agent.gamma,
                "epsilone": agent.epsilon,
            }
        )

        time.sleep(3)


if __name__ == "__main__":
    train_q_learning()
