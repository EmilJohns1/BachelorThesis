import random
import numpy as np
import gymnasium as gym
import time

from util.logger import write_to_json
from util.reward_visualizer import plot_rewards

def create_bins():
    bins = [
        np.linspace(-4.8, 4.8, 6)[1:-1],
        np.linspace(-3.0, 3.0, 6)[1:-1],
        np.linspace(-0.418, 0.418, 6)[1:-1],
        np.linspace(-3.5, 3.5, 6)[1:-1]
    ]
    return bins

def discretize_state(state, bins):
    state_idx = []
    for i, val in enumerate(state):
        state_idx.append(np.digitize(val, bins[i]))
    return tuple(state_idx)

class QLearningAgent:
    def __init__(self, action_space, state_bins,
                 alpha=0.1, gamma=0.8,
                 epsilon=1.0, epsilon_min=0.01,
                 epsilon_decay=0.995):
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
        else:
            return np.argmax(self.q_table[state])

    def learn(self, state, action, reward, next_state, done):
        old_value = self.q_table[state][action]
        next_max = np.max(self.q_table[next_state])

        new_value = (1 - self.alpha) * old_value + self.alpha * (reward + self.gamma * next_max * (not done))
        self.q_table[state][action] = new_value

        if done:
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

def train_q_learning(env_name="CartPole-v1", episodes=100, seed=42):
    np.random.seed(seed)
    random.seed(seed)

    env = gym.make(env_name)
    env.reset(seed=seed)

    bins = create_bins()
    agent = QLearningAgent(env.action_space, bins)

    episode_rewards = []

    for ep in range(episodes):
        state, _ = env.reset()
        discrete_state = discretize_state(state, bins)
        done = False
        total_reward = 0

        while not done:
            action = agent.choose_action(discrete_state)
            next_state, reward, done, truncated, _ = env.step(action)
            total_reward += reward
            next_discrete_state = discretize_state(next_state, bins)
            agent.learn(discrete_state, action, reward, next_discrete_state, done or truncated)
            discrete_state = next_discrete_state

        episode_rewards.append(total_reward)
        print(f"Episode {ep+1}, Reward: {total_reward}, Epsilon: {agent.epsilon:.3f}")

    data = {
        "episode_rewards": episode_rewards,
        "parameters": {
            "episodes": episodes,
            "alpha": agent.alpha,
            "gamma": agent.gamma,
            "epsilon": agent.epsilon,
            "epsilon_min": agent.epsilon_min,
            "epsilon_decay": agent.epsilon_decay,
            "seed": seed
        }
    }
    write_to_json(data)
    plot_rewards(episode_rewards)
    env.close()

if __name__ == "__main__":
    train_q_learning()
