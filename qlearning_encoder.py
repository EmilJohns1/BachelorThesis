import random
import numpy as np
import gymnasium as gym
import time

from util.logger import write_to_json
from util.reward_visualizer import plot_rewards

class Neuron_Encoder:
    def __init__(self, n=10, gaussian_width=0.4):
        self.n = n
        self.gaussian_width = gaussian_width
        self.dim1 = np.linspace(-2.4, 2.4, n)
        self.dim2 = np.linspace(-5, 5, n)
        self.dim3 = np.linspace(-0.21, 0.21, n)
        self.dim4 = np.linspace(-4, 4, n)
        self.codebooks = [self.dim1, self.dim2, self.dim3, self.dim4]

    def encode(self, state):
        encoded = []
        for i, val in enumerate(state):
            distances = val - self.codebooks[i]
            weights = np.exp(-np.square(distances) / self.gaussian_width)
            encoded.append(weights)
        return np.concatenate(encoded)

class RBFQLearningAgent:
    def __init__(self, action_space, encoder,
                 alpha=0.01, gamma=0.99,
                 epsilon=1.0, epsilon_decay=0.995,
                 epsilon_min=0.01):
        self.action_space = action_space
        self.encoder = encoder
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min

        sample_state = np.zeros(4)
        self.n_features = len(self.encoder.encode(sample_state))
        self.n_actions = action_space.n

        self.weights = np.zeros((self.n_features, self.n_actions))

    def predict(self, state):
        features = self.encoder.encode(state)
        return np.dot(features, self.weights)

    def choose_action(self, state):
        if np.random.random() < self.epsilon:
            return self.action_space.sample()
        else:
            q_values = self.predict(state)
            return np.argmax(q_values)

    def learn(self, state, action, reward, next_state, done):
        features = self.encoder.encode(state)
        q_current = np.dot(features, self.weights[:, action])
        
        q_next = np.max(self.predict(next_state))
        target = reward + (0 if done else self.gamma * q_next)
        error = target - q_current
        
        self.weights[:, action] += self.alpha * error * features

        if done:
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

def train_rbf_q_learning(env_name="CartPole-v1", episodes=100, seed=42):
    np.random.seed(seed)
    random.seed(seed)
    
    env = gym.make(env_name)
    env.reset(seed=seed)

    encoder = Neuron_Encoder(n=12, gaussian_width=3.0)
    agent = RBFQLearningAgent(env.action_space, encoder)

    episode_rewards = []

    for ep in range(episodes):
        state, _ = env.reset()
        done = False
        total_reward = 0

        while not done:
            action = agent.choose_action(state)
            next_state, reward, done, truncated, _ = env.step(action)
            total_reward += reward
            agent.learn(state, action, reward, next_state, done or truncated)
            state = next_state

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
            "seed": seed,
        }
    }
    
    write_to_json(data)
    plot_rewards(episode_rewards)

    env.close()

if __name__ == "__main__":
    train_rbf_q_learning()
