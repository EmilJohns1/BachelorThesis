import random
import numpy as np
from collections import deque

import torch
import torch.nn as nn
import torch.optim as optim

from env_manager import EnvironmentManager
from util.logger import write_to_json
from util.reward_visualizer import plot_rewards


class DQNNetwork(nn.Module):
    def __init__(self, input_dim, action_dim, hidden_size=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, action_dim)
        )

    def forward(self, x):
        return self.net(x)


class ReplayBuffer:
    def __init__(self, capacity=100):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size=64):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            np.array(states),
            np.array(actions),
            np.array(rewards, dtype=np.float32),
            np.array(next_states),
            np.array(dones, dtype=np.float32)
        )

    def __len__(self):
        return len(self.buffer)


class Neuron_Encoder:
    def __init__(self, n=10):
        self.dim1 = np.linspace(-2.4, 2.4, n)
        self.dim2 = np.linspace(-5, 5, n)
        self.dim3 = np.linspace(-0.21, 0.21, n)
        self.dim4 = np.linspace(-4, 4, n)
        self.codebooks = [self.dim1, self.dim2, self.dim3, self.dim4]

    def encode(self, state):
        gaussian_width = 0.4
        encoded = []
        for i, val in enumerate(state):
            distances = val - self.codebooks[i]
            weights = np.exp(-np.square(distances) / gaussian_width)
            encoded.append(weights)
        return np.concatenate(encoded)


class DQNAgent:
    def __init__(
        self,
        state_dim,
        action_dim,
        lr=1e-3,
        gamma=0.99,
        epsilon_start=1.0,
        epsilon_end=0.01,
        epsilon_decay=5,
        buffer_capacity=10000,
        batch_size=64,
        target_update_freq=100,
        use_encoder=False,
        encoder_n=10
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.epsilon_step = 0
        self.batch_size = batch_size
        self.use_encoder = use_encoder

        if self.use_encoder:
            self.encoder = Neuron_Encoder(n=encoder_n)
            input_dim = encoder_n * 4  # 4 state variables
        else:
            input_dim = state_dim

        self.main_net = DQNNetwork(input_dim, action_dim)
        self.target_net = DQNNetwork(input_dim, action_dim)
        self.target_net.load_state_dict(self.main_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.main_net.parameters(), lr=lr)
        self.replay_buffer = ReplayBuffer(capacity=buffer_capacity)
        self.target_update_freq = target_update_freq
        self.learn_step_counter = 0

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.main_net.to(self.device)
        self.target_net.to(self.device)

    def process_state(self, state):
        if self.use_encoder:
            return self.encoder.encode(state)
        else:
            return state

    def select_action(self, state):
        self.epsilon_step += 1
        self.epsilon = max(
            self.epsilon_end,
            1.0 - (self.epsilon_step / self.epsilon_decay) * (1.0 - self.epsilon_end)
        )
        if random.random() < self.epsilon:
            return random.randint(0, self.action_dim - 1)
        else:
            processed_state = self.process_state(state)
            state_tensor = torch.FloatTensor(processed_state).unsqueeze(0).to(self.device)
            with torch.no_grad():
                q_values = self.main_net(state_tensor)
            return q_values.argmax(dim=1).item()

    def store(self, state, action, reward, next_state, done):
        self.replay_buffer.push(state, action, reward, next_state, done)

    def update(self):
        if len(self.replay_buffer) < self.batch_size:
            return

        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)

        # If using the encoder, process each state individually.
        if self.use_encoder:
            states = np.array([self.encoder.encode(s) for s in states])
            next_states = np.array([self.encoder.encode(s) for s in next_states])

        states_tensor = torch.FloatTensor(states).to(self.device)
        actions_tensor = torch.LongTensor(actions).to(self.device)
        rewards_tensor = torch.FloatTensor(rewards).unsqueeze(-1).to(self.device)
        next_states_tensor = torch.FloatTensor(next_states).to(self.device)
        dones_tensor = torch.FloatTensor(dones).unsqueeze(-1).to(self.device)

        q_values = self.main_net(states_tensor)
        q_value = q_values.gather(1, actions_tensor.unsqueeze(1))

        with torch.no_grad():
            next_q_values = self.target_net(next_states_tensor)
            next_q_value = next_q_values.max(dim=1, keepdim=True)[0]
            target_q_value = rewards_tensor + self.gamma * (1.0 - dones_tensor) * next_q_value

        loss = nn.MSELoss()(q_value, target_q_value)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.learn_step_counter += 1
        if self.learn_step_counter % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.main_net.state_dict())

    def save(self, path="dqn_cartpole.pth"):
        torch.save(self.main_net.state_dict(), path)

    def load(self, path="dqn_cartpole.pth"):
        self.main_net.load_state_dict(torch.load(path))
        self.target_net.load_state_dict(self.main_net.state_dict())


def train_dqn_encoder(
    episodes=100,
    max_steps=500,
    lr=1e-3,
    gamma=0.99,
    epsilon_start=1.0,
    epsilon_end=0.01,
    epsilon_decay=20,
    buffer_capacity=10000,
    batch_size=64,
    target_update_freq=100,
    use_encoder=False,  # Set True to use the Neuron_Encoder
    seed=random.randint(0, 2 ** 32 - 1)
):
    env_manager = EnvironmentManager(env_name="CartPole-v1", render_mode=None, seed=seed)
    state_dim = env_manager.env.observation_space.shape[0]
    action_dim = env_manager.env.action_space.n

    agent = DQNAgent(
        state_dim,
        action_dim,
        lr=lr,
        gamma=gamma,
        epsilon_start=epsilon_start,
        epsilon_end=epsilon_end,
        epsilon_decay=epsilon_decay,
        buffer_capacity=buffer_capacity,
        batch_size=batch_size,
        target_update_freq=target_update_freq,
        use_encoder=use_encoder
    )

    episode_rewards = []
    total_steps = 0

    for ep in range(episodes):
        state, _ = env_manager.reset()
        episode_reward = 0

        for step in range(max_steps):
            action = agent.select_action(state)
            next_state, reward, done, truncated, _ = env_manager.step(action)
            episode_reward += reward

            # Store the transition (combine done and truncated)
            agent.store(state, action, reward, next_state, done or truncated)
            agent.update()

            state = next_state
            total_steps += 1
            if done or truncated:
                break

        episode_rewards.append(episode_reward)
        print(f"Episode {ep+1}, Reward: {episode_reward}, Epsilon: {agent.epsilon:.3f}")

    data = {
        "episode_rewards": episode_rewards,
        "parameters": {
            "episodes": episodes,
            "max_steps": max_steps,
            "lr": lr,
            "gamma": gamma,
            "epsilon_start": epsilon_start,
            "epsilon_end": epsilon_end,
            "epsilon_decay": epsilon_decay,
            "buffer_capacity": buffer_capacity,
            "batch_size": batch_size,
            "target_update_freq": target_update_freq,
            "use_encoder": use_encoder
        }
    }
    write_to_json(data)
    plot_rewards(episode_rewards)

    env_manager.close()

    env_h = EnvironmentManager(render_mode="human")
    state, _ = env_h.reset()
    done = False
    truncated = False
    while not (done or truncated):
        action = agent.select_action(state)
        state, reward, done, truncated, _ = env_h.step(action)
    env_h.close()


if __name__ == "__main__":
    # Set use_encoder=True to enable the neuron encoder
    train_dqn_encoder(use_encoder=True)
