import random
import numpy as np
from collections import deque

import torch
import torch.nn as nn
import torch.optim as optim

from env_manager import EnvironmentManager
from util.logger import write_to_json
from util.reward_visualizer import plot_rewards


def flatten_state(state):
    return np.array(state).flatten()

# Positional encoding. Trying to find out if it works better than baseline.
class PositionalEncoder:
    def __init__(self, input_dim, num_frequencies=10):
        self.input_dim = input_dim
        self.num_frequencies = num_frequencies
        self.freq_bands = 2.0 ** np.arange(num_frequencies)

    def encode(self, x: np.ndarray) -> np.ndarray:
        encoded = [x]
        for freq in self.freq_bands:
            encoded.append(np.sin(freq * x))
            encoded.append(np.cos(freq * x))
        return np.concatenate(encoded)


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
    def __init__(self, capacity=10000):
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


class DQNAgent:
    def __init__(
        self,
        obs_shape,
        action_dim,
        lr=1e-3,
        gamma=0.99,
        epsilon_start=1.0,
        epsilon_end=0.01,
        epsilon_decay=500,
        buffer_capacity=10000,
        batch_size=64,
        target_update_freq=100,
        use_encoder=False,
        pos_enc_freqs=10
    ):

        if isinstance(obs_shape, (tuple, list)):
            self.raw_dim = int(np.prod(obs_shape))
        else:
            self.raw_dim = int(obs_shape)

        self.use_encoder = use_encoder
        self.pos_enc_freqs = pos_enc_freqs
        if self.use_encoder:
            self.pos_encoder = PositionalEncoder(self.raw_dim, pos_enc_freqs)
            self.input_dim = self.raw_dim * (1 + 2 * pos_enc_freqs)
        else:
            self.input_dim = self.raw_dim

        self.action_dim = action_dim
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.epsilon_step = 0
        self.batch_size = batch_size


        self.main_net = DQNNetwork(self.input_dim, action_dim)
        self.target_net = DQNNetwork(self.input_dim, action_dim)
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
        flat = flatten_state(state)
        if self.use_encoder:
            return self.pos_encoder.encode(flat)
        return flat

    def select_action(self, state):
        self.epsilon_step += 1
        self.epsilon = max(
            self.epsilon_end,
            self.epsilon - (1.0 - self.epsilon_end) / self.epsilon_decay
        )
        if random.random() < self.epsilon:
            return random.randrange(self.action_dim)
        proc = self.process_state(state)
        tensor = torch.FloatTensor(proc).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_vals = self.main_net(tensor)
        return q_vals.argmax(dim=1).item()

    def store(self, state, action, reward, next_state, done):
        self.replay_buffer.push(state, action, reward, next_state, done)

    def update(self):
        if len(self.replay_buffer) < self.batch_size:
            return
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)
        states = np.array([self.process_state(s) for s in states])
        next_states = np.array([self.process_state(s) for s in next_states])

        states_t = torch.FloatTensor(states).to(self.device)
        actions_t = torch.LongTensor(actions).unsqueeze(1).to(self.device)
        rewards_t = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_states_t = torch.FloatTensor(next_states).to(self.device)
        dones_t = torch.FloatTensor(dones).unsqueeze(1).to(self.device)

        q_values = self.main_net(states_t)
        q_value = q_values.gather(1, actions_t)

        with torch.no_grad():
            next_q = self.target_net(next_states_t).max(dim=1, keepdim=True)[0]
            target_q = rewards_t + self.gamma * (1.0 - dones_t) * next_q

        loss = nn.MSELoss()(q_value, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.learn_step_counter += 1
        if self.learn_step_counter % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.main_net.state_dict())

    def save(self, path="dqn_model.pth"):
        torch.save(self.main_net.state_dict(), path)

    def load(self, path="dqn_model.pth"):
        self.main_net.load_state_dict(torch.load(path))
        self.target_net.load_state_dict(self.main_net.state_dict())


def train_dqn(
    env_name: str = "CartPole-v1",
    episodes: int = 100,
    max_steps: int = 500,
    lr: float = 1e-3,
    gamma: float = 0.99,
    epsilon_start: float = 1.0,
    epsilon_end: float = 0.01,
    epsilon_decay: int = 500,
    buffer_capacity: int = 5000,
    batch_size: int = 64,
    target_update_freq: int = 100,
    seed: int = None,
    use_encoder: bool = False,
    pos_enc_freqs: int = 10
):

    if seed is None:
        seed = random.randint(0, 2**32 - 1)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    env_manager = EnvironmentManager(
        render_mode=None,
        seed=seed,
        environment=env_name
    )
    obs_shape = env_manager.env.observation_space.shape
    action_dim = env_manager.env.action_space.n

    agent = DQNAgent(
        obs_shape,
        action_dim,
        lr=lr,
        gamma=gamma,
        epsilon_start=epsilon_start,
        epsilon_end=epsilon_end,
        epsilon_decay=epsilon_decay,
        buffer_capacity=buffer_capacity,
        batch_size=batch_size,
        target_update_freq=target_update_freq,
        use_encoder=use_encoder,
        pos_enc_freqs=pos_enc_freqs
    )

    episode_rewards = []
    for ep in range(1, episodes + 1):
        state, _ = env_manager.reset()
        total_reward = 0
        for _ in range(max_steps):
            action = agent.select_action(state)
            next_state, reward, done, truncated, _ = env_manager.step(action)
            total_reward += reward
            agent.store(state, action, reward, next_state, done or truncated)
            agent.update()
            state = next_state
            if done or truncated:
                break
        episode_rewards.append(total_reward)
        print(f"Episode {ep}/{episodes}  Reward: {total_reward:.2f}  Epsilon: {agent.epsilon:.3f}")

    # Log & visualize
    write_to_json({
        "episode_rewards": episode_rewards,
        "parameters": {
            "env_name": env_name,
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
            "use_encoder": use_encoder,
            "pos_enc_freqs": pos_enc_freqs
        }
    })
    plot_rewards(episode_rewards)
    env_manager.close()

    env_vis = EnvironmentManager(
        render_mode="human",
        seed=seed,
        environment=env_name
    )
    state, _ = env_vis.reset()
    print("Training complete. Starting human-rendered mode (CTRL+C to exit)")
    try:
        while True:
            action = agent.select_action(state)
            state, _, done, truncated, _ = env_vis.step(action)
            if done or truncated:
                state, _ = env_vis.reset()
    except KeyboardInterrupt:
        pass
    finally:
        env_vis.close()


if __name__ == "__main__":
    train_dqn()
