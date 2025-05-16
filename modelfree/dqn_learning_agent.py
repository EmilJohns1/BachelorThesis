import random
import numpy as np
from collections import deque

import torch
import torch.nn as nn
import torch.optim as optim

from env_manager import EnvironmentManager
from util.logger import write_to_json

from util.reward_visualizer import plot_multiple_runs, plot_avg_rewards_recursive, compare_experiments



def flatten_state(state):
    return np.array(state).flatten()

# Positional encoding. Trying to find out if it works better than baseline.
class PositionalEncoder:
    def __init__(self, d_model=4):
        self.d_model = d_model

    def encode_scalar(self, x):
        pe = np.zeros(self.d_model, dtype=np.float32)
        for i in range(0, self.d_model, 2):
            # 10000^(i/d_model)
            div_term = np.exp(-np.log(10000.0) * i / self.d_model)
            pe[i]   = np.sin(x * div_term)
            if i + 1 < self.d_model:
                pe[i+1] = np.cos(x * div_term)
        return pe

    def encode(self, state: np.ndarray) -> np.ndarray:
        # state is e.g. [x1, x2, x3, x4]
        # returns [PE(x1), PE(x2), PE(x3), PE(x4)]
        return np.concatenate([self.encode_scalar(x) for x in state])

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
        gamma=1.0,
        epsilon_start=1.0,
        epsilon_end=0.01,
        epsilon_decay=500,
        buffer_capacity=20000,
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
            self.pos_encoder = PositionalEncoder(d_model=pos_enc_freqs)
            self.input_dim = self.raw_dim * pos_enc_freqs
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
    
    def select_greedy_action(self, state):
        # pure greedy for evaluation
        tensor = torch.FloatTensor(self.process_state(state)).unsqueeze(0).to(self.device)
        with torch.no_grad():
            return self.main_net(tensor).argmax(dim=1).item()

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
    testing_episodes: int = 100,
    runs: int = 75,
    max_steps: int = 500,
    lr: float = 1e-3,
    gamma: float = 1.0,
    epsilon_start: float = 1.0,
    epsilon_end: float = 0.01,
    epsilon_decay: int = 500,
    buffer_capacity: int = 5000,
    batch_size: int = 64,
    target_update_freq: int = 100,
    use_encoder: bool = False,
    pos_enc_freqs: int = 10
):
    
    for run in range(26, 26 + runs):
        # New random seed per run
        seed_train = random.randint(0, 2**32 - 1)
        seed_test  = random.randint(0, 2**32 - 1)

        random.seed(seed_train)
        np.random.seed(seed_train)
        torch.manual_seed(seed_train)
        if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed_train)

        env_train = EnvironmentManager(render_mode=None, environment=env_name, seed=seed_train)
        obs_shape = env_train.env.observation_space.shape
        action_dim = env_train.env.action_space.n
        agent = DQNAgent(
            obs_shape, action_dim,
            lr=lr, gamma=gamma,
            epsilon_start=epsilon_start, epsilon_end=epsilon_end, epsilon_decay=epsilon_decay,
            buffer_capacity=buffer_capacity, batch_size=batch_size,
            target_update_freq=target_update_freq,
            use_encoder=use_encoder, pos_enc_freqs=pos_enc_freqs
        )

        # Training phase
        training_rewards = []
        for ep in range(1, episodes + 1):
            state, _ = env_train.reset()
            total = 0.0
            for _ in range(max_steps):
                action = agent.select_action(state)
                next_s, r, done, truncated, _ = env_train.step(action)
                agent.store(state, action, r, next_s, done or truncated)
                agent.update()
                state = next_s
                total += r
                if done or truncated: break
            training_rewards.append(total)
            print(f"[Run {run} Train] Ep {ep}/{episodes} → Reward: {total:.2f}")
        env_train.close()
        write_to_json({
            "title": f"train{run}",
            "phase": "training",
            "environment": env_name,
            "seed": seed_train,
            "episodes": episodes,
            "rewards": training_rewards,

            "lr": lr,
            "gamma": gamma,
            "epsilon_start": epsilon_start,
            "epsilon_end": epsilon_end,
            "epsilon_decay": epsilon_decay,
            "buffer_capacity": buffer_capacity,
            "batch_size": batch_size,
            "target_update_freq": target_update_freq,
            "use_encoder": use_encoder,
            "pos_enc_freqs": pos_enc_freqs,
        })


        # Testing phase
        random.seed(seed_test)
        np.random.seed(seed_test)
        torch.manual_seed(seed_test)
        if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed_test)

        

        # Setup testing env no learning
        env_test = EnvironmentManager(render_mode=None, environment=env_name, seed=seed_test)
        agent.epsilon = 0.0
        agent.main_net.eval()

        testing_rewards = []
        for ep in range(1, testing_episodes + 1):
            state, _ = env_test.reset()
            total = 0.0
            for _ in range(max_steps):
                action = agent.select_greedy_action(state)
                state, r, done, truncated, _ = env_test.step(action)
                total += r
                if done or truncated: break
            testing_rewards.append(total)
            print(f"[Run {run} Test ] Ep {ep}/{testing_episodes} → Reward: {total:.2f}")
        env_test.close()
        write_to_json({
            "title": f"test{run}",
            "phase": "testing",
            "environment": env_name,
            "seed": seed_test,
            "episodes": testing_episodes,
            "rewards": testing_rewards,


            "lr": lr,
            "gamma": gamma,
            "epsilon_during_test": agent.epsilon,
            "buffer_capacity": buffer_capacity,
            "batch_size": batch_size,
            "target_update_freq": target_update_freq,
            "use_encoder": use_encoder,
            "pos_enc_freqs": pos_enc_freqs,

        })

if __name__ == "__main__":
    train_dqn()