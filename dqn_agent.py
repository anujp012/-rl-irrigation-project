"""
dqn_agent.py

ONE shared Q-network used by every zone (parameter sharing across agents).
Each zone queries the same network with its own local observation and gets
its own action index back; every zone's experience feeds the same replay
buffer.
"""
import random
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

OBS_DIM = 8
N_ACTIONS = 5


class QNetwork(nn.Module):
    def __init__(self, obs_dim=OBS_DIM, n_actions=N_ACTIONS, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, n_actions),
        )

    def forward(self, x):
        return self.net(x)


class ReplayBuffer:
    def __init__(self, capacity=50000):
        self.buffer = deque(maxlen=capacity)

    def push(self, obs, action_idx, reward, next_obs, done):
        self.buffer.append((obs, action_idx, reward, next_obs, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        obs, action_idx, reward, next_obs, done = zip(*batch)
        return (np.array(obs, dtype=np.float32), np.array(action_idx, dtype=np.int64),
                np.array(reward, dtype=np.float32), np.array(next_obs, dtype=np.float32),
                np.array(done, dtype=np.float32))

    def __len__(self):
        return len(self.buffer)


class DQNAgent:
    def __init__(self, obs_dim=OBS_DIM, n_actions=N_ACTIONS, lr=1e-3, gamma=0.95,
                 buffer_size=50000, device="cpu"):
        self.device = device
        self.q_net = QNetwork(obs_dim, n_actions).to(device)
        self.target_net = QNetwork(obs_dim, n_actions).to(device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.optimizer = torch.optim.Adam(self.q_net.parameters(), lr=lr)
        self.buffer = ReplayBuffer(buffer_size)
        self.gamma = gamma
        self.n_actions = n_actions

    def select_action(self, obs, epsilon):
        if random.random() < epsilon:
            return random.randrange(self.n_actions)
        with torch.no_grad():
            obs_t = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
            q_values = self.q_net(obs_t)
            return int(q_values.argmax(dim=1).item())

    def store(self, obs, action_idx, reward, next_obs, done):
        self.buffer.push(obs, action_idx, reward, next_obs, done)

    def train_step(self, batch_size=64):
        if len(self.buffer) < batch_size:
            return None
        obs, action_idx, reward, next_obs, done = self.buffer.sample(batch_size)
        obs_t = torch.tensor(obs, device=self.device)
        action_t = torch.tensor(action_idx, device=self.device)
        reward_t = torch.tensor(reward, device=self.device)
        next_obs_t = torch.tensor(next_obs, device=self.device)
        done_t = torch.tensor(done, device=self.device)

        q_values = self.q_net(obs_t).gather(1, action_t.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            next_q = self.target_net(next_obs_t).max(dim=1)[0]
            target = reward_t + self.gamma * next_q * (1 - done_t)
        loss = F.smooth_l1_loss(q_values, target)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_net.parameters(), max_norm=10.0)
        self.optimizer.step()
        return loss.item()

    def update_target(self):
        self.target_net.load_state_dict(self.q_net.state_dict())