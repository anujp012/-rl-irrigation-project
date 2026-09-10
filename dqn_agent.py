"""
dqn_agent.py

ONE shared Q-network used by every zone (parameter sharing across agents --
a standard, documented multi-agent RL technique). Each zone queries the same
network with its own local observation and gets its own action back; every
zone's experience feeds the same replay buffer, so with 4 zones you
effectively get 4x the training data per simulated day for one shared policy.

Action space is discretized into liter "tiers" -- this keeps the first
working version to plain DQN (simple to implement, debug, and explain in a
viva). Continuous actions via PPO/SAC/DDPG are a natural stretch goal once
this is working and you have time left.
"""
import random
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ACTION_TIERS = [0.0, 2.5, 5.0, 7.5, 10.0]  # must match irrigation_env.py's action_levels exactly
OBS_DIM = 8   # must match irrigation_env.py's _get_obs() output length
N_ACTIONS = len(ACTION_TIERS)


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
        return (np.array(obs, dtype=np.float32),