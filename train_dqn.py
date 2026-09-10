"""
train_dqn.py

The training loop that was missing: ties MultiZoneIrrigationEnv +
DQNAgent together and actually runs episodes.

Multi-agent detail: all n_zones share ONE DQNAgent (parameter sharing, per
dqn_agent.py's docstring). So each simulated day produces n_zones separate
(obs, action, reward, next_obs, done) transitions -- all pushed into the
SAME replay buffer, not n_zones separate buffers.

Usage:
    python train_dqn.py --episodes 500 --weather synthetic
    python train_dqn.py --episodes 500 --weather real
"""

import argparse
import json
import os

import numpy as np
import torch

from dqn_agent import DQNAgent, N_ACTIONS, OBS_DIM
from irrigation_env import MultiZoneIrrigationEnv


def train(episodes=500, weather_source="synthetic", n_zones=4, season_length=100,
          daily_water_budget=25.0, target_update_every=10, batch_size=64,
          eps_start=1.0, eps_end=0.05, eps_decay_episodes=400, seed=0,
          checkpoint_dir="checkpoints", log_path="logs/train_log.json"):

    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(os.path.dirname(log_path), exist_ok=True)

    env = MultiZoneIrrigationEnv(n_zones=n_zones, season_length=season_length,
                                  daily_water_budget=daily_water_budget,
                                  weather_source=weather_source, seed=seed)
    agent = DQNAgent(obs_dim=OBS_DIM, n_actions=N_ACTIONS)

    history = {"episode": [], "total_reward": [], "avg_loss": [], "epsilon": []}
    rng = np.random.default_rng(seed)

    for ep in range(episodes):
        epsilon = max(eps_end, eps_start - (eps_start - eps_end) * ep / eps_decay_episodes)

        ep_seed = int(rng.integers(0, 1_000_000))  # different weather window every episode
        obs, infos = env.reset(seed=ep_seed)
        ep_reward = 0.0
        losses = []

        while env.agents:
            current_agents = list(env.agents)  # snapshot: env.agents may empty out inside step()
            actions = {a: agent.select_action(obs[a], epsilon) for a in current_agents}
            next_obs, rewards, term, trunc, infos = env.step(actions)

            for a in current_agents:
                agent.store(obs[a], actions[a], rewards[a], next_obs[a], trunc[a])
                ep_reward += rewards[a]

            loss = agent.train_step(batch_size=batch_size)
            if loss is not None:
                losses.append(loss)

            obs = next_obs

        if ep % target_update_every == 0:
            agent.update_target()

        avg_loss = float(np.mean(losses)) if losses else None
        history["episode"].append(ep)
        history["total_reward"].append(ep_reward)
        history["avg_loss"].append(avg_loss)
        history["epsilon"].append(epsilon)

        if ep % 10 == 0 or ep == episodes - 1:
            loss_str = "n/a" if avg_loss is None else f"{avg_loss:.3f}"
            print(f"ep {ep:4d}  reward={ep_reward:9.1f}  avg_loss={loss_str}  epsilon={epsilon:.3f}")

    ckpt_path = os.path.join(checkpoint_dir, "dqn_final.pt")
    torch.save(agent.q_net.state_dict(), ckpt_path)
    with open(log_path, "w") as f:
        json.dump(history, f, indent=2)

    print(f"\nSaved checkpoint to {ckpt_path}")
    print(f"Saved training log to {log_path}")
    return agent, history


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=500)
    parser.add_argument("--weather", choices=["synthetic", "real"], default="synthetic")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    train(episodes=args.episodes, weather_source=args.weather, seed=args.seed)