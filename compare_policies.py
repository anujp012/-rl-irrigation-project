"""
compare_policies.py

Runs random, fixed_schedule, threshold, and the trained independent
multi-agent DQN (4 separate networks) through the SAME set of real weather
windows and compares total seasonal reward.

Usage:
    python compare_policies.py
"""
import numpy as np
import torch

from irrigation_env import MultiZoneIrrigationEnv
from dqn_agent import DQNAgent, N_ACTIONS, OBS_DIM
from policies import random_policy, threshold_policy, fixed_schedule_policy

CHECKPOINT_PATH = "checkpoints/dqn_independent_final.pt"
N_TRIALS = 20


def load_agents():
    """Loads the 4 independent agents from one checkpoint file that stores
    {zone_name: state_dict}."""
    state_dicts = torch.load(CHECKPOINT_PATH, map_location="cpu")
    agents = {}
    for zone, state_dict in state_dicts.items():
        agent = DQNAgent(obs_dim=OBS_DIM, n_actions=N_ACTIONS)
        agent.q_net.load_state_dict(state_dict)
        agents[zone] = agent
    return agents


def run_policy(policy_fn, seed, agents=None):
    env = MultiZoneIrrigationEnv(n_zones=4, season_length=100, daily_water_budget=25.0,
                                  weather_source="real")
    obs, infos = env.reset(seed=seed)
    total = 0.0
    while env.agents:
        if agents is not None:
            actions = {a: agents[a].select_action(obs[a], epsilon=0.0) for a in env.agents}
        else:
            actions = policy_fn(env, obs)
        obs, rewards, term, trunc, infos = env.step(actions)
        total += sum(rewards.values())
    return total


def main():
    agents = load_agents()

    seeds = range(2000, 2000 + N_TRIALS)
    results = {"random": [], "fixed_schedule": [], "threshold": [], "trained_dqn": []}

    for s in seeds:
        results["random"].append(run_policy(random_policy, s))
        results["fixed_schedule"].append(run_policy(fixed_schedule_policy, s))
        results["threshold"].append(run_policy(threshold_policy, s))
        results["trained_dqn"].append(run_policy(None, s, agents=agents))

    print(f"Compared across {N_TRIALS} identical real weather windows:\n")
    print(f"{'Policy':<18}{'Mean reward':<15}{'Median':<12}{'Worst case'}")
    for name, vals in results.items():
        vals = np.array(vals)
        print(f"{name:<18}{vals.mean():<15.1f}{np.median(vals):<12.1f}{vals.min():.1f}")


if __name__ == "__main__":
    main()