"""
health_comparison.py

Presentable version of compare_policies.py -- % of days each zone stayed
in its healthy moisture range, and water used to get there. Uses the
independent multi-agent DQN (4 separate networks).

Usage:
    python health_comparison.py
"""
import numpy as np
import torch

from irrigation_env import MultiZoneIrrigationEnv, get_stage
from dqn_agent import DQNAgent, N_ACTIONS, OBS_DIM
from policies import random_policy, threshold_policy, fixed_schedule_policy

CHECKPOINT_PATH = "checkpoints/dqn_independent_final.pt"
N_TRIALS = 20


def load_agents():
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
    healthy_days, total_days, total_water = 0, 0, 0.0
    while env.agents:
        if agents is not None:
            actions = {a: agents[a].select_action(obs[a], epsilon=0.0) for a in env.agents}
        else:
            actions = policy_fn(env, obs)
        obs, rewards, term, trunc, infos = env.step(actions)
        _, _, lo, hi, _ = get_stage(env.day - 1)
        for a in infos:
            total_days += 1
            total_water += infos[a]["applied_liters"]
            if lo <= infos[a]["moisture"] <= hi:
                healthy_days += 1
    return healthy_days / total_days * 100, total_water


def main():
    agents = load_agents()

    seeds = range(2000, 2000 + N_TRIALS)
    policies = {"Random": random_policy, "Fixed Schedule": fixed_schedule_policy,
                "Threshold": threshold_policy, "Trained DQN (independent)": None}

    print(f"Compared across {N_TRIALS} identical real weather windows:\n")
    print(f"{'Policy':<28}{'% days healthy':<18}{'Avg water used/season'}")
    for name, fn in policies.items():
        healthy_pcts, waters = [], []
        for s in seeds:
            h, w = run_policy(fn, s, agents=(agents if name.startswith("Trained") else None))
            healthy_pcts.append(h)
            waters.append(w)
        print(f"{name:<28}{np.mean(healthy_pcts):<18.1f}{np.mean(waters):.0f} L")


if __name__ == "__main__":
    main()