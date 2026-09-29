"""
yield_comparison.py

Averages relative crop yield across the SAME 20 real-weather seeds used in
compare_policies.py -- because one season (like the seed=42 result you just
saw) could be a lucky or unlucky window, not a general pattern.

Usage:
    python yield_comparison.py
"""
import os
import numpy as np
import torch

from irrigation_env import MultiZoneIrrigationEnv, get_stage
from dqn_agent import DQNAgent, N_ACTIONS, OBS_DIM
from policies import random_policy, threshold_policy, fixed_schedule_policy
from yield_model import estimate_season_yield

CHECKPOINT_PATH = "checkpoints/dqn_independent_final.pt"
N_TRIALS = 20


def load_agents():
    if not os.path.exists(CHECKPOINT_PATH):
        return None
    state_dicts = torch.load(CHECKPOINT_PATH, map_location="cpu")
    agents = {}
    for zone, state_dict in state_dicts.items():
        agent = DQNAgent(obs_dim=OBS_DIM, n_actions=N_ACTIONS)
        agent.q_net.load_state_dict(state_dict)
        agents[zone] = agent
    return agents


def run_one_trial(policy_fn, seed, agents=None):
    env = MultiZoneIrrigationEnv(n_zones=4, season_length=100, daily_water_budget=25.0,
                                  weather_source="real")
    obs, infos = env.reset(seed=seed)
    zone_day_records = {a: [] for a in env.possible_agents}
    day = 0
    while env.agents:
        if agents is not None:
            actions = {a: agents[a].select_action(obs[a], epsilon=0.0) for a in env.agents}
        else:
            actions = policy_fn(env, obs)
        obs, rewards, term, trunc, infos = env.step(actions)
        for a in infos:
            zone_day_records[a].append({"day": day, "moisture": infos[a]["moisture"]})
        day += 1

    zone_yields = [estimate_season_yield(zone_day_records[z], get_stage)["relative_yield_pct"]
                   for z in zone_day_records]
    return float(np.mean(zone_yields))


def main():
    agents = load_agents()
    seeds = range(2000, 2000 + N_TRIALS)
    policies = {"Random": random_policy, "Fixed Schedule": fixed_schedule_policy,
                "Threshold": threshold_policy}
    if agents is not None:
        policies["Trained DQN"] = None
    else:
        print("(no checkpoints/dqn_independent_final.pt yet -- skipping Trained DQN row until training finishes)\n")

    print(f"Average relative yield across {N_TRIALS} real weather windows:\n")
    print(f"{'Policy':<18}{'Avg yield %':<14}{'Std dev'}")
    for name, fn in policies.items():
        yields = [run_one_trial(fn, s, agents=(agents if name == "Trained DQN" else None)) for s in seeds]
        print(f"{name:<18}{np.mean(yields):<14.1f}{np.std(yields):.1f}")


if __name__ == "__main__":
    main()