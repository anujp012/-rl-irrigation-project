"""
diagnose_seed42.py

Runs threshold, fixed, and the trained independent DQN all on ONE chosen
weather seed -- prints day-by-day moisture, water applied, and reward for
the DQN's last 30 days, so we can see exactly what it does as things get
bad, not just the final crash.

Usage:
    python diagnose_seed42.py --seed 42
    python diagnose_seed42.py --seed 30
    python diagnose_seed42.py --seed 35
"""
import argparse
import torch
import numpy as np

from irrigation_env import MultiZoneIrrigationEnv, get_stage
from dqn_agent import DQNAgent, N_ACTIONS, OBS_DIM
from policies import threshold_policy, fixed_schedule_policy
from yield_model import estimate_season_yield

CHECKPOINT_PATH = "checkpoints/dqn_independent_final.pt"
SEED = 42  # placeholder -- overwritten by --seed when run directly


def load_agents():
    state_dicts = torch.load(CHECKPOINT_PATH, map_location="cpu")
    agents = {}
    for zone, state_dict in state_dicts.items():
        agent = DQNAgent(obs_dim=OBS_DIM, n_actions=N_ACTIONS)
        agent.q_net.load_state_dict(state_dict)
        agents[zone] = agent
    return agents


def run_and_summarize(label, policy_fn=None, agents=None, verbose_zone=None):
    env = MultiZoneIrrigationEnv(n_zones=4, season_length=100, daily_water_budget=25.0,
                                  weather_source="real")
    obs, infos = env.reset(seed=SEED)
    zone_day_records = {a: [] for a in env.possible_agents}
    day = 0
    print(f"\n=== {label} ===")
    while env.agents:
        if agents is not None:
            actions = {a: agents[a].select_action(obs[a], epsilon=0.0) for a in env.agents}
        else:
            actions = policy_fn(env, obs)
        obs, rewards, term, trunc, infos = env.step(actions)
        for a in infos:
            zone_day_records[a].append({"day": day, "moisture": infos[a]["moisture"]})
        if verbose_zone and verbose_zone in infos and day >= 70:
            z = infos[verbose_zone]
            print(f"day {day:3d}  moisture={z['moisture']:6.2f}  applied={z['applied_liters']:5.2f}L  "
                  f"reward={rewards[verbose_zone]:7.2f}  water_cost={z['water_cost']:.2f}")
        day += 1

    zone_yields = [estimate_season_yield(zone_day_records[z], get_stage)["relative_yield_pct"]
                   for z in zone_day_records]
    print(f"{label}: avg yield = {np.mean(zone_yields):.1f}%  "
          f"(per zone: {[round(y, 1) for y in zone_yields]})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    SEED = args.seed

    agents = load_agents()
    run_and_summarize("THRESHOLD", policy_fn=threshold_policy)
    run_and_summarize("FIXED", policy_fn=fixed_schedule_policy)
    run_and_summarize("TRAINED DQN (independent)", agents=agents, verbose_zone="zone_0")