"""
paired_comparison.py

Per-seed paired comparison instead of raw averages -- since Threshold and
Trained DQN were tested on the SAME 20 weather windows, this checks how
often DQN actually wins seed-by-seed, which is a much more reliable signal
than comparing two noisy averages.

Usage:
    python paired_comparison.py
"""
import numpy as np
import torch

from irrigation_env import MultiZoneIrrigationEnv, get_stage
from dqn_agent import DQNAgent, N_ACTIONS, OBS_DIM
from policies import threshold_policy, fixed_schedule_policy
from yield_model import estimate_season_yield

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
    seeds = list(range(2000, 2000 + N_TRIALS))

    print(f"{'Seed':<8}{'Fixed':<10}{'Threshold':<12}{'Trained DQN':<14}{'DQN beats Threshold?'}")
    dqn_wins = 0
    diffs = []
    for s in seeds:
        y_fixed = run_one_trial(fixed_schedule_policy, s)
        y_threshold = run_one_trial(threshold_policy, s)
        y_dqn = run_one_trial(None, s, agents=agents)
        won = y_dqn > y_threshold
        dqn_wins += int(won)
        diffs.append(y_dqn - y_threshold)
        print(f"{s:<8}{y_fixed:<10.1f}{y_threshold:<12.1f}{y_dqn:<14.1f}{'YES' if won else 'no'}")

    print(f"\nTrained DQN beat Threshold on {dqn_wins}/{N_TRIALS} weather windows")
    print(f"Average paired difference (DQN - Threshold): {np.mean(diffs):+.1f} points "
          f"(std of the DIFFERENCE itself: {np.std(diffs):.1f})")


if __name__ == "__main__":
    main()