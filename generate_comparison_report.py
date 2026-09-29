"""
generate_comparison_report.py

Runs the full policy comparison ONCE -- reward, yield, the paired
significance test, and the specific threshold-vs-fixed failure-mode seeds
-- and saves everything to one JSON file for the frontend's Policy
Comparison page. Precomputed on purpose: this runs ~100 full 100-day
seasons, too slow to recompute live on every page load.

Also compares your OLD shared-parameter DQN checkpoint against the NEW
independent-agent one, if both exist.

Usage:
    python generate_comparison_report.py
"""
import json
import os
from math import comb

import numpy as np
import torch

from irrigation_env import MultiZoneIrrigationEnv, get_stage
from dqn_agent import DQNAgent, N_ACTIONS, OBS_DIM
from policies import random_policy, threshold_policy, fixed_schedule_policy
from yield_model import estimate_season_yield

INDEPENDENT_CHECKPOINT = "checkpoints/dqn_independent_final.pt"
SHARED_CHECKPOINT = "checkpoints/dqn_final.pt"
N_TRIALS = 20


def load_independent_agents():
    state_dicts = torch.load(INDEPENDENT_CHECKPOINT, map_location="cpu")
    agents = {}
    for zone, state_dict in state_dicts.items():
        agent = DQNAgent(obs_dim=OBS_DIM, n_actions=N_ACTIONS)
        agent.q_net.load_state_dict(state_dict)
        agents[zone] = agent
    return agents


def load_shared_agent():
    agent = DQNAgent(obs_dim=OBS_DIM, n_actions=N_ACTIONS)
    agent.q_net.load_state_dict(torch.load(SHARED_CHECKPOINT, map_location="cpu"))
    return agent


def run_one_trial(seed, policy_fn=None, independent_agents=None, shared_agent=None):
    """Runs one season under exactly one of: a baseline policy_fn, a dict of
    independent per-zone agents, or one shared agent used by every zone.
    Returns (total_reward, avg_zone_yield_pct)."""
    env = MultiZoneIrrigationEnv(n_zones=4, season_length=100, daily_water_budget=25.0,
                                  weather_source="real")
    obs, infos = env.reset(seed=seed)
    zone_day_records = {a: [] for a in env.possible_agents}
    total_reward = 0.0
    day = 0
    while env.agents:
        if independent_agents is not None:
            actions = {a: independent_agents[a].select_action(obs[a], epsilon=0.0) for a in env.agents}
        elif shared_agent is not None:
            actions = {a: shared_agent.select_action(obs[a], epsilon=0.0) for a in env.agents}
        else:
            actions = policy_fn(env, obs)
        obs, rewards, term, trunc, infos = env.step(actions)
        for a in infos:
            zone_day_records[a].append({"day": day, "moisture": infos[a]["moisture"]})
            total_reward += rewards[a]
        day += 1
    zone_yields = [estimate_season_yield(zone_day_records[z], get_stage)["relative_yield_pct"]
                   for z in zone_day_records]
    return total_reward, float(np.mean(zone_yields))


def sign_test_two_tailed(wins, n):
    p_obs = sum(comb(n, k) * 0.5 ** n for k in range(wins, n + 1))
    return min(1.0, 2 * p_obs)


def paired_stats(a, b, n):
    """a, b: arrays of the SAME length, paired by seed. Returns win/loss
    count for a over b, mean/std of the difference, sign test p, t stat."""
    diff = np.array(a) - np.array(b)
    wins = int((diff > 0).sum())
    mean_diff = float(diff.mean())
    std_diff = float(diff.std(ddof=1)) if n > 1 else 0.0
    sign_p = sign_test_two_tailed(wins, n)
    t_stat = mean_diff / (std_diff / np.sqrt(n)) if std_diff > 0 else float("nan")
    return {"wins": wins, "losses": n - wins, "mean_diff": round(mean_diff, 2),
            "std_diff": round(std_diff, 2), "sign_test_p": round(sign_p, 5),
            "t_statistic": round(t_stat, 3) if std_diff > 0 else None}


def main():
    seeds = list(range(2000, 2000 + N_TRIALS))
    baseline_policies = {"random": random_policy, "fixed_schedule": fixed_schedule_policy,
                          "threshold": threshold_policy}

    have_independent = os.path.exists(INDEPENDENT_CHECKPOINT)
    have_shared = os.path.exists(SHARED_CHECKPOINT)
    independent_agents = load_independent_agents() if have_independent else None
    shared_agent = load_shared_agent() if have_shared else None

    if not have_independent:
        print("NOTE: no independent-agent checkpoint found -- skipping trained_dqn_independent")
    if not have_shared:
        print("NOTE: no shared-parameter checkpoint found -- skipping trained_dqn_shared (ablation)")

    raw = {name: {"reward": [], "yield": []} for name in baseline_policies}
    if have_independent:
        raw["trained_dqn_independent"] = {"reward": [], "yield": []}
    if have_shared:
        raw["trained_dqn_shared"] = {"reward": [], "yield": []}

    for s in seeds:
        for name, fn in baseline_policies.items():
            r, y = run_one_trial(s, policy_fn=fn)
            raw[name]["reward"].append(r)
            raw[name]["yield"].append(y)
        if have_independent:
            r, y = run_one_trial(s, independent_agents=independent_agents)
            raw["trained_dqn_independent"]["reward"].append(r)
            raw["trained_dqn_independent"]["yield"].append(y)
        if have_shared:
            r, y = run_one_trial(s, shared_agent=shared_agent)
            raw["trained_dqn_shared"]["reward"].append(r)
            raw["trained_dqn_shared"]["yield"].append(y)

    summary = {}
    for name, vals in raw.items():
        rewards, yields = np.array(vals["reward"]), np.array(vals["yield"])
        summary[name] = {
            "reward_mean": round(float(rewards.mean()), 1),
            "reward_median": round(float(np.median(rewards)), 1),
            "reward_worst": round(float(rewards.min()), 1),
            "yield_mean": round(float(yields.mean()), 1),
            "yield_std": round(float(yields.std()), 1),
        }

    report = {"n_trials": N_TRIALS, "seeds": seeds, "summary": summary, "per_seed": raw}

    if have_independent:
        report["paired_dqn_vs_threshold"] = paired_stats(
            raw["trained_dqn_independent"]["yield"], raw["threshold"]["yield"], N_TRIALS)

        failure_seeds = []
        for i, s in enumerate(seeds):
            if raw["threshold"]["yield"][i] < raw["fixed_schedule"]["yield"][i]:
                dqn_y = raw["trained_dqn_independent"]["yield"][i]
                failure_seeds.append({
                    "seed": s,
                    "fixed": round(raw["fixed_schedule"]["yield"][i], 1),
                    "threshold": round(raw["threshold"]["yield"][i], 1),
                    "trained_dqn": round(dqn_y, 1),
                    "dqn_beats_both": bool(dqn_y > raw["fixed_schedule"]["yield"][i]
                                            and dqn_y > raw["threshold"]["yield"][i]),
                })
        report["threshold_vs_fixed_failure_seeds"] = failure_seeds

    if have_independent and have_shared:
        report["paired_independent_vs_shared"] = paired_stats(
            raw["trained_dqn_independent"]["yield"], raw["trained_dqn_shared"]["yield"], N_TRIALS)

    os.makedirs("data", exist_ok=True)
    with open("data/comparison_report.json", "w") as f:
        json.dump(report, f, indent=2)

    print("\nSaved data/comparison_report.json\n")
    print(json.dumps(summary, indent=2))
    if "paired_independent_vs_shared" in report:
        print("\nIndependent vs shared architecture (yield):")
        print(json.dumps(report["paired_independent_vs_shared"], indent=2))


if __name__ == "__main__":
    main()