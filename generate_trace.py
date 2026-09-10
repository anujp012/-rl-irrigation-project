"""
generate_trace.py

Runs the multi-zone environment for one full season using a chosen policy,
saves the day-by-day trace as JSON (the website's data contract), and prints
summary metrics -- total water used, stress-days, total reward -- so you can
directly compare policies. These printed numbers ARE the evidence for your
comparison table/slides.

Usage:
    python generate_trace.py fixed
    python generate_trace.py threshold
    python generate_trace.py random
"""

import sys
import json
import os
from irrigation_env import MultiZoneIrrigationEnv
from policies import fixed_schedule_policy, threshold_policy, random_policy

POLICIES = {
    "fixed": fixed_schedule_policy,
    "threshold": threshold_policy,
    "random": random_policy,
}


def run_policy(env, policy_fn, policy_name):
    obs, infos = env.reset(seed=42)
    trace = {
        "meta": {
            "policy": policy_name,
            "n_zones": env.n_zones,
            "season_length": env.season_length,
            "daily_water_budget": env.daily_water_budget,
        },
        "days": [],
    }
    totals = {"water": 0.0, "stress_days": 0, "reward": 0.0}
    day = 0
    while env.agents:
        actions = policy_fn(env, obs)
        obs, rewards, term, trunc, infos = env.step(actions)
        w = env.weather.get_day(min(day, env.season_length - 1))
        trace["days"].append({
            "day": day,
            "weather": w,
            "zones": {
                a: {
                    "moisture": round(infos[a]["moisture"], 2),
                    "stage": infos[a]["stage"],
                    "applied_liters": round(infos[a]["applied_liters"], 2),
                    "reward": round(rewards[a], 2),
                }
                for a in infos
            },
        })
        for a in infos:
            totals["water"] += infos[a]["applied_liters"]
            totals["reward"] += rewards[a]
            if infos[a]["moisture"] < 20:
                totals["stress_days"] += 1
        day += 1
    return trace, totals


if __name__ == "__main__":
    policy_name = sys.argv[1] if len(sys.argv) > 1 else "random"
    if policy_name not in POLICIES:
        print(f"Unknown policy '{policy_name}'. Choose from: {list(POLICIES)}")
        sys.exit(1)
    policy_fn = POLICIES[policy_name]

    env = MultiZoneIrrigationEnv(n_zones=4, season_length=100, daily_water_budget=25.0, seed=42)
    trace, totals = run_policy(env, policy_fn, policy_name)

    os.makedirs("data", exist_ok=True)
    out_path = f"data/trace_{policy_name}.json"
    with open(out_path, "w") as f:
        json.dump(trace, f, indent=2)

    print(f"=== {policy_name.upper()} policy | {env.season_length}-day season | {env.n_zones} zones ===")
    print(f"Total water used:    {totals['water']:.1f} L")
    print(f"Total stress-days:   {totals['stress_days']}  (moisture below wilting point)")
    print(f"Total reward:        {totals['reward']:.1f}  (higher/less negative = better)")
    print(f"Saved trace to {out_path}")