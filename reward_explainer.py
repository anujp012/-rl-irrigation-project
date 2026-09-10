"""
reward_explainer.py

Demonstrates, with real numbers, HOW the reward function judges an
irrigation decision -- and, critically, what happens when a decision that
LOOKED reasonable (e.g. trusting an 85% rain forecast) turns out wrong on a
given day. This script is meant to be run and shown directly: it is your
evidence for defending the reward function design.

Reward math is imported from irrigation_env.compute_reward -- the SAME
function irrigation_env.py's step() calls -- so this script can never
silently drift out of sync with the real reward again.

PART 1 shows what the reward would have been for every possible action,
given one fixed day's conditions -- proving the reward correctly rewards
better decisions and punishes worse ones in that instance.

PART 2 (Monte Carlo) proves the deeper point: a single wrong outcome does
NOT mean the strategy was bad. It runs the SAME decision (skip vs water)
across thousands of simulated days where it truly does rain 85% of the
time, and shows that "trust the forecast" wins on average, even though it
loses badly on the unlucky days. This is exactly what PPO/DQN training does
across many episodes -- learn from the statistics of many trials, not from
any single labeled "correct answer" (which RL never has).

Usage:
    python reward_explainer.py
"""

import numpy as np
from irrigation_env import MultiZoneIrrigationEnv, get_stage, compute_reward

ACTION_LEVELS = [0.0, 2.5, 5.0, 7.5, 10.0]


def evaluate_all_actions(label, day, starting_moisture, temperature, humidity,
                          rain_probability, actual_rainfall, water_cost):
    stage_idx, stage_name, lo, hi, wilt = get_stage(day)
    evap = MultiZoneIrrigationEnv._evapotranspiration(temperature, humidity, stage_idx)

    print(f"--- {label} ---")
    print(f"Day {day} ({stage_name} stage, optimal moisture {lo}-{hi}%, wilting point {wilt}%)")
    print(f"Moisture now: {starting_moisture}%  Temp: {temperature}C  Humidity: {humidity}%  "
          f"Rain prob: {rain_probability:.0%}  Actual rainfall today: {actual_rainfall}mm")
    print(f"{'Water applied (L)':<20}{'-> New moisture':<18}{'Reward'}")

    best_action, best_reward = None, float("-inf")
    rows = []
    for level in ACTION_LEVELS:
        new_m = max(0.0, min(100.0, starting_moisture - evap + level + actual_rainfall))
        reward = compute_reward(new_m, lo, hi, wilt, level, water_cost)
        rows.append((level, new_m, reward))
        if reward > best_reward:
            best_reward, best_action = reward, level

    for level, new_m, reward in rows:
        marker = "  <-- BEST" if level == best_action else ""
        print(f"{level:<20.1f}{new_m:<18.1f}{reward:.2f}{marker}")
    print()


def monte_carlo_rain_bet(rain_probability=0.85, trials=2000, day=45, starting_moisture=58,
                          temperature=28, humidity=70, water_cost=1.0, seed=0):
    rng = np.random.default_rng(seed)
    stage_idx, stage_name, lo, hi, wilt = get_stage(day)
    evap = MultiZoneIrrigationEnv._evapotranspiration(temperature, humidity, stage_idx)

    skip_rewards, water_rewards = [], []
    for _ in range(trials):
        rained = rng.random() < rain_probability
        actual_rainfall = rng.exponential(10) if rained else 0.0
        for level, bucket in [(0.0, skip_rewards), (5.0, water_rewards)]:
            new_m = max(0.0, min(100.0, starting_moisture - evap + level + actual_rainfall))
            bucket.append(compute_reward(new_m, lo, hi, wilt, level, water_cost))

    print(f"--- PART 2: Monte Carlo -- {trials} simulated days, rain probability = {rain_probability:.0%} ---")
    print(f"Strategy 'always SKIP irrigation':  average reward = {np.mean(skip_rewards):.2f}  "
          f"(worst day: {np.min(skip_rewards):.2f})")
    print(f"Strategy 'always WATER 5L':          average reward = {np.mean(water_rewards):.2f}  "
          f"(worst day: {np.min(water_rewards):.2f})")
    print("-> Skipping loses badly on unlucky no-rain days (see 'worst day'), but wins")
    print("   ON AVERAGE because the forecast is right most of the time. This is what")
    print("   the RL agent actually learns from -- the statistics across many training")
    print("   episodes, not a single 'correct answer' for any one day.\n")


if __name__ == "__main__":
    evaluate_all_actions(
        "SCENARIO 1: Hot & dry, rain unlikely -> agent SHOULD irrigate",
        day=45, starting_moisture=58, temperature=35, humidity=28,
        rain_probability=0.05, actual_rainfall=0.0, water_cost=1.0,
    )
    evaluate_all_actions(
        "SCENARIO 2: Rain arrived today -> agent SHOULD have skipped",
        day=45, starting_moisture=58, temperature=28, humidity=70,
        rain_probability=0.85, actual_rainfall=12.0, water_cost=1.0,
    )
    evaluate_all_actions(
        "SCENARIO 3: Soil already near-saturated -> skip regardless",
        day=45, starting_moisture=72, temperature=30, humidity=60,
        rain_probability=0.3, actual_rainfall=0.0, water_cost=1.0,
    )
    monte_carlo_rain_bet()