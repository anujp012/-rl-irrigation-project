"""
test_random_policy.py

Runs the multi-zone environment for one full season with RANDOM actions.
This is just a sanity check that the simulation runs cleanly and produces
plausible numbers -- it is NOT the trained agent. Once IPPO agents are added
(phase 2), this same loop structure gets reused with real policies.
"""

from irrigation_env import MultiZoneIrrigationEnv

env = MultiZoneIrrigationEnv(n_zones=4, season_length=100, daily_water_budget=25.0, seed=42)
obs, infos = env.reset(seed=42)

total_water = {a: 0.0 for a in env.possible_agents}
stress_days = {a: 0 for a in env.possible_agents}
ep_reward = {a: 0.0 for a in env.possible_agents}

day = 0
while env.agents:
    actions = {a: env.action_spaces[a].sample() for a in env.agents}
    obs, rewards, term, trunc, infos = env.step(actions)
    for a in infos:
        total_water[a] += infos[a]["applied_liters"]
        ep_reward[a] += rewards[a]
        if infos[a]["moisture"] < 20:
            stress_days[a] += 1
    day += 1

print(f"=== Random policy sanity check ({day}-day season, 4 zones) ===")
for a in env.possible_agents:
    print(f"{a}: total_water={total_water[a]:6.1f}L   stress_days={stress_days[a]:3d}   "
          f"episode_reward={ep_reward[a]:8.1f}")
print(f"\nFarm total water used: {sum(total_water.values()):.1f}L "
      f"(budget was {env.daily_water_budget}L/day x {day} days = {env.daily_water_budget*day:.0f}L)")