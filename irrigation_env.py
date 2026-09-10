"""
irrigation_env.py

Multi-zone, multi-agent irrigation environment (PettingZoo ParallelEnv API).

- Each zone is a separate agent with its own local observation and reward.
- All agents share ONE farm-wide weather feed and ONE daily water budget --
  if the sum of everyone's requested water exceeds today's budget, requests
  are scaled down fairly and each agent is penalized in proportion to how
  much of its request got clipped. That penalty signal is what teaches
  agents restraint when other zones also need water.
- weather_source picks between synthetic (WeatherSimulator) and real
  historical India data (RealWeatherSimulator) -- both expose the identical
  get_day(t) interface, so nothing else in this file changes either way.

State per agent (8 values): own soil moisture, own growth-stage index, today's
temperature, humidity, rain probability, water cost, yesterday's shared-budget
compliance fraction, and the mean moisture of the OTHER zones.

Action per agent: discrete irrigation amount for its own zone today
(0, 2.5, 5, 7.5, or 10 liters).
"""

import numpy as np
from gymnasium import spaces
from pettingzoo import ParallelEnv

GROWTH_STAGES = [
    ("seedling",   15, 40, 55, 15),
    ("vegetative", 40, 45, 65, 15),
    ("flowering",  60, 55, 75, 20),
    ("fruiting",   85, 50, 70, 18),
    ("maturity",  100, 30, 50, 15),
]


def get_stage(day):
    for idx, (name, end_day, lo, hi, wilt) in enumerate(GROWTH_STAGES):
        if day < end_day:
            return idx, name, lo, hi, wilt
    name, end_day, lo, hi, wilt = GROWTH_STAGES[-1]
    return len(GROWTH_STAGES) - 1, name, lo, hi, wilt


def compute_reward(new_moisture, lo, hi, wilt, applied_liters, water_cost, overuse_liters=0.0):
    """Single source of truth for the per-zone reward -- step() and
    reward_explainer.py both call this, so they can't drift apart again."""
    if new_moisture < lo:
        moisture_penalty, health_bonus = lo - new_moisture, 0.0
    elif new_moisture > hi:
        moisture_penalty, health_bonus = (new_moisture - hi) * 0.5, 0.0
    else:
        moisture_penalty, health_bonus = 0.0, 3.0

    stress_penalty = 15.0 if new_moisture < wilt else 0.0
    water_penalty = applied_liters * water_cost * 0.3
    overuse_penalty = overuse_liters * 1.0
    conservation_bonus = 0.1 * (10.0 - applied_liters) if health_bonus > 0 else 0.0

    return health_bonus + conservation_bonus - moisture_penalty - stress_penalty - water_penalty - overuse_penalty


class MultiZoneIrrigationEnv(ParallelEnv):
    metadata = {"name": "multi_zone_irrigation_v0"}

    def __init__(self, n_zones=4, season_length=100, daily_water_budget=25.0,
                 seed=None, weather_source="synthetic"):
        self.n_zones = n_zones
        self.season_length = season_length
        self.daily_water_budget = daily_water_budget
        self.weather_source = weather_source  # "synthetic" or "real"
        self.possible_agents = [f"zone_{i}" for i in range(n_zones)]
        self.agents = list(self.possible_agents)
        self._seed = seed

        self.action_levels = np.array([0.0, 2.5, 5.0, 7.5, 10.0])
        self.action_spaces = {a: spaces.Discrete(len(self.action_levels)) for a in self.possible_agents}

        low = np.array([0, 0, 0, 0, 0, 0, 0, 0], dtype=np.float32)
        high = np.array([100, len(GROWTH_STAGES) - 1, 50, 100, 1, 5, 1, 100], dtype=np.float32)
        self.observation_spaces = {
            a: spaces.Box(low=low, high=high, dtype=np.float32) for a in self.possible_agents
        }

        self.weather = None
        self.moisture = {}
        self.day = 0
        self.budget_compliance = 1.0

    def reset(self, seed=None, options=None):
        use_seed = seed if seed is not None else self._seed
        self.agents = list(self.possible_agents)
        self.day = 0

        if self.weather_source == "real":
            from weather_sim import RealWeatherSimulator
            self.weather = RealWeatherSimulator(self.season_length, seed=use_seed)
        else:
            from weather_sim import WeatherSimulator
            self.weather = WeatherSimulator(self.season_length, seed=use_seed)

        rng = np.random.default_rng(use_seed)
        self.moisture = {a: float(rng.uniform(35, 55)) for a in self.agents}
        self.budget_compliance = 1.0
        obs = {a: self._get_obs(a) for a in self.agents}
        infos = {a: {} for a in self.agents}
        return obs, infos

    def _get_obs(self, agent):
        w = self.weather.get_day(self.day)
        idx, _, _, _, _ = get_stage(self.day)
        others = [self.moisture[o] for o in self.agents if o != agent]
        mean_other = float(np.mean(others)) if others else 0.0
        return np.array([
            self.moisture[agent], idx, w["temperature"], w["humidity"],
            w["rain_probability"], w["water_cost"], self.budget_compliance, mean_other,
        ], dtype=np.float32)

    @staticmethod
    def _evapotranspiration(temp, humidity, stage_idx):
        stage_factor = [0.6, 1.0, 1.3, 1.1, 0.7][stage_idx]
        base = 0.15 * (temp - 10) * (1 - humidity / 150)
        return max(0.5, base * stage_factor * 3.0)

    def step(self, actions):
        w = self.weather.get_day(self.day)
        stage_idx, stage_name, lo, hi, wilt = get_stage(self.day)

        requested = {a: self.action_levels[actions[a]] for a in self.agents}
        total_requested = sum(requested.values())

        scale = self.daily_water_budget / total_requested if total_requested > self.daily_water_budget else 1.0
        applied = {a: requested[a] * scale for a in self.agents}
        overuse = {a: requested[a] - applied[a] for a in self.agents}

        total_overuse = sum(overuse.values())
        self.budget_compliance = 1.0 - min(total_overuse / self.daily_water_budget, 1.0)

        rewards, infos = {}, {}
        for a in self.agents:
            evap = self._evapotranspiration(w["temperature"], w["humidity"], stage_idx)
            new_m = float(np.clip(self.moisture[a] - evap + applied[a] + w["actual_rainfall"], 0, 100))
            self.moisture[a] = new_m

            rewards[a] = compute_reward(new_m, lo, hi, wilt, applied[a], w["water_cost"], overuse[a])
            infos[a] = {
                "moisture": new_m, "stage": stage_name, "applied_liters": applied[a],
                "overuse_liters": overuse[a], "water_cost": w["water_cost"],
            }

        self.day += 1
        terminations = {a: False for a in self.agents}
        truncations = {a: self.day >= self.season_length for a in self.agents}
        obs = {a: self._get_obs(a) for a in self.agents}

        if self.day >= self.season_length:
            self.agents = []

        return obs, rewards, terminations, truncations, infos