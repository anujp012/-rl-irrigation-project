"""
weather_sim.py

Two interchangeable weather sources, both exposing the SAME get_day(t) ->
dict interface, so swapping one for the other requires no changes anywhere
else in the project:

- WeatherSimulator: fully synthetic (sinusoidal seasonal curve + noise).
- RealWeatherSimulator: backed by real daily India weather data (2000-2023),
  columns: Country, Date, Temp_Max, Temp_Min, Temp_Mean, Precipitation_Sum,
  Windspeed_Max, Windgusts_Max, Sunshine_Duration.

The real dataset doesn't include humidity or a rain-probability forecast, so
those two fields are approximated (documented inline). Everything else
(temperature, actual rainfall) is the real recorded value for that day.
"""

from pathlib import Path

import numpy as np
import pandas as pd

DEFAULT_CSV_PATH = Path(__file__).parent / "data" / "India_weather_data.csv"


class WeatherSimulator:
    def __init__(self, season_length=100, seed=None):
        self.season_length = season_length
        self.rng = np.random.default_rng(seed)
        self._generate()

    def _generate(self):
        days = np.arange(self.season_length)

        self.temperature = (
            26 + 8 * np.sin(2 * np.pi * days / self.season_length)
            + self.rng.normal(0, 1.5, self.season_length)
        )

        self.humidity = (
            55 + 15 * np.cos(2 * np.pi * days / self.season_length)
            + self.rng.normal(0, 5, self.season_length)
        )
        self.humidity = np.clip(self.humidity, 20, 95)

        base_rain_prob = 0.25 + 0.2 * np.sin(2 * np.pi * (days + 20) / self.season_length)
        self.rain_probability = np.clip(
            base_rain_prob + self.rng.normal(0, 0.05, self.season_length), 0.02, 0.95
        )
        self.actual_rainfall = np.array([
            self.rng.exponential(8) if self.rng.random() < p else 0.0
            for p in self.rain_probability
        ])

        self.water_cost = np.ones(self.season_length)
        n_shortage_days = max(1, int(0.15 * self.season_length))
        shortage_days = self.rng.choice(self.season_length, size=n_shortage_days, replace=False)
        self.water_cost[shortage_days] = self.rng.uniform(2.5, 5.0, n_shortage_days)

    def get_day(self, t):
        t = min(t, self.season_length - 1)
        return {
            "temperature": float(self.temperature[t]),
            "humidity": float(self.humidity[t]),
            "rain_probability": float(self.rain_probability[t]),
            "actual_rainfall": float(self.actual_rainfall[t]),
            "water_cost": float(self.water_cost[t]),
        }


class RealWeatherSimulator:
    """
    Drop-in replacement for WeatherSimulator, backed by real historical data.

    Each construction with a given seed samples a random `season_length`-day
    WINDOW out of the full 24-year record, instead of always replaying the
    same fixed season -- so different training episodes see genuinely
    different real historical stretches (monsoon onset, dry spells, winter
    starts...), the same way the synthetic version varied its noise every
    reset.

    Field-by-field honesty about what's real vs. approximated:
    - temperature: REAL (Temp_Mean for that exact date)
    - actual_rainfall: REAL (Precipitation_Sum for that exact date)
    - rain_probability: APPROXIMATED -- there's no forecast column, so this
      uses the CLIMATOLOGICAL probability of rain for that time of year: the
      fraction of years in the dataset that had any rain within +/-3 days of
      that calendar date. E.g. "historically, ~70% of mid-July days see rain"
      becomes the rain_probability an agent observes on a mid-July day.
    - humidity: APPROXIMATED -- no humidity column exists, so it's derived
      from the REAL temperature and rainfall for these exact days (cooler +
      recently-rainy => higher humidity), not an independently invented curve.
    - water_cost: SYNTHETIC, same model as WeatherSimulator -- this is an
      economic/infrastructure signal, not something a weather dataset would
      ever contain.

    Limitation worth stating in your report: this is a NATIONAL daily
    average, not a specific farm's microclimate -- fine for demonstrating
    realistic seasonal patterns, but a genuine simplification if asked.
    """

    _full_df = None  # class-level cache: parse the CSV once, not once per episode

    def __init__(self, season_length=100, seed=None, csv_path=None):
        self.season_length = season_length
        self.rng = np.random.default_rng(seed)

        if RealWeatherSimulator._full_df is None:
            path = csv_path or DEFAULT_CSV_PATH
            df = pd.read_csv(path)
            df["Date"] = pd.to_datetime(df["Date"], format="%d-%m-%Y")
            df = df.sort_values("Date").reset_index(drop=True)
            df["doy"] = df["Date"].dt.dayofyear
            RealWeatherSimulator._full_df = df

        full_df = RealWeatherSimulator._full_df
        rain_prob_by_doy = self._climatological_rain_probability(full_df)

        max_start = len(full_df) - season_length
        start_idx = int(self.rng.integers(0, max_start)) if max_start > 0 else 0
        window = full_df.iloc[start_idx:start_idx + season_length].reset_index(drop=True)

        self.temperature = window["Temp_Mean"].to_numpy(dtype=float)
        self.actual_rainfall = window["Precipitation_Sum"].to_numpy(dtype=float)
        self.rain_probability = np.array([rain_prob_by_doy[d] for d in window["doy"]])

        # Humidity proxy: cooler and recently-rainy days run more humid.
        # Not measured -- derived from the real temp/rain for these days.
        temp_component = 90 - 1.3 * self.temperature
        rain_component = 15 * np.clip(self.actual_rainfall, 0, 20) / 20
        self.humidity = np.clip(temp_component + rain_component, 20, 95)

        self.water_cost = np.ones(season_length)
        n_shortage_days = max(1, int(0.15 * season_length))
        shortage_days = self.rng.choice(season_length, size=n_shortage_days, replace=False)
        self.water_cost[shortage_days] = self.rng.uniform(2.5, 5.0, n_shortage_days)

    @staticmethod
    def _climatological_rain_probability(full_df, window_days=3):
        doy_arr = full_df["doy"].to_numpy()
        rained_arr = (full_df["Precipitation_Sum"].to_numpy() > 0).astype(float)
        prob_by_doy = {}
        for doy in range(1, 367):
            diff = np.abs(doy_arr - doy)
            circ_diff = np.minimum(diff, 366 - diff)  # wraps around Dec 31 <-> Jan 1
            mask = circ_diff <= window_days
            prob_by_doy[doy] = float(rained_arr[mask].mean()) if mask.any() else 0.3
        return prob_by_doy

    def get_day(self, t):
        t = min(t, self.season_length - 1)
        return {
            "temperature": float(self.temperature[t]),
            "humidity": float(self.humidity[t]),
            "rain_probability": float(self.rain_probability[t]),
            "actual_rainfall": float(self.actual_rainfall[t]),
            "water_cost": float(self.water_cost[t]),
        }


if __name__ == "__main__":
    print("--- Synthetic ---")
    w = WeatherSimulator(season_length=5, seed=1)
    for t in range(5):
        print(t, w.get_day(t))

    print("\n--- Real (India data) ---")
    rw = RealWeatherSimulator(season_length=5, seed=1)
    for t in range(5):
        print(t, rw.get_day(t))
