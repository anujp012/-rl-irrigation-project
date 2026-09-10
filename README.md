# Smart Multi-Zone Irrigation — RL Project

Everything here is the FINAL, verified set of files. If you had earlier
downloads, delete them and use only what's in this folder.

## The RL spec (what to write in your report)

| | |
|---|---|
| **Agent** | One decision-maker per irrigation zone. All zones share one policy network (parameter sharing). |
| **State** (per zone) | soil moisture, day-of-season fraction, crop stage, temperature, windspeed, rain probability, remaining shared water budget, cumulative water used |
| **Action** (per zone) | liters to apply today, chosen from tiers [0, 10, 20, 30, 40, 50] |
| **Reward** (per zone) | penalizes moisture outside a healthy 60–80% band, minus a cost per liter used |
| **Goal** | maximize total reward across every zone over a season — keep every zone healthy without exceeding the field's shared daily water supply |
| **What makes it hard** | all zones draw from ONE shared daily water budget, so agents must implicitly learn to hold back so other zones can get water — real coordination, not N copies of one problem |

## File-by-file

```
multi_zone_env.py     the environment/simulator — soil, weather, water-sharing logic
real_weather.py        reads real historical weather from your Kaggle CSV
dqn_agent.py            the actual RL agent (shared Q-network + replay buffer)
train_real_data.py      RUN THIS to train the agent on real weather
requirements.txt        pip install -r requirements.txt

optional_testing/       NOT part of the final pipeline. Only use if you want to
                         re-verify the environment mechanics with fake weather.
    stub_weather.py
    test_multi_zone_env.py
```

## Setup (do this once)

1. Copy every file above (except `optional_testing/`, unless you want it) into
   your `RL_project` folder.
2. Create a `data` folder inside `RL_project` if you don't have one, and put
   your `India_weather_data.csv` inside it:
   `RL_project/data/India_weather_data.csv`
3. Open a terminal in `RL_project` and run:
   ```
   pip install -r requirements.txt
   ```

## Run it

```
cd RL_project
python train_real_data.py
```

This trains for 300 episodes (~1–2 minutes on a normal laptop) and prints
progress every 20 episodes. You should see the reward number climb from
strongly negative to positive — that's the agent learning. At the end it
saves:
- `trained_dqn_real.pt` — the trained model weights
- `training_rewards_real.json` — reward per episode, for your report's
  learning-curve chart

## What's still left to do

1. **A threshold baseline** (e.g. "irrigate 30L if moisture < 60%, else 0")
   run through the same environment, so you can show the trained agent
   beating a simple rule in your report.
2. **Merge into your existing dashboard** — if you have `generate_trace.py`
   and a web dashboard already, adapt them to read from this environment
   instead of your old single-agent one.
3. **Write-up** — the table at the top of this README is your MDP
   definition section, almost verbatim.