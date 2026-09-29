"""
main.py -- FastAPI backend for the irrigation project.

WHAT THIS DOES, PLAINLY:
Instead of you manually running generate_trace.py and uploading the JSON
file by hand, this server runs your actual environment + policies + trained
agent LIVE, whenever the frontend asks for it, and hands back the result as
JSON. The frontend below then displays it. No manual file steps.

ENDPOINTS:
  GET /                          -> serves the Live Monitor dashboard page
  GET /analytics                 -> serves the Training & Evidence page
  GET /api/policies              -> list of policies you can pick from
  GET /api/trace?policy=X&seed=N -> runs policy X for one season, returns
                                     the full day-by-day trace (same JSON
                                     shape your generate_trace.py produces)
  GET /api/comparison?trials=N   -> runs ALL policies across N different
                                     real weather windows and returns
                                     health%/water/reward for each --
                                     this is the same thing health_comparison.py
                                     prints, but as JSON for the webpage
  GET /api/comparison-report     -> serves the SAVED file data/comparison_report.json
                                     (written by generate_comparison_report.py).
                                     Instant, because nothing is re-run.
  GET /api/training-log          -> serves the SAVED file logs/train_log.json
                                     (written by train_dqn.py)

Run with:
    uvicorn main:app --reload
Then open http://127.0.0.1:8000 in your browser.
"""
import json
import os
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, JSONResponse

from irrigation_env import MultiZoneIrrigationEnv, get_stage
from policies import random_policy, threshold_policy, fixed_schedule_policy
from dqn_agent import DQNAgent, N_ACTIONS, OBS_DIM

app = FastAPI(title="Smart Irrigation RL API")

# Folders that hold saved files. Path(__file__).parent means "the folder this
# main.py lives in", so these work no matter where you launch uvicorn from.
DATA_DIR = Path(__file__).parent / "data"
LOGS_DIR = Path(__file__).parent / "logs"

# Independent-agent checkpoint: a DICT of {zone_name: state_dict}, one
# separate trained network per zone -- NOT the old single shared network.
CHECKPOINT_PATH = "checkpoints/dqn_independent_final.pt"
N_ZONES = 4
SEASON_LENGTH = 100
DAILY_WATER_BUDGET = 25.0

POLICY_FUNCTIONS = {
    "random": random_policy,
    "fixed": fixed_schedule_policy,
    "threshold": threshold_policy,
    # "trained_dqn" is handled specially below since it needs the loaded model,
    # not a plain function -- see run_episode()
}

# Load the trained agents once at startup, not on every request. This is now
# a DICT of {zone_name: DQNAgent}, one independent agent per zone -- not one
# shared agent reused across zones.
_trained_agents: Optional[dict] = None


def get_trained_agents() -> dict:
    global _trained_agents
    if _trained_agents is None:
        if not os.path.exists(CHECKPOINT_PATH):
            raise HTTPException(
                status_code=404,
                detail=f"No trained model found at {CHECKPOINT_PATH}. Run "
                       f"'python train_dqn.py --episodes 500 --weather real' first.",
            )
        state_dicts = torch.load(CHECKPOINT_PATH, map_location="cpu")
        agents = {}
        for zone, state_dict in state_dicts.items():
            agent = DQNAgent(obs_dim=OBS_DIM, n_actions=N_ACTIONS)
            agent.q_net.load_state_dict(state_dict)
            agents[zone] = agent
        _trained_agents = agents
    return _trained_agents


def run_episode(policy_name: str, seed: int, collect_trace: bool = False):
    """
    Runs ONE season with the given policy. This is the same logic as
    generate_trace.py and compare_policies.py -- centralized here so both
    the trace endpoint and the comparison endpoint use identical code
    (avoids the two ever silently drifting apart).
    """
    env = MultiZoneIrrigationEnv(n_zones=N_ZONES, season_length=SEASON_LENGTH,
                                  daily_water_budget=DAILY_WATER_BUDGET, weather_source="real")
    obs, infos = env.reset(seed=seed)

    agent = get_trained_agents() if policy_name == "trained_dqn" else None
    policy_fn = POLICY_FUNCTIONS.get(policy_name)
    if agent is None and policy_fn is None:
        raise HTTPException(status_code=400, detail=f"Unknown policy '{policy_name}'")

    trace_days = []
    totals = {"water": 0.0, "reward": 0.0, "healthy_days": 0, "total_zone_days": 0}
    day = 0

    while env.agents:
        if agent is not None:
            actions = {a: agent[a].select_action(obs[a], epsilon=0.0) for a in env.agents}
        else:
            actions = policy_fn(env, obs)

        obs, rewards, term, trunc, infos = env.step(actions)
        _, _, lo, hi, _ = get_stage(env.day - 1)

        if collect_trace:
            w = env.weather.get_day(min(day, env.season_length - 1))
            trace_days.append({
                "day": day,
                "weather": w,
                "zones": {
                    a: {
                        "moisture": round(infos[a]["moisture"], 2),
                        "stage": infos[a]["stage"],
                        "applied_liters": round(infos[a]["applied_liters"], 2),
                        "reward": round(rewards[a], 2),
                    } for a in infos
                },
            })

        for a in infos:
            totals["water"] += infos[a]["applied_liters"]
            totals["reward"] += rewards[a]
            totals["total_zone_days"] += 1
            if lo <= infos[a]["moisture"] <= hi:
                totals["healthy_days"] += 1
        day += 1

    return trace_days, totals


@app.get("/api/policies")
def list_policies():
    return {"policies": ["random", "fixed", "threshold", "trained_dqn"]}


@app.get("/api/trace")
def get_trace(policy: str = Query(..., description="random | fixed | threshold | trained_dqn"),
              seed: int = Query(42, description="which real weather window to use")):
    trace_days, totals = run_episode(policy, seed, collect_trace=True)
    return JSONResponse({
        "meta": {
            "policy": policy,
            "n_zones": N_ZONES,
            "season_length": SEASON_LENGTH,
            "daily_water_budget": DAILY_WATER_BUDGET,
            "weather_source": "real",
        },
        "days": trace_days,
    })


@app.get("/api/comparison")
def get_comparison(trials: int = Query(20, ge=1, le=100, description="number of different real weather windows to test on")):
    policy_names = ["random", "fixed", "threshold", "trained_dqn"]
    seeds = range(2000, 2000 + trials)
    results = {}

    for name in policy_names:
        rewards, healthy_pcts, waters = [], [], []
        for s in seeds:
            _, totals = run_episode(name, s, collect_trace=False)
            rewards.append(totals["reward"])
            healthy_pcts.append(totals["healthy_days"] / totals["total_zone_days"] * 100)
            waters.append(totals["water"])
        results[name] = {
            "mean_reward": round(float(np.mean(rewards)), 1),
            "median_reward": round(float(np.median(rewards)), 1),
            "worst_reward": round(float(np.min(rewards)), 1),
            "pct_days_healthy": round(float(np.mean(healthy_pcts)), 1),
            "avg_water_liters": round(float(np.mean(waters)), 0),
        }

    return JSONResponse({"trials": trials, "results": results})


# ---------------------------------------------------------------------------
# Endpoints that serve SAVED files instead of re-running simulations.
# Same pattern for both: check the file exists, 404 with a helpful message
# if not, otherwise load the JSON and return it.
# ---------------------------------------------------------------------------

@app.get("/api/comparison-report")
def get_comparison_report():
    file_path = DATA_DIR / "comparison_report.json"
    if not file_path.exists():
        raise HTTPException(
            status_code=404,
            detail="No comparison report yet. Run: python generate_comparison_report.py",
        )
    with open(file_path) as f:
        return json.load(f)


@app.get("/api/training-log")
def get_training_log():
    file_path = LOGS_DIR / "train_log.json"
    if not file_path.exists():
        raise HTTPException(
            status_code=404,
            detail="No training log found. Run: python train_dqn.py --episodes ... --weather real",
        )
    with open(file_path) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Page routes -- keep these last so they don't shadow any /api/* route.
# ---------------------------------------------------------------------------

@app.get("/")
def serve_dashboard():
    return FileResponse("dashboard.html")


@app.get("/analytics")
def serve_analytics():
    return FileResponse("analytics.html")