"""
apply_yield.py

Loads an existing trace_*.json (produced by generate_trace.py) and runs
each zone's real moisture history through yield_model.py -- this is the
first time the yield model touches actual simulated data instead of
hand-made test cases.

Usage:
    python apply_yield.py data/trace_fixed.json
    python apply_yield.py data/trace_threshold.json
"""
import sys
import json

from irrigation_env import get_stage
from yield_model import estimate_season_yield


def main():
    trace_path = sys.argv[1] if len(sys.argv) > 1 else "data/trace_fixed.json"
    with open(trace_path) as f:
        trace = json.load(f)

    zone_names = list(trace["days"][0]["zones"].keys())
    print(f"=== Yield estimate for {trace['meta']['policy'].upper()} policy ({trace_path}) ===\n")

    all_yields = []
    for zone in zone_names:
        day_records = [{"day": d["day"], "moisture": d["zones"][zone]["moisture"]} for d in trace["days"]]
        result = estimate_season_yield(day_records, get_stage)
        all_yields.append(result["relative_yield_pct"])
        print(f"{zone}: {result['relative_yield_pct']}% relative yield  "
              f"({result['estimated_kg']}kg, value {result['estimated_value']})")

    avg_yield = sum(all_yields) / len(all_yields)
    print(f"\nFarm average: {avg_yield:.1f}% relative yield across {len(zone_names)} zones")


if __name__ == "__main__":
    main()