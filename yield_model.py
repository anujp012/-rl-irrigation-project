"""
yield_model.py

Post-hoc crop yield estimation from a season's moisture trace. Applied
AFTER a trace already exists (baseline or trained DQN) -- does NOT change
the RL reward function and does NOT require retraining anything.

Loosely modeled on the FAO "yield response factor" (Ky) idea: yield loss
is proportional to water-stress severity, weighted by how sensitive the
crop's CURRENT growth stage is to that stress.
"""
KY_BY_STAGE = {
    "seedling": 0.35,
    "vegetative": 0.45,
    "flowering": 1.35,   
    "fruiting": 1.05,
    "maturity": 0.25,
}

def daily_stress_fraction(moisture, lo, hi, wilt):
    """0.0 = no stress today, 1.0 = maximum stress today."""
    if lo <= moisture <= hi:
        return 0.0
    if moisture < lo:
        if moisture <= wilt:
            return 1.0
        return (lo - moisture) / (lo - wilt)  # ramps 0 -> 1 as moisture falls from lo to wilt
    excess_room = max(100 - hi, 1e-6)
    return min((moisture - hi) / excess_room, 1.0) * 0.5

def estimate_season_yield(day_records, get_stage_fn):
    """
    day_records: list of {"day": int, "moisture": float} for ONE zone,
                 pulled from a trace_*.json's "days" list.
    get_stage_fn: pass in irrigation_env.py's get_stage function.
    """
    n_days = len(day_records)
    if n_days == 0:
        return {"relative_yield_pct": 0.0, "estimated_kg": 0.0, "estimated_value": 0.0, "daily_stress": []}

    total_loss = 0.0
    daily_stress = []
    for rec in day_records:
        _, stage_name, lo, hi, wilt = get_stage_fn(rec["day"])
        stress = daily_stress_fraction(rec["moisture"], lo, hi, wilt)
        ky = KY_BY_STAGE.get(stage_name, 0.5)
        total_loss += ky * stress / n_days
        daily_stress.append(round(stress, 3))

    relative_yield = max(0.0, 1.0 - min(total_loss, 1.0))
    return {
        "relative_yield_pct": round(relative_yield * 100, 1),
        "estimated_kg": round(relative_yield * POTENTIAL_YIELD_KG_PER_ZONE, 1),
        "estimated_value": round(relative_yield * POTENTIAL_YIELD_KG_PER_ZONE * PRICE_PER_KG, 0),
        "daily_stress": daily_stress,
    }
POTENTIAL_YIELD_KG_PER_ZONE = 300.0   # illustrative, per zone per season, at 100% relative yield
PRICE_PER_KG = 18.0                   # illustrative market priceif __name__ == "__main__":

if __name__ == "__main__":
    def fake_get_stage(day):
        return (0, "flowering", 55, 75, 20)  # pretend every day is flowering, for a clean test

    print("--- Sanity check 1: a perfect season (always 65% moisture, always healthy) ---")
    perfect = [{"day": d, "moisture": 65} for d in range(100)]
    print(estimate_season_yield(perfect, fake_get_stage))

    print("\n--- Sanity check 2: a disastrous season (always 10%, always below wilt) ---")
    disaster = [{"day": d, "moisture": 10} for d in range(100)]
    print(estimate_season_yield(disaster, fake_get_stage))

    print("\n--- Sanity check 3: half healthy, half wilted ---")
    mixed = [{"day": d, "moisture": 65 if d < 50 else 10} for d in range(100)]
    print(estimate_season_yield(mixed, fake_get_stage))