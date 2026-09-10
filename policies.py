"""
policies.py

Baseline (non-RL) irrigation policies, used for comparison against the
trained multi-agent RL system in Phase 5. Each is a function:

    policy_fn(env, obs) -> {agent_name: action_index}

where action_index selects from env.action_levels = [0, 2.5, 5, 7.5, 10] liters.
"""

from irrigation_env import get_stage


def fixed_schedule_policy(env, obs):
    """
    Traditional fixed-schedule irrigation: apply the same amount every day
    to every zone, regardless of actual soil moisture, weather, or growth
    stage. This is what most real farms WITHOUT smart sensors still do --
    it's the "Traditional irrigation" baseline from your slides.
    """
    FIXED_LEVEL_INDEX = 2  # 5.0 liters, every zone, every day
    return {a: FIXED_LEVEL_INDEX for a in env.agents}


def threshold_policy(env, obs):
    """
    Simple sensor-based rule: irrigate a zone only when its moisture drops
    below the LOWER bound of the current growth stage's optimal range.
    This represents a "smart-ish but non-learning" system -- the kind of
    threshold-based controller real off-the-shelf smart irrigation kits use.
    It does NOT look at weather (e.g. upcoming rain) or the shared water
    budget the way your RL agents will -- that gap is the whole point of
    the project.
    """
    _, _, lo, hi, _ = get_stage(env.day)
    actions = {}
    for a in env.agents:
        moisture = env.moisture[a]
        actions[a] = 3 if moisture < lo else 0  # 7.5L if below threshold, else none
    return actions


def random_policy(env, obs):
    """Pure random actions -- a sanity-check lower bound, not a real baseline."""
    return {a: env.action_spaces[a].sample() for a in env.agents}