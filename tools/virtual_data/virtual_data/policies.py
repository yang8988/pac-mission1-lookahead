"""Placement policies used to roll scenarios forward.

They only decide which *valid* (hard-mask-passing) candidate is executed so
that the virtual data covers diverse intermediate pallets. They are data
collection policies, not the production planner (stages 5-3~5-6).

* dblf   : deepest-bottom-left-fill = valid candidate with the lowest
           (z, y, x); independent of the 5-1 output order
* random : uniform among valid candidates
* mixed  : dblf, but with probability epsilon a random top-N valid one
* planner: donghan's PlacementPlanner (5-3~5-6) with this backend, if present
"""


def dblf_key(candidate):
    p = candidate.target_pose
    return (round(p.z, 6), round(p.y, 6), round(p.x, 6), candidate.candidate_id)


def choose(policy, candidate_set, rng, cfg, planner_call=None):
    valid = sorted(candidate_set.valid, key=dblf_key)
    if not valid:
        return None, "NO_VALID_CANDIDATE"
    if policy == "dblf":
        return valid[0], "dblf"
    if policy == "random":
        return rng.choice(valid), "random"
    if policy == "mixed":
        if rng.random() < cfg.epsilon:
            return rng.choice(valid[: cfg.top_n_random]), "mixed_random"
        return valid[0], "mixed_dblf"
    if policy == "planner":
        if planner_call is None:
            raise RuntimeError("planner policy needs pac_planning (run fetch_team_deps.sh)")
        chosen = planner_call(valid)
        if chosen is None:
            return valid[0], "planner_fallback_dblf"
        return chosen, "planner"
    raise ValueError(f"Unknown policy {policy}")
