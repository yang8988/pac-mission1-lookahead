"""Carton strength scenarios: the TRUE top-load capacity of every box.

Box specifications are not published, so the virtual data covers many
cases instead of one guess. Each scenario draws a *strength profile*; each
SKU gets a carton grade and every box a condition factor:

    BCT (McKee)  = 5.87 * ECT * sqrt(t * perimeter)
    true limit   = BCT * environment factor * box condition factor

The planner (5-2) never sees these values. It keeps its own *assumed*
capacity (``config/taehyeon/candidates.yaml`` load_model). Comparing both
measures the real crush risk of an assumption without any measurement.

Mission requirement "dented/damaged boxes": a damaged box has a weak top. If
the stage-2 inspection detects it (``detect_probability``), the planner gets
``capacity_overrides_n[box_id] = 0`` (it may be placed, nothing on top).
Undetected damage stays hidden from the planner.

All grade / environment numbers are generic development assumptions for
corrugated board, not measured values of the competition boxes.
"""

from dataclasses import dataclass
import math

from pac_candidates.loads import mckee_capacity_n
from pac_candidates.pallet_model import G

# grade -> (ECT range N/m, board thickness m). Generic corrugated board.
CARTON_GRADES = {
    "single_E": ((2500.0, 3500.0), 0.0015),
    "single_B": ((3500.0, 5000.0), 0.0030),
    "single_C": ((4500.0, 6000.0), 0.0040),
    "double_BC": ((7000.0, 10000.0), 0.0070),
}

# profile -> grade weights, environment factor range (humidity / storage
# time / stacking duration), damage probability.
STRENGTH_PROFILES = {
    "strong": ({"single_C": 0.4, "double_BC": 0.6}, (0.6, 0.8), 0.0),
    "nominal": ({"single_B": 0.6, "single_C": 0.4}, (0.5, 0.7), 0.02),
    "weak": ({"single_E": 0.5, "single_B": 0.5}, (0.4, 0.6), 0.04),
    "humid": ({"single_B": 0.5, "single_C": 0.5}, (0.3, 0.45), 0.03),
    "mixed": (
        {"single_E": 0.25, "single_B": 0.3, "single_C": 0.25, "double_BC": 0.2},
        (0.35, 0.8),
        0.03,
    ),
    # Stress test: thin board, soaked/long-stored, frequent damage.
    "extreme": ({"single_E": 0.7, "single_B": 0.3}, (0.2, 0.3), 0.10),
}
PROFILE_ORDER = tuple(STRENGTH_PROFILES)


@dataclass(frozen=True)
class StrengthScenario:
    profile: str
    sku_grade: dict  # sku_id -> grade
    true_capacity_n: dict  # box_id -> true top-load limit (N)
    damaged: tuple  # box ids with a damaged top
    detected: tuple  # damaged AND detected by inspection


def _weighted(rng, weights):
    names = sorted(weights)
    total = sum(weights[n] for n in names)
    value = rng.random() * total
    for name in names:
        value -= weights[name]
        if value <= 0:
            return name
    return names[-1]


def bct_n(dx, dy, ect, thickness):
    """McKee box compression strength (N): the same formula as the planner's
    assumed capacity, without its safety factor."""
    return mckee_capacity_n(dx, dy, ect, thickness, 1.0)


def draw(arrivals, profile, rng, damage_factor=0.25, detect_probability=0.8):
    """Draw the true strength of every arriving box for one scenario."""
    grades, env_range, p_damage = STRENGTH_PROFILES[profile]
    env = rng.uniform(*env_range)
    sku_grade = {}
    sku_ect = {}
    capacities = {}
    damaged = []
    detected = []
    for box in arrivals:
        if box.sku_id not in sku_grade:
            grade = _weighted(rng, grades)
            sku_grade[box.sku_id] = grade
            (lo, hi), _ = CARTON_GRADES[grade]
            sku_ect[box.sku_id] = rng.uniform(lo, hi)
        grade = sku_grade[box.sku_id]
        thickness = CARTON_GRADES[grade][1]
        # per-box manufacturing scatter +-10 %
        scatter = rng.uniform(0.9, 1.1)
        capacity = bct_n(box.size.x, box.size.y, sku_ect[box.sku_id], thickness) * env * scatter
        if rng.random() < p_damage:
            capacity *= damage_factor
            damaged.append(box.box_id)
            if rng.random() < detect_probability:
                detected.append(box.box_id)
        capacities[box.box_id] = capacity
    return StrengthScenario(profile, sku_grade, capacities, tuple(damaged), tuple(detected))


def true_overloads(final_state, scenario, cand_config):
    """Boxes whose TRUE limit is exceeded by the load actually stacked on them.

    Loads are computed with the same top-down model as 5-2 (lever shares).
    Returns (overloaded box ids, max true load ratio).
    """
    from pac_common import PlanningContext

    from pac_candidates.pallet_model import PalletModel

    capacities = {
        b.box_id: scenario.true_capacity_n.get(b.box_id, math.inf)
        for b in final_state.pallet.boxes
    }
    finite = {k: v for k, v in capacities.items() if math.isfinite(v)}
    context = PlanningContext(catalog={}, pallet_max_weight_kg=1e9, capacity_overrides_n=finite)
    model = PalletModel(final_state, context, cand_config)
    over = []
    worst = 0.0
    for g in model.boxes:
        load = model.top_load_n[g.box_id]
        cap = capacities.get(g.box_id, math.inf)
        ratio = load / cap if cap > 0 else (0.0 if load <= 0 else math.inf)
        worst = max(worst, ratio)
        if ratio > 1.0:
            over.append(g.box_id)
    return over, worst


def footprint_capacity_summary(scenario, arrivals):
    """Min / median true capacity in kg (for reports)."""
    values = sorted(scenario.true_capacity_n[b.box_id] / G for b in arrivals)
    if not values:
        return {}
    return {
        "min_kg": round(values[0], 2),
        "median_kg": round(values[len(values) // 2], 2),
        "max_kg": round(values[-1], 2),
    }


__all__ = [
    "CARTON_GRADES",
    "PROFILE_ORDER",
    "STRENGTH_PROFILES",
    "StrengthScenario",
    "draw",
    "true_overloads",
]
