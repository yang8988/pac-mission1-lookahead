"""Dense grid-search oracle to measure 5-1 candidate quality.

For a (box, state) every pose on an ``step``-metre grid (both yaws, drop
height) is checked with the SAME hard mask. Compared with the 5-1 candidate
set this answers:

* recall_any   : if any feasible pose exists, does 5-1 offer >= 1 valid one?
* best_top_gap : (lowest feasible top z from 5-1) - (lowest from the grid)
* best_support : support ratio of the best (lowest-top) candidate vs grid
"""

from dataclasses import dataclass
import math

from pac_common import Pose3D

from pac_candidates.candidate_generation import candidate_yaws
from pac_candidates.geometry import Rect, rotated_dims
from pac_candidates.pallet_model import box_tolerance


@dataclass(frozen=True)
class OracleResult:
    grid_feasible: int
    grid_evaluated: int
    best_grid_top: float | None
    best_grid_key: tuple | None


def _key(z_top, x, y):
    return (round(z_top, 4), round(y, 4), round(x, 4))


def grid_search(backend, box, state, step=0.02):
    model = backend.model_for(state)
    tol = box_tolerance(backend.config.uncertainty, backend._uncertain(box.box_id))
    margin = model.half_gap + tol
    p = state.pallet.size
    feasible = 0
    evaluated = 0
    best = None
    for yaw in candidate_yaws(box, backend.config):
        dx, dy, dz = rotated_dims(box.size, yaw)
        nx = int(math.floor((p.x - dx - 2 * tol) / step + 1e-9))
        ny = int(math.floor((p.y - dy - 2 * tol) / step + 1e-9))
        xs = [tol + i * step for i in range(nx + 1)] + [p.x - dx - tol]
        ys = [tol + j * step for j in range(ny + 1)] + [p.y - dy - tol]
        for x in xs:
            for y in ys:
                expanded = Rect(x - margin, y - margin, x + dx + margin, y + dy + margin)
                z = model.resting_z(expanded)
                outcome = backend.evaluate_pose(box, Pose3D("pallet", x, y, z, yaw=yaw), state)
                evaluated += 1
                if outcome.codes:
                    continue
                feasible += 1
                key = _key(z + dz, x, y)
                if best is None or key < best:
                    best = key
    return OracleResult(feasible, evaluated, best[0] if best else None, best)


def compare(backend, box, state, candidate_set, step=0.02):
    oracle = grid_search(backend, box, state, step)
    best_cand = None
    for cand in candidate_set.valid:
        dz = rotated_dims(box.size, cand.target_pose.yaw)[2]
        key = _key(cand.target_pose.z + dz, cand.target_pose.x, cand.target_pose.y)
        if best_cand is None or key < best_cand:
            best_cand = key
    return {
        "grid_feasible": oracle.grid_feasible,
        "grid_evaluated": oracle.grid_evaluated,
        "candidate_valid": candidate_set.valid_count,
        "candidate_generated": candidate_set.generated_count,
        "missed_all": oracle.grid_feasible > 0 and candidate_set.valid_count == 0,
        "candidate_only": oracle.grid_feasible == 0 and candidate_set.valid_count > 0,
        "best_grid_top": oracle.best_grid_top,
        "best_candidate_top": best_cand[0] if best_cand else None,
        "best_top_gap": (
            best_cand[0] - oracle.best_grid_top
            if best_cand is not None and oracle.best_grid_top is not None
            else None
        ),
    }
