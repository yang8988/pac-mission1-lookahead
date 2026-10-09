"""Stage 5-2: Hard Mask -- checks that are NEVER skipped (time budget or not).

Checks (team ``RejectCode`` in brackets, detailed reason strings in details):

1. orientation: upright, axis-aligned, allowed yaw      [INVALID_STATE]
2. pallet boundary incl. size tolerance                  [OUT_OF_BOUND]
3. max stack height incl. size tolerance                 [HEIGHT_LIMIT]
4. box overlap / lateral clearance delta                 [BOX_COLLISION]
5. top-down descent column free                          [APPROACH_FAIL]
6. resting contact (no floating placement)               [LOW_SUPPORT]
7. minimum support area ratio                            [LOW_SUPPORT]
8. LBCP stability with CoG uncertainty delta             [COG_VIOLATION]
9. carton top-load capacity of every box below          [LOAD_VIOLATION]
10. heavy-on-light (mission requirement)                 [LOAD_VIOLATION]
11. pallet max load                                      [LOAD_VIOLATION]
12. pallet CoG allowed region                            [COG_VIOLATION]

Checks run from cheap to expensive. Runtime (``collect_all=False``) stops at
the first failing group -- the verdict (valid / codes) is identical, only the
list of secondary reasons is shorter. ``collect_all=True`` (virtual data
labels) evaluates every check so the rejected list carries every reason.
"""

from dataclasses import dataclass
import math

from pac_common import ConstraintEvidence, RejectCode as R

from .geometry import (
    LEN_EPS,
    STABILITY_EPS,
    footprint,
    inside_margin,
    quarter_turns,
    rotated_dims,
    same_yaw,
)
from .loads import load_ratio, overloaded
from .pallet_model import G, box_tolerance, cog_delta

EVIDENCE_SOURCE = "LBCP_EMS_DELTA_V1"
# Finite stand-in for an unbounded load ratio (a 0 N capacity box carrying
# load). ConstraintEvidence requires finite numbers.
RATIO_CAP = 1e6


@dataclass(frozen=True)
class MaskOutcome:
    codes: tuple
    reasons: tuple
    metrics: dict
    evidence: object  # ConstraintEvidence or None

    @property
    def valid(self):
        return not self.codes


def _clip01(value):
    return min(1.0, max(0.0, value))


def pallet_cog_limits(model, mass_after):
    cfg = model.config.constraints.pallet_cog
    p = model.pallet_size
    ramp = 1.0
    if cfg.ramp_mass_kg > 0:
        ramp = min(1.0, mass_after / cfg.ramp_mass_kg)
    ratio = cfg.final_half_extent_ratio + (1.0 - cfg.final_half_extent_ratio) * (
        1.0 - ramp
    )
    return 0.5 * p.x * ratio, 0.5 * p.y * ratio


def evaluate(model, box, pose, *, is_uncertain=False, collect_all=True):
    """Evaluate the hard constraints for ``box`` at ``pose`` on ``model``."""
    cfg = model.config
    cons = cfg.constraints
    unc = cfg.uncertainty
    codes = []
    reasons = []
    metrics = {}

    def fail(code, reason):
        if code not in codes:
            codes.append(code)
        reasons.append(reason)

    def rejected():
        return MaskOutcome(tuple(codes), tuple(reasons), metrics, None)

    # 1. orientation ---------------------------------------------------
    if abs(pose.roll) > 1e-9 or abs(pose.pitch) > 1e-9:
        fail(R.INVALID_STATE, "ORIENTATION_NOT_UPRIGHT")
        return MaskOutcome(tuple(codes), tuple(reasons), metrics, None)
    try:
        quarter_turns(pose.yaw)
    except ValueError:
        fail(R.INVALID_STATE, "ORIENTATION_NOT_AXIS_ALIGNED")
        return MaskOutcome(tuple(codes), tuple(reasons), metrics, None)
    if not any(same_yaw(pose.yaw, y) for y in box.allowed_yaws_rad):
        fail(R.INVALID_STATE, "ORIENTATION_NOT_ALLOWED")
        if not collect_all:
            return rejected()

    dx, dy, dz = rotated_dims(box.size, pose.yaw)
    tol = box_tolerance(unc, is_uncertain)
    rect = footprint(pose.x, pose.y, dx, dy)
    expanded = rect.expand(model.half_gap + tol)
    z = pose.z
    top = z + dz
    metrics.update(
        footprint_x_m=dx, footprint_y_m=dy, top_m=top, size_tolerance_m=tol
    )

    # 2. boundary --------------------------------------------------------
    if not model.pallet_rect.contains_rect(rect.expand(tol), tol=1e-9):
        fail(R.OUT_OF_BOUND, "PALLET_BOUNDARY")

    # 3. height -------------------------------------------------------------
    if z < -model.height_tol:
        fail(R.INVALID_STATE, "BELOW_PALLET_SURFACE")
    if top + tol > model.pallet_size.z + 1e-9:
        fail(R.HEIGHT_LIMIT, "MAX_STACK_HEIGHT")
    if codes and not collect_all:
        return rejected()

    # 6. resting contact (cheap, vectorised) -------------------------------
    rest = model.resting_z(expanded)
    metrics["resting_z_m"] = rest
    if z > rest + model.height_tol:
        fail(R.LOW_SUPPORT, "FLOATING")
        if not collect_all:
            return rejected()

    # 4./5. collision, clearance and descent column ------------------------
    for i in model.overlapping_expanded(expanded):
        g = model.boxes[i]
        actual_overlap = g.rect.overlaps(rect)
        vertical_overlap = (
            g.top > z + model.height_tol and g.bottom < top - model.height_tol
        )
        if vertical_overlap:
            fail(
                R.BOX_COLLISION,
                ("OVERLAP:" if actual_overlap else "CLEARANCE:") + g.box_id,
            )
        elif g.bottom >= top - model.height_tol and g.top > z:
            fail(R.APPROACH_FAIL, "OVERHEAD_OCCUPIED:" + g.box_id)
    if codes and not collect_all:
        return rejected()

    # 11. pallet max load (cheap) ------------------------------------------
    mass_after = model.mass_kg + box.weight_kg
    if mass_after > model.max_weight_kg + 1e-9:
        fail(R.LOAD_VIOLATION, "PALLET_MAX_WEIGHT")
        if not collect_all:
            return rejected()

    # 7./8. support area and LBCP stability ------------------------------------
    on_floor, found = model.contacts_for(rect, tol, z)
    if on_floor:
        support_ratio = 1.0
    else:
        support_ratio = min(1.0, sum(a for _, a, _ in found) / rect.area)
    metrics["support_ratio"] = support_ratio
    if support_ratio + 1e-9 < cons.min_support_ratio:
        fail(R.LOW_SUPPORT, "SUPPORT_RATIO")
        if not collect_all:
            return rejected()

    polygon = model.support_polygon(rect, tol, on_floor, found)
    center = rect.center
    ddx, ddy = cog_delta(unc, dx, dy, is_uncertain)
    lbcp_margin = inside_margin(polygon, center, ddx, ddy)
    center_margin = inside_margin(polygon, center)
    metrics.update(
        support_polygon=tuple(polygon),
        lbcp_margin_m=lbcp_margin if math.isfinite(lbcp_margin) else -1.0,
        cog_delta_m=(ddx, ddy),
        on_floor=on_floor,
    )
    if lbcp_margin < cons.min_lbcp_margin_m - STABILITY_EPS:
        fail(R.COG_VIOLATION, "LBCP_UNSTABLE")
        if not collect_all:
            return rejected()

    # 9./10. loads --------------------------------------------------------------
    weight_n = box.weight_kg * G
    contacts = ()
    if not on_floor and found:
        contacts = model.contact_list(rect, found, center)
    tipping = []
    extra = model.propagate(contacts, weight_n, unstable=tipping)
    for box_id in tipping:
        fail(R.COG_VIOLATION, "SUPPORTER_TIPS:" + box_id)
    max_ratio = model.base_max_load_ratio
    load_margin = model.base_min_load_margin
    for box_id, add in extra.items():
        g = model.by_id[box_id]
        load = model.top_load_n[box_id] + add
        ratio = load_ratio(load, g.capacity_n)
        max_ratio = max(max_ratio, ratio)
        load_margin = min(load_margin, 1.0 - ratio)
        if overloaded(load, g.capacity_n):
            fail(R.LOAD_VIOLATION, "BOX_CAPACITY:" + box_id)
    hol = cons.heavy_on_light
    if hol.enabled:
        for contact in contacts:
            # per_box: supporters carrying a negligible share are ignored;
            # share: the compared load is already scaled by the share, so
            # every supporter is checked (no min_share loophole)
            if hol.mode != "share" and contact.share < hol.min_share:
                continue
            sup = model.by_id[contact.supporter_id]
            load_kg = box.weight_kg * (contact.share if hol.mode == "share" else 1.0)
            if load_kg > hol.max_weight_ratio * sup.weight_kg + hol.tolerance_kg:
                fail(R.LOAD_VIOLATION, "HEAVY_ON_LIGHT:" + sup.box_id)
    metrics.update(
        supporter_shares={c.supporter_id: c.share for c in contacts},
        max_load_ratio=min(max_ratio, RATIO_CAP),
    )

    # 12. pallet CoG region --------------------------------------------------------
    p = model.pallet_size
    if mass_after > 0:
        cx = (model.moment[0] + box.weight_kg * center[0]) / mass_after
        cy = (model.moment[1] + box.weight_kg * center[1]) / mass_after
        cz = (model.moment[2] + box.weight_kg * (z + 0.5 * dz)) / mass_after
    else:  # massless box on an empty pallet: CoG undefined -> centre
        cx, cy, cz = 0.5 * p.x, 0.5 * p.y, 0.0
    hx, hy = pallet_cog_limits(model, mass_after)
    off_x, off_y = abs(cx - 0.5 * p.x), abs(cy - 0.5 * p.y)
    pallet_cog_ratio = min(
        _clip01((hx - off_x) / hx) if hx > 0 else 0.0,
        _clip01((hy - off_y) / hy) if hy > 0 else 0.0,
    )
    metrics.update(pallet_cog_after_m=(cx, cy, cz), pallet_cog_limit_m=(hx, hy))
    pc = cons.pallet_cog
    if pc.enabled and (off_x > hx + 1e-9 or off_y > hy + 1e-9):
        improving = False
        if pc.allow_improving and model.mass_kg > 0:
            bx = abs(model.moment[0] / model.mass_kg - 0.5 * p.x)
            by = abs(model.moment[1] / model.mass_kg - 0.5 * p.y)
            improving = (off_x <= hx + 1e-9 or off_x < bx - 1e-9) and (
                off_y <= hy + 1e-9 or off_y < by - 1e-9
            )
        if not improving:
            fail(R.COG_VIOLATION, "PALLET_COG")

    if codes:
        return rejected()

    # Evidence for stage 5-3/5-6 (all normalised as the team contract asks).
    half_min = 0.5 * min(dx, dy)
    lbcp_ratio = _clip01(lbcp_margin / half_min) if half_min > 0 else 0.0
    if on_floor:
        centering = 0.5
    else:
        centering = min(0.5, max(0.0, center_margin / min(dx, dy)))
    dependency = len(model.ancestors(extra.keys())) if extra else 0
    evidence = ConstraintEvidence(
        support_ratio=_clip01(support_ratio),
        cog_margin_ratio=min(lbcp_ratio, pallet_cog_ratio),
        load_margin_ratio=_clip01(load_margin),
        pallet_load_margin_ratio=_clip01(1.0 - mass_after / model.max_weight_kg),
        max_load_ratio=min(max(0.0, max_ratio), RATIO_CAP),
        support_centering=centering,
        dependency_count=dependency,
        source=EVIDENCE_SOURCE,
    )
    metrics.update(
        lbcp_margin_ratio=lbcp_ratio,
        pallet_cog_margin_ratio=pallet_cog_ratio,
        dependency_count=dependency,
    )
    return MaskOutcome((), (), metrics, evidence)


__all__ = ["EVIDENCE_SOURCE", "MaskOutcome", "evaluate", "LEN_EPS"]
