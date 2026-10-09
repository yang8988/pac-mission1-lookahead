"""Stage 5-1: placement candidate generation.

Candidate set = EMS anchors U Extreme Points, for every allowed yaw.

* EMS (heightmap based): Empty Maximal Spaces of the coordinate-compressed
  heightmap, computed in the delta-inflated plane so that every anchored box
  automatically keeps the required clearance from its neighbours.
  Each EMS gives 5 reference points: 4 corners + centre.
* Extreme Points: corner points created by the placed boxes (right/front
  neighbours, their projections toward the origin walls, top-face corner).
* Orientation: ``config.yaw_set_rad`` (1st stage: 0 / 90 deg) intersected
  with ``box.allowed_yaws_rad``; yaws with the same AABB are kept once.
* Duplicates: candidates of the same yaw within ``dedup_distance_m`` in x
  and y (and the same level) are merged (80 mm by default).

``z`` is never taken from the EMS: it is the drop height of the inflated
footprint on the current heightmap, so a candidate always rests on
something. Feasibility is decided by the hard mask (5-2), not here.
"""

from dataclasses import dataclass
import math

import numpy as np

from .geometry import LEN_EPS, Rect, quarter_turns, rotated_dims, same_yaw
from .pallet_model import _cluster_levels, box_tolerance


@dataclass(frozen=True)
class RawCandidate:
    x: float
    y: float
    z: float
    yaw: float
    dims: tuple  # rotated (dx, dy, dz)
    source: str  # "EMS" | "EP"
    anchor: str  # corner_ll ... center | ep_* name
    ems: object  # Ems or None (EMS of origin / containing EMS)
    expanded: Rect
    support_est: float = 1.0  # vectorised pre-mask support ratio estimate
    proxy_ok: bool = True  # passes the cheap pre-mask estimate
    rank: tuple = ()  # precomputed priority key

    def priority(self):
        """Deepest-bottom-left-fill order; corners before centres."""
        return self.rank or _rank(self.x, self.y, self.z, self.anchor, self.yaw)


def _rank(x, y, z, anchor, yaw):
    # micrometre integer keys: exact ties, cheaper than round()
    return (
        int(z * 1e6 + 0.5),
        int(y * 1e6 + 0.5),
        int(x * 1e6 + 0.5),
        1 if anchor == "center" else 0,
        yaw,
    )


def candidate_yaws(box, config):
    """Allowed yaws in configured order; equal-footprint yaws deduplicated.

    A configured yaw is matched to the box's own allowed yaw with the same
    footprint (same quarter-turn parity), and that allowed value is output,
    so boxes listing e.g. ``pi`` or ``-pi/2`` still get candidates and the
    hard mask's orientation check accepts them.
    """
    allowed = []
    for a in box.allowed_yaws_rad:
        try:
            allowed.append((a, quarter_turns(a) % 2))
        except ValueError:
            continue
    result = []
    footprints = set()
    # empty yaw_set_rad: take every orientation the SKU itself allows
    yaws = config.generation.yaw_set_rad or tuple(a for a, _ in allowed)
    for yaw in yaws:
        try:
            parity = quarter_turns(yaw) % 2
        except ValueError:
            continue
        exact = [a for a, _ in allowed if same_yaw(yaw, a)]
        same_fp = [a for a, par in allowed if par == parity]
        if not (exact or same_fp):
            continue
        chosen = (exact or same_fp)[0]
        dims = rotated_dims(box.size, chosen)
        key = (round(dims[0], 9), round(dims[1], 9))
        if key in footprints:
            continue
        footprints.add(key)
        result.append(chosen)
    return result


def _balance_raw(model, box, dims, margin, hol, min_support, step, max_level=math.inf, keep=6,
                 spacing=0.08):
    """Balanced anchors for heavy-on-light ``share`` mode.

    A heavy box may rest on lighter boxes only where its weight is split
    between them, often a narrow window over a seam or a junction of several
    supporters that corner/centre anchors miss. For every top-surface level
    that has a box lighter than the new one, the area covered by that level's
    boxes is scanned once on a ``step`` grid with the vectorised estimates
    (drop height = level, support ratio, load split); up to ``keep`` passing
    points and ``keep // 2`` nearly passing ones (the area estimate can be
    stricter than the mask's lever shares), ``spacing`` apart, become anchors.
    The hard mask still re-checks every candidate exactly.
    """
    width = dims[0] + 2.0 * margin
    depth = dims[1] + 2.0 * margin
    bound = model.expanded_bounds
    limit = hol.max_weight_ratio
    # one scan per top-surface level; tops within height_tol form one level
    # (same clustering as the EMS levels, so a level is never split in two)
    tops = sorted(g.top for g in model.boxes)
    groups = [
        [g for g in model.boxes if lo - LEN_EPS <= g.top <= hi + LEN_EPS]
        for lo, hi in _cluster_levels(tops, model.height_tol)
    ]
    out = []
    for group in groups:
        level = max(g.top for g in group)
        if level <= model.height_tol or level >= max_level:
            continue
        if level + dims[2] > model.pallet_size.z + 1e-9:
            continue
        if box.weight_kg <= limit * min(g.weight_kg for g in group) + hol.tolerance_kg:
            continue
        x0 = max(bound.x0, min(g.expanded.x0 for g in group) - width)
        y0 = max(bound.y0, min(g.expanded.y0 for g in group) - depth)
        x1 = min(bound.x1, max(g.expanded.x1 for g in group) + width) - width
        y1 = min(bound.y1, max(g.expanded.y1 for g in group) + depth) - depth
        xs = np.arange(x0, x1 + LEN_EPS, step)
        ys = np.arange(y0, y1 + LEN_EPS, step)
        if xs.size == 0 or ys.size == 0:
            continue
        gx, gy = np.meshgrid(xs, ys, indexing="ij")
        gx, gy = gx.ravel(), gy.ravel()
        rects = np.stack([gx, gy, gx + width, gy + depth], axis=1)
        zs = model.resting_z_many(rects)
        actual = rects + np.array([margin, margin, -margin, -margin])
        sup, heavy, excess = model.support_estimates(actual, zs, box.weight_kg, hol, with_excess=True)
        base = (np.abs(zs - level) <= model.height_tol) & (sup >= min_support - 1e-9)
        # most balanced first: the edge of the feasible window is where the
        # area estimate and the mask's lever shares disagree most
        passing = sorted(
            np.flatnonzero(base & ~heavy).tolist(), key=lambda i: (round(float(excess[i]), 2), gy[i], gx[i])
        )
        nearly = sorted(np.flatnonzero(base & heavy).tolist(), key=lambda i: (excess[i], gy[i], gx[i]))
        chosen = []
        for order, quota in ((passing, keep), (nearly, keep // 2)):
            taken = 0
            for i in order:
                x, y = float(gx[i]), float(gy[i])
                if all(abs(x - px) >= spacing or abs(y - py) >= spacing for px, py in chosen):
                    chosen.append((x, y))
                    taken += 1
                    if taken >= quota:
                        break
        out.extend((x, y, "EMS", "balance", None) for x, y in chosen)
    return out


def _ems_raw(model, box, yaw, dims, margin, anchors):
    out = []
    width = dims[0] + 2.0 * margin
    depth = dims[1] + 2.0 * margin
    for ems in model.ems():
        r = ems.rect
        if r.width + LEN_EPS < width or r.depth + LEN_EPS < depth:
            continue
        if ems.level + dims[2] > ems.top + LEN_EPS:
            continue
        cx, cy = r.center
        points = {
            "corner_ll": (r.x0, r.y0),
            "corner_lr": (r.x1 - width, r.y0),
            "corner_ul": (r.x0, r.y1 - depth),
            "corner_ur": (r.x1 - width, r.y1 - depth),
            "center": (cx - 0.5 * width, cy - 0.5 * depth),
        }
        for anchor in anchors:
            ex, ey = points[anchor]
            out.append((ex, ey, "EMS", anchor, ems))
    return out


def _project(model, x, y, z, axis):
    """Slide (x, y) toward the origin wall along ``axis`` (0 = x, 1 = y)
    until it meets an inflated box whose top is above ``z``."""
    bound = model.expanded_bounds
    best = bound.x0 if axis == 0 else bound.y0
    for g in model.boxes:
        if g.top <= z + model.height_tol:
            continue
        e = g.expanded
        if axis == 0:
            if e.y0 <= y + LEN_EPS and y < e.y1 - LEN_EPS and e.x1 <= x + LEN_EPS:
                best = max(best, e.x1)
        else:
            if e.x0 <= x + LEN_EPS and x < e.x1 - LEN_EPS and e.y1 <= y + LEN_EPS:
                best = max(best, e.y1)
    return best


def extreme_points(model):
    """Inflated-plane extreme points (anchor name, x, y); cached per model."""
    if model._extreme_points is None:
        model._extreme_points = tuple(_compute_extreme_points(model))
    return model._extreme_points


def _compute_extreme_points(model):
    bound = model.expanded_bounds
    points = [("ep_origin", bound.x0, bound.y0)]
    for g in model.boxes:
        e = g.expanded
        z = g.bottom
        points.append(("ep_right", e.x1, e.y0))
        points.append(("ep_front", e.x0, e.y1))
        points.append(("ep_right_proj", e.x1, _project(model, e.x1, e.y0, z, 1)))
        points.append(("ep_front_proj", _project(model, e.x0, e.y1, z, 0), e.y1))
        points.append(("ep_top", e.x0, e.y0))
    return points


def _ep_raw(model, box, dims, margin):
    width = dims[0] + 2.0 * margin
    depth = dims[1] + 2.0 * margin
    bound = model.expanded_bounds
    out = []
    for anchor, ex, ey in extreme_points(model):
        if ex + width > bound.x1 + LEN_EPS or ey + depth > bound.y1 + LEN_EPS:
            continue
        if ex < bound.x0 - LEN_EPS or ey < bound.y0 - LEN_EPS:
            continue
        out.append((ex, ey, "EP", anchor, None))
    return out


def containing_ems(model, expanded, z, tol=1e-7):
    """Largest EMS at ``z``'s level that contains the inflated footprint.

    Vectorised over the snapshot's EMS (arrays cached on the model); same
    result as checking ``ems.rect.contains_rect`` one by one.
    """
    arr = getattr(model, "_ems_arrays", None)
    if arr is None:
        ems = model.ems()
        arr = (
            ems,
            np.array([(e.rect.x0, e.rect.y0, e.rect.x1, e.rect.y1) for e in ems], dtype=float).reshape(-1, 4),
            np.array([e.level for e in ems], dtype=float),
            np.array([e.rect.area for e in ems], dtype=float),
        )
        model._ems_arrays = arr
    ems, rects, levels, areas = arr
    if not ems:
        return None
    ok = (
        (np.abs(levels - z) <= model.height_tol)
        & (rects[:, 0] <= expanded.x0 + tol)
        & (rects[:, 1] <= expanded.y0 + tol)
        & (rects[:, 2] >= expanded.x1 - tol)
        & (rects[:, 3] >= expanded.y1 - tol)
    )
    idx = np.flatnonzero(ok)
    if idx.size == 0:
        return None
    return ems[int(idx[np.argmax(areas[idx])])]


def _lowest_valid_z(model, box, yaw, anchors, zs_arr, ok_arr, margin, tries=6):
    """Lowest z of a likely-valid anchor that passes the exact hard mask."""
    from pac_common import Pose3D

    from .hard_mask import evaluate

    uncertain = box.box_id in model.uncertain_ids
    order = sorted(np.flatnonzero(ok_arr).tolist(), key=lambda i: zs_arr[i])
    for i in order[:tries]:
        ex, ey = anchors[i][0], anchors[i][1]
        pose = Pose3D("pallet", ex + margin, ey + margin, float(zs_arr[i]), yaw=yaw)
        if not evaluate(model, box, pose, is_uncertain=uncertain, collect_all=False).codes:
            return float(zs_arr[i])
    return math.inf


def _estimate(model, box, config, anchors, w, d, margin, dims, tol):
    """Drop height, support estimate and pre-mask verdict per anchor."""
    if not anchors:
        return np.zeros(0), np.zeros(0), np.zeros(0, dtype=bool)
    rects = np.array([(ex, ey, ex + w, ey + d) for ex, ey, _, _, _ in anchors], dtype=float)
    zs_arr = model.resting_z_many(rects)
    actual = rects + np.array([margin, margin, -margin, -margin])
    sup_arr, heavy_arr = model.support_estimates(
        actual, zs_arr, box.weight_kg, config.constraints.heavy_on_light
    )
    ok_arr = (
        (sup_arr >= config.constraints.min_support_ratio - 1e-9)
        & ~heavy_arr
        & (zs_arr + dims[2] + tol <= model.pallet_size.z + 1e-9)
    )
    return zs_arr, sup_arr, ok_arr


def raw_candidates(model, box, config):
    """All candidates before deduplication, sorted by priority."""
    gen = config.generation
    uncertain = box.box_id in model.uncertain_ids
    tol = box_tolerance(config.uncertainty, uncertain)
    margin = model.half_gap + tol
    seen = set()
    result = []
    hol = config.constraints.heavy_on_light
    balance = gen.use_ems and gen.balance_anchors and hol.enabled and hol.mode == "share"
    for yaw in candidate_yaws(box, config):
        dims = rotated_dims(box.size, yaw)
        anchors = []
        if gen.use_ems:
            anchors.extend(_ems_raw(model, box, yaw, dims, margin, gen.ems_anchors))
        if gen.use_extreme_points:
            anchors.extend(_ep_raw(model, box, dims, margin))
        w = dims[0] + 2.0 * margin
        d = dims[1] + 2.0 * margin
        zs_arr, sup_arr, ok_arr = _estimate(model, box, config, anchors, w, d, margin, dims, tol)
        if balance and model.boxes:
            # only levels below the lowest regular anchor that the exact hard
            # mask accepts can improve the (deepest-first) result; the cheap
            # estimate alone is not a verdict (capacity, LBCP, CoG, lever
            # shares), so the cutoff is confirmed with the mask
            best = _lowest_valid_z(model, box, yaw, anchors, zs_arr, ok_arr, margin)
            extra = _balance_raw(
                model, box, dims, margin, hol, config.constraints.min_support_ratio,
                gen.balance_step_m, max_level=best - model.height_tol,
            )
            if extra:
                z2, s2, o2 = _estimate(model, box, config, extra, w, d, margin, dims, tol)
                anchors.extend(extra)
                zs_arr = np.concatenate([zs_arr, z2])
                sup_arr = np.concatenate([sup_arr, s2])
                ok_arr = np.concatenate([ok_arr, o2])
        if not anchors:
            continue
        zs = zs_arr.tolist()
        for (ex, ey, source, anchor, ems), z, sup, ok in zip(
            anchors, zs, sup_arr.tolist(), ok_arr.tolist()
        ):
            expanded = Rect(ex, ey, ex + w, ey + d)
            x = ex + margin
            y = ey + margin
            rank = _rank(x, y, z, anchor, yaw)
            key = rank[:3] + (yaw,)
            if key in seen:
                continue
            seen.add(key)
            if ems is not None and abs(ems.level - z) > model.height_tol:
                ems = None  # resolved lazily for kept candidates only
            result.append(
                RawCandidate(
                    x, y, float(z), yaw, dims, source, anchor, ems, expanded,
                    float(sup), bool(ok), rank,
                )
            )
    result.sort(key=RawCandidate.priority)
    return result


def deduplicate(raws, distance, height_tol, is_valid=None):
    """Greedy duplicate removal within ``distance`` (Chebyshev in x/y).

    ``raws`` must be in priority order. The first candidate of a cluster is
    kept. With ``is_valid`` (mask-aware mode) a kept candidate that fails the
    hard mask is replaced by a later duplicate that passes it, so dedup never
    hides the only valid neighbour. Validity is evaluated lazily, only for
    candidates that actually collide. Returns (kept, validity-by-index).
    """
    if distance <= 0.0:
        return list(raws), {}
    validity = {}

    def valid(i):
        if i not in validity:
            validity[i] = bool(is_valid(raws[i]))
        return validity[i]

    kept = set()
    buckets = {}
    location = {}

    def conflicts(c, bx, by):
        for gx in (bx - 1, bx, bx + 1):
            for gy in (by - 1, by, by + 1):
                for j in buckets.get((gx, gy), ()):
                    o = raws[j]
                    if (
                        abs(o.x - c.x) < distance - 1e-12
                        and abs(o.y - c.y) < distance - 1e-12
                        and abs(o.z - c.z) <= height_tol
                        and same_yaw(o.yaw, c.yaw)
                    ):
                        yield j

    def keep(i, bx, by):
        kept.add(i)
        location[i] = (bx, by)
        buckets.setdefault((bx, by), []).append(i)

    dropped = []
    replaced = False
    for i, c in enumerate(raws):
        bx = math.floor(c.x / distance)
        by = math.floor(c.y / distance)
        hits = list(conflicts(c, bx, by))
        if hits:
            if is_valid is None or not valid(i):
                dropped.append(i)
                continue
            invalid_hits = [j for j in hits if not valid(j)]
            if len(invalid_hits) != len(hits):
                dropped.append(i)
                continue  # a valid representative already exists
            for j in invalid_hits:
                kept.discard(j)
                buckets[location[j]].remove(j)
                dropped.append(j)
            replaced = True
        keep(i, bx, by)
    if replaced:
        # A replaced representative may have been the only thing that hid
        # an earlier candidate further away: give dropped candidates another
        # look (priority order) so no region loses its representative.
        for i in sorted(dropped):
            if i in kept:
                continue
            c = raws[i]
            bx = math.floor(c.x / distance)
            by = math.floor(c.y / distance)
            if not any(True for _ in conflicts(c, bx, by)):
                keep(i, bx, by)
    return [raws[i] for i in sorted(kept)], validity
