"""Derived, immutable geometric model of one pallet snapshot.

Built once per ``SystemState`` (cached by the backend) and shared by the
candidate generator (5-1) and the hard mask (5-2):

* per-box AABB, measurement tolerance and the delta-inflated footprint,
* support graph (which box rests on which, with contact areas),
* LBCP (load-bearing convex polygon) of every placed box,
* cumulative top load of every box and its allowable capacity,
* pallet mass / CoG totals,
* a coordinate-compressed heightmap and Empty Maximal Spaces (lazy).

LBCP (Load Bearable Convex Polygon): a box resting on the pallet bears load
anywhere on its (tolerance-shrunk) footprint. A box resting on other boxes
bears load inside the convex hull of its contact regions clipped to the
supporters' own LBCPs. A placement is stable when the worst-case CoG (centre
+- delta) lies inside the support polygon; every force it transmits then acts
inside the supporters' LBCPs, so the boxes below stay stable as well.
"""

from dataclasses import dataclass
import math

import numpy as np

from .geometry import (
    LEN_EPS,
    STABILITY_EPS,
    Rect,
    clip_polygon_to_rect,
    convex_hull,
    footprint,
    inside_margin,
    polygon_area,
    polygon_centroid,
    rotated_dims,
    shrink_rect_polygon,
)
from .loads import load_ratio, mckee_capacity_n, overloaded, split_force

G = 9.80665


@dataclass(frozen=True)
class Contact:
    supporter_id: str
    area_m2: float  # actual footprint intersection (unshrunk)
    polygon: tuple  # contact region clipped to the supporter's LBCP
    centroid: tuple
    share: float  # fraction of this box's total downward force


@dataclass(frozen=True)
class BoxGeom:
    box_id: str
    sku_id: str
    weight_kg: float
    lo: tuple
    hi: tuple
    rect: Rect  # actual footprint
    tolerance_m: float  # +- size tolerance (uncertain boxes multiplied)
    expanded: Rect  # footprint + half clearance + tolerance
    capacity_n: float
    uncertain: bool

    @property
    def top(self):
        return self.hi[2]

    @property
    def bottom(self):
        return self.lo[2]


@dataclass(frozen=True)
class Ems:
    """Empty maximal space in the delta-inflated (expanded) plane."""

    rect: Rect  # expanded-space rectangle
    level: float  # floor height z
    top: float  # ceiling (max stack height)


def box_tolerance(uncertainty, uncertain):
    tol = uncertainty.size_tolerance_m
    return tol * uncertainty.uncertain_multiplier if uncertain else tol


def capacity_for(box_id, sku_id, dx, dy, context, load_cfg):
    if context is not None:
        if box_id in context.capacity_overrides_n:
            return float(context.capacity_overrides_n[box_id])
        spec = context.catalog.get(sku_id)
        if spec is not None:
            return float(spec.top_load_capacity_n)
    return mckee_capacity_n(
        dx,
        dy,
        load_cfg.ect_n_per_m,
        load_cfg.board_thickness_m,
        load_cfg.safety_factor,
    )


class PalletModel:
    def __init__(self, state, context, config):
        self.config = config
        self.context = context
        unc = config.uncertainty
        self.height_tol = unc.height_tolerance_m
        self.half_gap = 0.5 * unc.lateral_clearance_m
        pallet = state.pallet
        self.pallet_size = pallet.size
        overhang = config.constraints.pallet_overhang_m
        self.pallet_rect = Rect(
            -overhang, -overhang, pallet.size.x + overhang, pallet.size.y + overhang
        )
        self.expanded_bounds = self.pallet_rect.expand(self.half_gap)
        self.uncertain_ids = (
            frozenset(context.uncertain_box_ids) if context else frozenset()
        )
        self.max_weight_kg = (
            float(context.pallet_max_weight_kg)
            if context is not None
            else config.constraints.default_pallet_max_weight_kg
        )
        geoms = []
        for placed in pallet.boxes:
            pose = placed.pose
            if abs(pose.roll) > 1e-9 or abs(pose.pitch) > 1e-9:
                raise ValueError("Only upright placed boxes are supported")
            dx, dy, dz = rotated_dims(placed.size, pose.yaw)
            uncertain = placed.box_id in self.uncertain_ids
            tol = box_tolerance(unc, uncertain)
            rect = footprint(pose.x, pose.y, dx, dy)
            geoms.append(
                BoxGeom(
                    box_id=placed.box_id,
                    sku_id=placed.sku_id,
                    weight_kg=float(placed.weight_kg),
                    lo=(pose.x, pose.y, pose.z),
                    hi=(pose.x + dx, pose.y + dy, pose.z + dz),
                    rect=rect,
                    tolerance_m=tol,
                    expanded=rect.expand(self.half_gap + tol),
                    capacity_n=capacity_for(
                        placed.box_id,
                        placed.sku_id,
                        dx,
                        dy,
                        context,
                        config.constraints.load_model,
                    ),
                    uncertain=uncertain,
                )
            )
        geoms.sort(key=lambda g: (g.bottom, g.box_id))
        self.boxes = tuple(geoms)
        self.by_id = {g.box_id: g for g in geoms}
        self.index = {g.box_id: i for i, g in enumerate(geoms)}
        n = len(geoms)
        self._exp = np.array(
            [(g.expanded.x0, g.expanded.y0, g.expanded.x1, g.expanded.y1) for g in geoms],
            dtype=float,
        ).reshape(n, 4)
        self._act = np.array(
            [(g.rect.x0, g.rect.y0, g.rect.x1, g.rect.y1) for g in geoms],
            dtype=float,
        ).reshape(n, 4)
        self._top = np.array([g.top for g in geoms], dtype=float)
        self._bottom = np.array([g.bottom for g in geoms], dtype=float)
        self._weight = np.array([g.weight_kg for g in geoms], dtype=float)
        self._build_support()
        self._build_loads()
        self.mass_kg = sum(g.weight_kg for g in geoms)
        self.moment = (
            sum(g.weight_kg * 0.5 * (g.lo[0] + g.hi[0]) for g in geoms),
            sum(g.weight_kg * 0.5 * (g.lo[1] + g.hi[1]) for g in geoms),
            sum(g.weight_kg * 0.5 * (g.lo[2] + g.hi[2]) for g in geoms),
        )
        self.max_top = max((g.top for g in geoms), default=0.0)
        self._ems = None
        self._grid = None
        self._extreme_points = None
        # Per-snapshot result caches owned by CandidateBackend (freed together
        # with this model when it leaves the backend's LRU).
        self.verdict_cache = {}
        self.generation_cache = {}

    # ------------------------------------------------------------------
    # Support graph and LBCP
    # ------------------------------------------------------------------

    @staticmethod
    def _overlap_mask(boxes, rect, tol=LEN_EPS):
        return (
            (np.minimum(boxes[:, 2], rect.x1) - np.maximum(boxes[:, 0], rect.x0) > tol)
            & (np.minimum(boxes[:, 3], rect.y1) - np.maximum(boxes[:, 1], rect.y0) > tol)
        )

    def overlapping_expanded(self, expanded_rect):
        """Indices of boxes whose inflated footprint overlaps ``expanded_rect``."""
        if not self.boxes:
            return ()
        return np.flatnonzero(self._overlap_mask(self._exp, expanded_rect)).tolist()

    def contacts_for(self, rect, tolerance, bottom, exclude=None):
        """Supporters of a footprint resting at ``bottom``.

        Returns (on_floor, [(geom, actual_area, clipped_polygon)]).
        """
        if bottom <= self.height_tol:
            return True, []
        shrunk_poly = shrink_rect_polygon(rect, tolerance)
        if not shrunk_poly:
            return False, []
        shrunk = Rect(*shrunk_poly[0], *shrunk_poly[2])
        found = []
        if not self.boxes:
            return False, found
        level = np.abs(self._top - bottom) <= self.height_tol
        hits = np.flatnonzero(level & self._overlap_mask(self._act, rect))
        for i in hits.tolist():
            geom = self.boxes[i]
            if geom.box_id == exclude:
                continue
            inter = rect.intersection(geom.rect)
            if inter is None:
                continue
            lbcp = self.lbcp.get(geom.box_id, ())
            clipped = clip_polygon_to_rect(lbcp, shrunk) if lbcp else ()
            found.append((geom, inter.area, clipped))
        return False, found

    def support_polygon(self, rect, tolerance, on_floor, found):
        if on_floor:
            return shrink_rect_polygon(rect, tolerance)
        pts = []
        for _, _, poly in found:
            pts.extend(poly)
        return convex_hull(pts) if len(pts) >= 3 else tuple(pts)

    def _build_support(self):
        self.lbcp = {}
        self.on_floor = {}
        self.raw_contacts = {}
        for geom in self.boxes:  # ascending bottom height
            on_floor, found = self.contacts_for(
                geom.rect, geom.tolerance_m, geom.bottom, exclude=geom.box_id
            )
            self.on_floor[geom.box_id] = on_floor
            self.raw_contacts[geom.box_id] = found
            self.lbcp[geom.box_id] = self.support_polygon(
                geom.rect, geom.tolerance_m, on_floor, found
            )

    # ------------------------------------------------------------------
    # Loads (top-down propagation, see loads.py)
    # ------------------------------------------------------------------

    def contact_list(self, rect, found, point):
        """Contacts with load shares for a force applied at ``point``.

        A supporter bears load only inside its LBCP, so each contact is the
        overlap clipped to the supporter's LBCP (``found`` already holds that
        polygon): its area and centroid drive the share. Contacts without a
        load-bearing part (e.g. resting on a supporter's unsupported
        overhang) carry no load.
        """
        usable = []
        for g, _, poly in found:
            if len(poly) < 3:
                continue
            area = polygon_area(poly)
            if area > LEN_EPS * LEN_EPS:
                usable.append((g, area, poly))
        if not usable:
            return ()
        areas = [a for _, a, _ in usable]
        centroids = [polygon_centroid(poly) for _, _, poly in usable]
        shares = split_force(
            self.config.constraints.load_model.share_model,
            areas,
            centroids,
            point,
        )
        return tuple(
            Contact(g.box_id, a, poly, c, s)
            for (g, a, poly), c, s in zip(usable, centroids, shares)
        )

    def _build_loads(self):
        total = {g.box_id: g.weight_kg * G for g in self.boxes}
        moment = {
            g.box_id: (
                g.weight_kg * G * g.rect.center[0],
                g.weight_kg * G * g.rect.center[1],
            )
            for g in self.boxes
        }
        self.contacts = {}
        for geom in reversed(self.boxes):  # top-down
            force = total[geom.box_id]
            point = (
                (moment[geom.box_id][0] / force, moment[geom.box_id][1] / force)
                if force > 0
                else geom.rect.center
            )
            contacts = self.contact_list(
                geom.rect, self.raw_contacts[geom.box_id], point
            )
            self.contacts[geom.box_id] = contacts
            for contact in contacts:
                part = force * contact.share
                total[contact.supporter_id] += part
                mx, my = moment[contact.supporter_id]
                moment[contact.supporter_id] = (
                    mx + part * contact.centroid[0],
                    my + part * contact.centroid[1],
                )
        # resultant force/moment each box passes down (for exact increments)
        self._total_n = total
        self._moment = moment
        self.top_load_n = {
            g.box_id: total[g.box_id] - g.weight_kg * G for g in self.boxes
        }
        self.base_max_load_ratio = 0.0
        self.base_min_load_margin = 1.0
        for g in self.boxes:
            ratio = load_ratio(self.top_load_n[g.box_id], g.capacity_n)
            self.base_max_load_ratio = max(self.base_max_load_ratio, ratio)
            self.base_min_load_margin = min(self.base_min_load_margin, 1 - ratio)

    def propagate(self, contacts, force_n, unstable=None):
        """Extra top load (N) on each box below when ``force_n`` is added.

        Exact for both share models: every affected box re-splits its new
        resultant (old load + extra, at the shifted point) exactly as the
        top-down pass of a full snapshot would, so the hard mask and a
        re-check of the resulting pallet always agree. (Re-using the old
        shares is only exact for the linear ``area`` model; with ``lever``
        the shares move with the resultant.)
        """
        extra_f = {}
        extra_m = {}

        def add(box_id, force, centroid):
            extra_f[box_id] = extra_f.get(box_id, 0.0) + force
            mx, my = extra_m.get(box_id, (0.0, 0.0))
            extra_m[box_id] = (mx + force * centroid[0], my + force * centroid[1])

        for contact in contacts:
            add(contact.supporter_id, force_n * contact.share, contact.centroid)
        for geom in reversed(self.boxes):  # same top-down order as _build_loads
            box_id = geom.box_id
            if box_id not in extra_f:
                continue
            old_total = self._total_n[box_id]
            total = old_total + extra_f[box_id]
            mx, my = self._moment[box_id]
            ex, ey = extra_m[box_id]
            point = ((mx + ex) / total, (my + ey) / total) if total > 0 else geom.rect.center
            if unstable is not None and not self.on_floor[box_id]:
                # safety net: the box's new resultant must stay on its LBCP
                if inside_margin(self.lbcp[box_id], point) < -STABILITY_EPS:
                    unstable.append(box_id)
            new_contacts = self.contact_list(geom.rect, self.raw_contacts[box_id], point)
            old_parts = {c.supporter_id: old_total * c.share for c in self.contacts[box_id]}
            for contact in new_contacts:
                delta = total * contact.share - old_parts.pop(contact.supporter_id, 0.0)
                if delta:
                    add(contact.supporter_id, delta, contact.centroid)
            for supporter_id, part in old_parts.items():  # not expected: same contact set
                old = next(c for c in self.contacts[box_id] if c.supporter_id == supporter_id)
                add(supporter_id, -part, old.centroid)
        return extra_f

    def ancestors(self, supporter_ids):
        seen = set()
        stack = list(supporter_ids)
        while stack:
            box_id = stack.pop()
            if box_id in seen:
                continue
            seen.add(box_id)
            stack.extend(c.supporter_id for c in self.contacts[box_id])
        return seen

    # ------------------------------------------------------------------
    # Height queries
    # ------------------------------------------------------------------

    def resting_z(self, expanded_rect):
        """Drop height: highest top among boxes whose inflated footprints
        overlap the inflated footprint of the new box (0 on the pallet)."""
        if not self.boxes:
            return 0.0
        mask = self._overlap_mask(self._exp, expanded_rect)
        if not mask.any():
            return 0.0
        return float(self._top[mask].max())

    def resting_z_many(self, rects):
        """Vectorised ``resting_z`` for an (k, 4) array of inflated rects."""
        rects = np.asarray(rects, dtype=float).reshape(-1, 4)
        if not self.boxes or rects.shape[0] == 0:
            return np.zeros(rects.shape[0])
        b = self._exp
        ox = np.minimum(rects[:, None, 2], b[None, :, 2]) - np.maximum(
            rects[:, None, 0], b[None, :, 0]
        )
        oy = np.minimum(rects[:, None, 3], b[None, :, 3]) - np.maximum(
            rects[:, None, 1], b[None, :, 1]
        )
        hit = (ox > LEN_EPS) & (oy > LEN_EPS)
        tops = np.where(hit, self._top[None, :], 0.0)
        return tops.max(axis=1)

    def support_estimates(self, rects, zs, weight_kg, heavy_cfg, with_excess=False):
        """Vectorised pre-mask estimates for actual footprints at ``zs``.

        Returns (support_ratio, heavy_on_light_violation) arrays, plus the
        worst estimated load / heavy-on-light limit ratio when
        ``with_excess`` (``> 1`` means the estimate rejects). Used only
        to order / de-duplicate candidates in 5-1; the hard mask computes the
        authoritative values (contact clipping, LBCP, lever load shares).
        """
        rects = np.asarray(rects, dtype=float).reshape(-1, 4)
        zs = np.asarray(zs, dtype=float)
        k = rects.shape[0]
        ratio = np.ones(k)
        heavy = np.zeros(k, dtype=bool)
        excess = np.zeros(k)
        if not self.boxes or k == 0:
            return (ratio, heavy, excess) if with_excess else (ratio, heavy)
        b = self._act
        ox = np.minimum(rects[:, None, 2], b[None, :, 2]) - np.maximum(
            rects[:, None, 0], b[None, :, 0]
        )
        oy = np.minimum(rects[:, None, 3], b[None, :, 3]) - np.maximum(
            rects[:, None, 1], b[None, :, 1]
        )
        level = np.abs(self._top[None, :] - zs[:, None]) <= self.height_tol
        contact = np.where(level & (ox > 0) & (oy > 0), ox * oy, 0.0)
        area = contact.sum(axis=1)
        own = (rects[:, 2] - rects[:, 0]) * (rects[:, 3] - rects[:, 1])
        lifted = zs > self.height_tol
        ratio[lifted] = np.minimum(1.0, area[lifted] / own[lifted])
        if heavy_cfg.enabled:
            share = np.divide(
                contact, area[:, None], out=np.zeros_like(contact), where=area[:, None] > 0
            )
            limit = heavy_cfg.max_weight_ratio * self._weight + heavy_cfg.tolerance_kg
            load = weight_kg * share if heavy_cfg.mode == "share" else weight_kg
            min_share = 0.0 if heavy_cfg.mode == "share" else heavy_cfg.min_share
            counted = (share >= min_share) & (share > 0)
            bad = counted & (load > limit[None, :])
            heavy = lifted & bad.any(axis=1)
            if with_excess:
                rel = np.where(counted, load / limit[None, :], 0.0)
                excess = np.where(lifted, rel.max(axis=1), 0.0)
        return (ratio, heavy, excess) if with_excess else (ratio, heavy)

    # ------------------------------------------------------------------
    # Compressed heightmap and Empty Maximal Spaces
    # ------------------------------------------------------------------

    def grid(self):
        if self._grid is None:
            b = self.expanded_bounds
            xs = {b.x0, b.x1}
            ys = {b.y0, b.y1}
            for g in self.boxes:
                e = g.expanded
                for v in (e.x0, e.x1):
                    if b.x0 < v < b.x1:
                        xs.add(v)
                for v in (e.y0, e.y1):
                    if b.y0 < v < b.y1:
                        ys.add(v)
            xs = np.array(sorted(_merge_close(sorted(xs))))
            ys = np.array(sorted(_merge_close(sorted(ys))))
            heights = np.zeros((len(ys) - 1, len(xs) - 1))
            for g in self.boxes:
                e = g.expanded
                # Edges within MERGE_TOL were merged into the next-lower
                # kept coordinate; look up with the same tolerance so a
                # merged edge maps onto that coordinate (not one cell off).
                off = MERGE_TOL + 1e-12
                c0 = np.searchsorted(xs, e.x0 - off, side="left")
                c1 = np.searchsorted(xs, e.x1 - off, side="left")
                r0 = np.searchsorted(ys, e.y0 - off, side="left")
                r1 = np.searchsorted(ys, e.y1 - off, side="left")
                c0, r0 = max(c0, 0), max(r0, 0)
                block = heights[r0:r1, c0:c1]
                np.maximum(block, g.top, out=block)
            self._grid = (xs, ys, heights)
        return self._grid

    def ems(self):
        if self._ems is None:
            self._ems = tuple(self._compute_ems())
        return self._ems

    def _compute_ems(self):
        xs, ys, heights = self.grid()
        ceiling = self.pallet_size.z
        tol = self.height_tol
        levels = _cluster_levels(np.unique(heights), tol)
        result = []
        for level_lo, level in levels:
            if level >= ceiling - tol:
                continue
            free = heights <= level + 1e-9
            floor = heights >= level_lo - 1e-9
            floor &= free
            if not floor.any():
                continue
            prefix = np.zeros((floor.shape[0] + 1, floor.shape[1] + 1), int)
            prefix[1:, 1:] = np.cumsum(np.cumsum(floor, axis=0), axis=1)
            for r0, r1, c0, c1 in maximal_rectangles(free):
                count = (
                    prefix[r1 + 1, c1 + 1]
                    - prefix[r0, c1 + 1]
                    - prefix[r1 + 1, c0]
                    + prefix[r0, c0]
                )
                if count == 0:
                    continue
                result.append(
                    Ems(
                        Rect(
                            float(xs[c0]),
                            float(ys[r0]),
                            float(xs[c1 + 1]),
                            float(ys[r1 + 1]),
                        ),
                        float(level),
                        float(ceiling),
                    )
                )
        return result

    def heightmap_raster(self, resolution_m=0.01):
        """Regular-grid heightmap (analysis/visualisation only)."""
        p = self.pallet_size
        nx = max(1, int(round(p.x / resolution_m)))
        ny = max(1, int(round(p.y / resolution_m)))
        grid = np.zeros((ny, nx))
        xc = (np.arange(nx) + 0.5) * p.x / nx
        yc = (np.arange(ny) + 0.5) * p.y / ny
        for g in self.boxes:
            mx = (xc >= g.rect.x0) & (xc <= g.rect.x1)
            my = (yc >= g.rect.y0) & (yc <= g.rect.y1)
            sub = grid[np.ix_(my, mx)]
            grid[np.ix_(my, mx)] = np.maximum(sub, g.top)
        return grid

    # ------------------------------------------------------------------
    # Diagnostics on the snapshot itself
    # ------------------------------------------------------------------

    def snapshot_issues(self):
        """Boxes of the given state that already violate stability/load."""
        issues = {}
        unc = self.config.uncertainty
        for g in self.boxes:
            codes = []
            if not self.on_floor[g.box_id]:
                poly = self.lbcp[g.box_id]
                dx = g.rect.width
                dy = g.rect.depth
                margin = _margin(poly, g.rect.center, unc, dx, dy, g.uncertain)
                if margin < self.config.constraints.min_lbcp_margin_m - STABILITY_EPS:
                    codes.append("UNSTABLE")
                force = self._total_n[g.box_id]
                if force > 0:
                    mx, my = self._moment[g.box_id]
                    if inside_margin(poly, (mx / force, my / force)) < -STABILITY_EPS:
                        codes.append("RESULTANT_OUTSIDE")
            if overloaded(self.top_load_n[g.box_id], g.capacity_n):
                codes.append("OVERLOADED")
            if codes:
                issues[g.box_id] = tuple(codes)
        return issues


def cog_delta(unc, dx, dy, uncertain):
    mult = unc.uncertain_multiplier if uncertain else 1.0
    return (
        mult * max(unc.cog_uncertainty_ratio * dx, unc.cog_uncertainty_min_m),
        mult * max(unc.cog_uncertainty_ratio * dy, unc.cog_uncertainty_min_m),
    )


def _margin(poly, center, unc, dx, dy, uncertain):
    from .geometry import inside_margin

    ddx, ddy = cog_delta(unc, dx, dy, uncertain)
    return inside_margin(poly, center, ddx, ddy)


# Same tolerance as the overlap tests (LEN_EPS): merging coordinates further
# apart would let an EMS edge sit inside a neighbour's inflated footprint
# (review 2026-10-08: a 5e-8 edge offset dropped valid floor candidates).
MERGE_TOL = LEN_EPS


def _merge_close(values, tol=MERGE_TOL):
    merged = []
    for v in values:
        if merged and v - merged[-1] <= tol:
            continue
        merged.append(v)
    return merged


def _cluster_levels(values, tol):
    """Group sorted heights closer than ``tol``; (lowest, highest) pairs."""
    clusters = []
    for v in values:
        v = float(v)
        if clusters and v - clusters[-1][0] <= tol:
            clusters[-1][1] = v
        else:
            clusters.append([v, v])
    return [(lo, hi) for lo, hi in clusters]


def maximal_rectangles(free):
    """All maximal all-True axis-aligned rectangles of a boolean grid.

    Yields (row0, row1, col0, col1) inclusive indices. Row-wise histogram
    with a monotone stack; a rectangle is kept when it cannot grow downward.
    Pure-Python lists: the compressed grids are small (< ~150 x 150) and
    per-cell numpy calls would dominate.
    """
    grid = free.tolist() if hasattr(free, "tolist") else free
    rows = len(grid)
    if rows == 0:
        return
    cols = len(grid[0])
    height = [0] * cols
    seen = set()
    for r in range(rows):
        row = grid[r]
        for c in range(cols):
            height[c] = height[c] + 1 if row[c] else 0
        if r + 1 < rows:
            below = grid[r + 1]
            prefix = [0] * (cols + 1)
            run = 0
            for c in range(cols):
                run += 1 if below[c] else 0
                prefix[c + 1] = run
        else:
            prefix = None
        h = height
        left = [0] * cols
        stack = []
        for c in range(cols):
            hc = h[c]
            while stack and h[stack[-1]] >= hc:
                stack.pop()
            left[c] = stack[-1] + 1 if stack else 0
            stack.append(c)
        right = [cols - 1] * cols
        stack = []
        for c in range(cols - 1, -1, -1):
            hc = h[c]
            while stack and h[stack[-1]] >= hc:
                stack.pop()
            right[c] = stack[-1] - 1 if stack else cols - 1
            stack.append(c)
        for c in range(cols):
            hc = h[c]
            if hc == 0:
                continue
            lo, hi = left[c], right[c]
            key = (r - hc + 1, r, lo, hi)
            if key in seen:
                continue
            seen.add(key)
            if prefix is not None and prefix[hi + 1] - prefix[lo] == hi - lo + 1:
                continue  # can grow downward -> not maximal
            yield key
