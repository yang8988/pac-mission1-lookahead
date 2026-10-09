"""Stage 5-3 placement rule "flat layers" (alternative to DBLF, no learning).

DBLF takes the lowest, then deepest, then left-most safe candidate. With
boxes of mixed heights that leaves a jagged top surface, and a box on the
next level then rarely finds the 70 % support it needs: in the diagnosis,
pallets were closed with ~75 % of the floor covered but only ~27 % filled,
and "support ratio too low" was by far the most frequent 5-2 rejection.

``LayerPlacer`` scores every 5-2-valid candidate (lower is better). Every
term is dimensionless, so the same weights hold for any pallet, height limit
or box catalogue:

    score = w_z      * z / H                          (prefer low spots; H = stack height limit)
          + w_level  * share of the ring around the footprint (cells above the
                       box base) whose height is NOT level with the box top
          + w_step   * mean step / h_ref               (step size, capped at h_ref)
          + w_flush  * (1 - share of the ring that is a pallet edge or level
                       with the box top)
          + w_support* (1 - support ratio)
          + w_corner * (x + y) / (X + Y)                (DBLF-like tie-break)

"Level" uses the same height tolerance as the 5-2 support test (a supporter
counts only if its top is within that tolerance of the box base), because ANY
larger step means a box placed across it later rests on the higher part only:
the step size does not matter for that, only whether there is a step. The step
size still matters for whether another box can fill it, so it enters with a
smaller weight, relative to h_ref = median SKU height of the catalogue
(auto-calibrated; set ``ref_height_m`` to override).
Every candidate passed 5-1/5-2 already; this only orders safe options.
"""

from dataclasses import dataclass

import numpy as np

from pac_candidates.geometry import rotated_dims


@dataclass
class LayerConfig:
    cell_m: float = 0.02
    w_z: float = 1.0
    w_level: float = 0.3
    w_step: float = 0.1
    w_flush: float = 0.05
    w_support: float = 0.1
    w_corner: float = 0.01
    level_tol_m: float = 0.0    # 0 = the 5-2 support height tolerance
    ref_height_m: float = 0.0   # 0 = median SKU height of the catalogue


def _stamp(grid, x, y, top, dx, dy, cell):
    i0, j0 = int(round(x / cell)), int(round(y / cell))
    i1, j1 = int(round((x + dx) / cell)), int(round((y + dy) / cell))
    region = grid[max(0, i0):max(0, i1), max(0, j0):max(0, j1)]
    np.maximum(region, top, out=region)


def heightmap(pallet, cell):
    nx = max(1, int(round(pallet.size.x / cell)))
    ny = max(1, int(round(pallet.size.y / cell)))
    grid = np.zeros((nx, ny))
    for b in pallet.boxes:
        dx, dy, dz = rotated_dims(b.size, b.pose.yaw)
        _stamp(grid, b.pose.x, b.pose.y, b.pose.z + dz, dx, dy, cell)
    return grid


def ring_metrics(grid, x, y, z, dx, dy, top, cell, tol, ref):
    """(non-level share, mean step / ref, flush share) on the 1-cell ring
    around a footprint; cells at or below the box base are open space."""
    nx, ny = grid.shape
    i0, j0 = int(round(x / cell)), int(round(y / cell))
    i1, j1 = int(round((x + dx) / cell)), int(round((y + dy) / cell))
    sides = (
        (grid[i0:i1, j0 - 1] if j0 - 1 >= 0 else None, i1 - i0),
        (grid[i0:i1, j1] if j1 < ny else None, i1 - i0),
        (grid[i0 - 1, j0:j1] if i0 - 1 >= 0 else None, j1 - j0),
        (grid[i1, j0:j1] if i1 < nx else None, j1 - j0),
    )
    total = flush = 0
    steps = []
    for line, n in sides:
        total += n
        if line is None:  # pallet edge
            flush += n
            continue
        h = np.asarray(line)
        flush += int(np.count_nonzero(np.abs(h - top) <= tol))
        above = h[h > z + tol]
        if above.size:
            steps.append(np.abs(above - top))
    if steps:
        d = np.concatenate(steps)
        non_level = float(np.count_nonzero(d > tol)) / d.size
        step = float(np.minimum(d / ref, 1.0).mean())
    else:
        non_level = step = 0.0
    return non_level, step, (flush / total if total else 1.0)


class LayerPlacer:
    """``placer(valid, box, state, backend) -> candidate`` (world ``placer``)."""

    wants_context = True
    name = "layer"

    def __init__(self, config=None):
        self.cfg = config or LayerConfig()

    def _reference(self, backend):
        """(level tolerance, reference height) for this backend (cached)."""
        key = id(backend)
        if getattr(self, "_ref_key", None) != key:
            cfg = self.cfg
            tol = cfg.level_tol_m or backend.config.uncertainty.height_tolerance_m
            ref = cfg.ref_height_m
            if ref <= 0:
                catalog = getattr(backend.context, "catalog", None) or {}
                heights = sorted(s.size.z for s in catalog.values())
                ref = heights[len(heights) // 2] if heights else 0.2
            self._ref_key, self._ref = key, (tol, ref)
        return self._ref

    def scores(self, valid, box, state, backend):
        cfg = self.cfg
        pallet = state.pallet
        grid = heightmap(pallet, cfg.cell_m)
        X, Y, H = pallet.size.x, pallet.size.y, pallet.size.z
        tol, ref = self._reference(backend)
        out = []
        for c in valid:
            p = c.target_pose
            dx, dy, dz = rotated_dims(box.size, p.yaw)
            top = p.z + dz
            non_level, step, flush = ring_metrics(grid, p.x, p.y, p.z, dx, dy, top, cfg.cell_m, tol, ref)
            verdict = backend.validate_constraints(box, c, state)
            support = float(verdict.details.get("metrics", {}).get("support_ratio", 1.0))
            score = (cfg.w_z * p.z / H + cfg.w_level * non_level + cfg.w_step * step
                     + cfg.w_flush * (1.0 - flush) + cfg.w_support * (1.0 - support)
                     + cfg.w_corner * (p.x + p.y) / (X + Y))
            out.append((score, round(p.z, 6), round(p.y, 6), round(p.x, 6), c.candidate_id, c))
        return out

    def __call__(self, valid, box, state, backend):
        if not valid:
            return None
        return min(self.scores(valid, box, state, backend), key=lambda t: t[:5])[-1]


def make_placer(name="layer", config=None):
    """World ``placer`` for a 5-3 rule name (``None`` = the world's DBLF)."""
    if name == "dblf":
        return None
    if name == "layer":
        return LayerPlacer(config)
    raise ValueError(f"unknown placer {name!r}")


def layer_config_from_dict(data):
    data = dict(data or {})
    unknown = set(data) - set(LayerConfig.__dataclass_fields__)
    if unknown:
        raise ValueError(f"unknown layer keys: {sorted(unknown)}")
    return LayerConfig(**data)


__all__ = ["LayerConfig", "LayerPlacer", "heightmap", "layer_config_from_dict", "make_placer", "ring_metrics"]
