"""Stage 5-3 placement rule "flat layers" (alternative to DBLF, no learning).

DBLF takes the lowest, then deepest, then left-most safe candidate. With
boxes of mixed heights that leaves a jagged top surface, and a box on the
next level then rarely finds the 70 % support it needs: in the diagnosis,
pallets were closed with ~75 % of the floor covered but only ~27 % filled,
and "support ratio too low" was by far the most frequent 5-2 rejection.

``LayerPlacer`` scores every 5-2-valid candidate (lower is better):

    score = w_z      * z                              (still prefer low spots)
          + w_step   * mean |neighbour height - box top|  over the ring of cells
                       around the footprint that are above the box base
                       (steps the box creates; capped per cell at step_cap_m)
          + w_flush  * (1 - share of the ring that is a pallet edge or a
                       neighbour flush with the box top)
          + w_support* (1 - support ratio)
          + w_corner * (x + y) / (X + Y)                (DBLF-like tie-break)

so a box goes where its top continues a level surface (next to boxes of the
same height, against the pallet edge) unless that costs much height.
Every candidate passed 5-1/5-2 already; this only orders safe options.
"""

from dataclasses import dataclass

import numpy as np

from pac_candidates.geometry import rotated_dims


@dataclass
class LayerConfig:
    cell_m: float = 0.02
    w_z: float = 1.0
    w_step: float = 1.0
    w_flush: float = 0.05
    w_support: float = 0.1
    w_corner: float = 0.01
    flush_tol_m: float = 0.01
    step_cap_m: float = 0.3


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


def ring_metrics(grid, x, y, z, dx, dy, top, cfg):
    """(mean capped step, flush share) on the 1-cell ring around a footprint."""
    cell = cfg.cell_m
    nx, ny = grid.shape
    i0, j0 = int(round(x / cell)), int(round(y / cell))
    i1, j1 = int(round((x + dx) / cell)), int(round((y + dy) / cell))
    sides = []  # (cells, edge_count)
    for line, inside in (
        (grid[i0:i1, j0 - 1] if j0 - 1 >= 0 else None, j0 - 1 >= 0),
        (grid[i0:i1, j1] if j1 < ny else None, j1 < ny),
        (grid[i0 - 1, j0:j1] if i0 - 1 >= 0 else None, i0 - 1 >= 0),
        (grid[i1, j0:j1] if i1 < nx else None, i1 < nx),
    ):
        sides.append(line if inside else None)
    total = flush = 0
    steps = []
    lengths = (i1 - i0, i1 - i0, j1 - j0, j1 - j0)
    for line, n in zip(sides, lengths):
        total += n
        if line is None:  # pallet edge
            flush += n
            continue
        h = np.asarray(line)
        flush += int(np.count_nonzero(np.abs(h - top) <= cfg.flush_tol_m))
        above = h[h > z + cfg.flush_tol_m]
        if above.size:
            steps.append(np.minimum(np.abs(above - top), cfg.step_cap_m))
    step = float(np.concatenate(steps).mean()) if steps else 0.0
    return step, (flush / total if total else 1.0)


class LayerPlacer:
    """``placer(valid, box, state, backend) -> candidate`` (world ``placer``)."""

    wants_context = True
    name = "layer"

    def __init__(self, config=None):
        self.cfg = config or LayerConfig()

    def scores(self, valid, box, state, backend):
        cfg = self.cfg
        pallet = state.pallet
        grid = heightmap(pallet, cfg.cell_m)
        X, Y = pallet.size.x, pallet.size.y
        out = []
        for c in valid:
            p = c.target_pose
            dx, dy, dz = rotated_dims(box.size, p.yaw)
            top = p.z + dz
            step, flush = ring_metrics(grid, p.x, p.y, p.z, dx, dy, top, cfg)
            verdict = backend.validate_constraints(box, c, state)
            support = float(verdict.details.get("metrics", {}).get("support_ratio", 1.0))
            score = (cfg.w_z * p.z + cfg.w_step * step + cfg.w_flush * (1.0 - flush)
                     + cfg.w_support * (1.0 - support) + cfg.w_corner * (p.x + p.y) / (X + Y))
            out.append((score, round(p.z, 6), round(p.y, 6), round(p.x, 6), c.candidate_id, c))
        return out

    def __call__(self, valid, box, state, backend):
        if not valid:
            return None
        return min(self.scores(valid, box, state, backend), key=lambda t: t[:5])[-1]


def layer_config_from_dict(data):
    data = dict(data or {})
    unknown = set(data) - set(LayerConfig.__dataclass_fields__)
    if unknown:
        raise ValueError(f"unknown layer keys: {sorted(unknown)}")
    return LayerConfig(**data)


__all__ = ["LayerConfig", "LayerPlacer", "heightmap", "layer_config_from_dict", "ring_metrics"]
