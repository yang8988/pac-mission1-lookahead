"""Split a vertical force among supporting contacts.

``area``  : share proportional to contact area (pressure-uniform model).
``lever`` : minimum-weighted-norm forces that also satisfy moment balance
            about the force's point of application (statically indeterminate
            problem solved as an equality-constrained least squares, weights =
            contact areas, with an active set that removes tensile forces).
            Supporters close to the resultant carry more, so a box resting
            mostly over one neighbour loads that neighbour realistically
            instead of spreading the weight by area.

Both models conserve the total force. When moment balance is impossible with
compressive forces (the resultant lies outside the contact centroids' hull),
``lever`` falls back to the nearest feasible compressive split and finally to
``area``. Stability itself is never decided here; that is the LBCP check.
"""

import math

import numpy as np


def area_shares(areas):
    total = float(sum(areas))
    if total <= 0.0:
        raise ValueError("Contacts need positive area")
    return [a / total for a in areas]


def _constraints(pts, point):
    """Force + moment equations that are satisfiable for these centroids.

    Moments are taken about the centroids' mean along the principal axes the
    centroids actually span: 2 axes in general, 1 if they are collinear (the
    point is then projected onto that line, i.e. moment balance across the
    line is impossible and must not distort the split along it), 0 if they
    coincide. The total-force equation is always kept exactly.
    """
    c = pts.mean(axis=0)
    q = pts - c
    _, sv, vt = np.linalg.svd(q, full_matrices=False)
    rank = int(np.sum(sv > 1e-9 * max(1.0, float(sv[0]) if sv.size else 1.0)))
    basis = vt[:rank]
    a = np.vstack([np.ones(len(pts)), (q @ basis.T).T])
    b = np.concatenate([[1.0], (np.asarray(point, dtype=float) - c) @ basis.T])
    return a, b


def lever_shares(areas, centroids, point):
    """Return fractional shares (sum 1, all >= 0) for each contact."""
    n = len(areas)
    if n == 0:
        raise ValueError("No contacts")
    if n == 1:
        return [1.0]
    weights = np.asarray(areas, dtype=float)
    if np.any(weights <= 0.0):
        raise ValueError("Contacts need positive area")
    pts = np.asarray(centroids, dtype=float)
    active = np.ones(n, dtype=bool)
    for _ in range(n):
        idx = np.flatnonzero(active)
        a, target = _constraints(pts[idx], point)
        d = weights[idx]
        # f = D A^T (A D A^T)^+ b   (minimises sum f_i^2 / a_i s.t. A f = b)
        gram = (a * d) @ a.T
        f_active = d * (a.T @ (np.linalg.pinv(gram, rcond=1e-10) @ target))
        if np.all(f_active >= -1e-12):
            shares = np.zeros(n)
            shares[idx] = np.clip(f_active, 0.0, None)
            total = shares.sum()
            if total <= 1e-12:
                break
            return list(shares / total)
        # Drop the most tensile contact and retry.
        worst = idx[int(np.argmin(f_active))]
        active[worst] = False
        if not active.any():
            break
    return area_shares(areas)


def split_force(model, areas, centroids, point):
    if model == "area":
        return area_shares(areas)
    if model == "lever":
        return lever_shares(areas, centroids, point)
    raise ValueError(f"Unknown load share model: {model}")


LOAD_REL_TOL = 1e-9
LOAD_ABS_TOL_N = 1e-9


def overloaded(load_n, capacity_n):
    """Single capacity test shared by the hard mask and snapshot re-checks."""
    return load_n > capacity_n * (1.0 + LOAD_REL_TOL) + LOAD_ABS_TOL_N


def load_ratio(load_n, capacity_n):
    if capacity_n > 0:
        return load_n / capacity_n
    return 0.0 if load_n <= LOAD_ABS_TOL_N else math.inf


def mckee_capacity_n(dx, dy, ect_n_per_m, board_thickness_m, safety_factor):
    """Allowable static top load of a corrugated carton (McKee BCT / SF)."""
    perimeter = 2.0 * (dx + dy)
    bct = 5.87 * ect_n_per_m * float(np.sqrt(board_thickness_m * perimeter))
    return bct / safety_factor
