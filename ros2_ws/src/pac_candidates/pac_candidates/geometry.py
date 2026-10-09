"""Exact axis-aligned and convex-polygon geometry (SI units, pallet frame).

Pose convention (team contract, docs/integration.md of stage 5-3~6):
``PlacementCandidate.target_pose`` / ``PlacedBox.pose`` is the MINIMUM x/y/z
corner of the box AABB after applying ``yaw``. ``z = 0`` is the pallet loading
surface. Only upright, axis-aligned yaws (multiples of pi/2) are supported.
"""

from dataclasses import dataclass
import math

HALF_PI = math.pi / 2.0
YAW_TOL = 1e-6
LEN_EPS = 1e-9
# Shared numeric tolerance for stability margins (placement check and the
# snapshot re-check must agree exactly on borderline cases).
STABILITY_EPS = 1e-9


def quarter_turns(yaw):
    """Return the number of quarter turns of ``yaw`` or raise ValueError."""
    if not math.isfinite(yaw):
        raise ValueError("yaw must be finite")
    quarter = round(yaw / HALF_PI)
    if abs(yaw - quarter * HALF_PI) > YAW_TOL:
        raise ValueError("Only axis-aligned yaws (k * pi/2) are supported")
    return quarter


def rotated_dims(size, yaw):
    """(dx, dy, dz) of the AABB of an upright box rotated by ``yaw``."""
    if quarter_turns(yaw) % 2:
        return size.y, size.x, size.z
    return size.x, size.y, size.z


def same_yaw(a, b):
    return abs(math.remainder(a - b, 2.0 * math.pi)) <= YAW_TOL


@dataclass(frozen=True)
class Rect:
    """Closed axis-aligned rectangle [x0, x1] x [y0, y1]."""

    x0: float
    y0: float
    x1: float
    y1: float

    @property
    def width(self):
        return self.x1 - self.x0

    @property
    def depth(self):
        return self.y1 - self.y0

    @property
    def area(self):
        return max(0.0, self.width) * max(0.0, self.depth)

    @property
    def center(self):
        return (0.5 * (self.x0 + self.x1), 0.5 * (self.y0 + self.y1))

    def expand(self, margin):
        return Rect(
            self.x0 - margin,
            self.y0 - margin,
            self.x1 + margin,
            self.y1 + margin,
        )

    def intersection(self, other):
        x0 = max(self.x0, other.x0)
        y0 = max(self.y0, other.y0)
        x1 = min(self.x1, other.x1)
        y1 = min(self.y1, other.y1)
        if x1 - x0 <= LEN_EPS or y1 - y0 <= LEN_EPS:
            return None
        return Rect(x0, y0, x1, y1)

    def overlaps(self, other, tol=LEN_EPS):
        """Open-interior overlap (touching edges do not overlap)."""
        return (
            min(self.x1, other.x1) - max(self.x0, other.x0) > tol
            and min(self.y1, other.y1) - max(self.y0, other.y0) > tol
        )

    def contains_rect(self, other, tol=LEN_EPS):
        return (
            other.x0 >= self.x0 - tol
            and other.y0 >= self.y0 - tol
            and other.x1 <= self.x1 + tol
            and other.y1 <= self.y1 + tol
        )

    def corners(self):
        return (
            (self.x0, self.y0),
            (self.x1, self.y0),
            (self.x1, self.y1),
            (self.x0, self.y1),
        )


def footprint(x, y, dx, dy):
    return Rect(x, y, x + dx, y + dy)


# --------------------------------------------------------------------------
# Convex polygons: tuples of (x, y) vertices in counter-clockwise order.
# --------------------------------------------------------------------------


def _cross(o, a, b):
    return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])


def convex_hull(points):
    """Andrew monotone chain; returns CCW hull without repeated endpoint."""
    pts = sorted(set((float(p[0]), float(p[1])) for p in points))
    if len(pts) <= 2:
        return tuple(pts)
    lower = []
    for p in pts:
        while len(lower) >= 2 and _cross(lower[-2], lower[-1], p) <= 1e-15:
            lower.pop()
        lower.append(p)
    upper = []
    for p in reversed(pts):
        while len(upper) >= 2 and _cross(upper[-2], upper[-1], p) <= 1e-15:
            upper.pop()
        upper.append(p)
    return tuple(lower[:-1] + upper[:-1])


def polygon_area(poly):
    if len(poly) < 3:
        return 0.0
    total = 0.0
    for i, a in enumerate(poly):
        b = poly[(i + 1) % len(poly)]
        total += a[0] * b[1] - b[0] * a[1]
    return 0.5 * total


def polygon_centroid(poly):
    """Area centroid of a simple polygon (vertex mean if degenerate)."""
    a = polygon_area(poly)
    if abs(a) <= 1e-15:
        n = max(1, len(poly))
        return (sum(q[0] for q in poly) / n, sum(q[1] for q in poly) / n)
    cx = cy = 0.0
    for i, p in enumerate(poly):
        q = poly[(i + 1) % len(poly)]
        cross = p[0] * q[1] - q[0] * p[1]
        cx += (p[0] + q[0]) * cross
        cy += (p[1] + q[1]) * cross
    return (cx / (6.0 * a), cy / (6.0 * a))


def rect_polygon(rect):
    return rect.corners()


def clip_polygon_to_rect(poly, rect):
    """Sutherland-Hodgman clip of a convex polygon by an AABB."""
    output = list(poly)
    edges = (
        (lambda p: p[0] >= rect.x0, lambda a, b: _at_x(a, b, rect.x0)),
        (lambda p: p[0] <= rect.x1, lambda a, b: _at_x(a, b, rect.x1)),
        (lambda p: p[1] >= rect.y0, lambda a, b: _at_y(a, b, rect.y0)),
        (lambda p: p[1] <= rect.y1, lambda a, b: _at_y(a, b, rect.y1)),
    )
    for inside, cut in edges:
        if not output:
            break
        source, output = output, []
        for i, current in enumerate(source):
            previous = source[i - 1]
            if inside(current):
                if not inside(previous):
                    output.append(cut(previous, current))
                output.append(current)
            elif inside(previous):
                output.append(cut(previous, current))
    return convex_hull(output) if len(output) >= 3 else tuple(output)


def _at_x(a, b, x):
    t = (x - a[0]) / (b[0] - a[0])
    return (x, a[1] + t * (b[1] - a[1]))


def _at_y(a, b, y):
    t = (y - a[1]) / (b[1] - a[1])
    return (a[0] + t * (b[0] - a[0]), y)


def polygon_bounds(poly):
    xs = [p[0] for p in poly]
    ys = [p[1] for p in poly]
    return Rect(min(xs), min(ys), max(xs), max(ys))


def shrink_rect_polygon(rect, margin):
    """Rectangle shrunk by ``margin`` (empty tuple if it vanishes)."""
    inner = Rect(
        rect.x0 + margin, rect.y0 + margin, rect.x1 - margin, rect.y1 - margin
    )
    if inner.width <= LEN_EPS or inner.depth <= LEN_EPS:
        return ()
    return inner.corners()


def inside_margin(poly, point, delta_x=0.0, delta_y=0.0):
    """Worst-case signed distance of an uncertain point to a convex polygon.

    The point may lie anywhere in ``[x +- delta_x] x [y +- delta_y]``.
    Positive: every possible point is strictly inside by that distance.
    Negative: some possible point is outside. Degenerate polygons (< 3
    vertices or ~zero area) return ``-inf`` because a line/point support
    cannot carry an uncertain resultant.
    """
    if len(poly) < 3 or polygon_area(poly) <= 1e-12:
        return -math.inf
    best = math.inf
    n = len(poly)
    for i in range(n):
        a = poly[i]
        b = poly[(i + 1) % n]
        ex, ey = b[0] - a[0], b[1] - a[1]
        length = math.hypot(ex, ey)
        if length <= 1e-15:
            continue
        # Inward normal for a CCW polygon.
        nx, ny = -ey / length, ex / length
        distance = (point[0] - a[0]) * nx + (point[1] - a[1]) * ny
        distance -= delta_x * abs(nx) + delta_y * abs(ny)
        best = min(best, distance)
    return best
