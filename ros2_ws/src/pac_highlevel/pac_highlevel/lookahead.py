"""Look-ahead search over the next N boxes visible on the conveyor (no learning).

At every stage-4 decision the policy tries each feasible action
(PLACE_CURRENT / BUFFER_CURRENT / RETRIEVE_BUFFER(i)) on a copy of the world
that only holds the current box, the buffer and the next ``horizon`` arrivals,
plays the copy to the end of that window and scores the final state with a
hand-written score (lost pallet volume). The best first action is executed;
the next box restarts the search (receding horizon).

Two search modes:
  ``pilot``  after the first action every decision is taken by ``RulePolicy``
             (cheap: one rollout per first action).
  ``beam``   every decision in the window branches, keeping the ``beam`` best
             states per first action (more search, more time).

Safety is unchanged: every simulated step goes through the same 5-1/5-2
masks as the real world, so the search only chooses among safe options.
Boxes beyond the window are never read.

Speed-ups (see docs/taehyeon/lookahead.md):
  * light clone: lists/dicts copied, immutable data and the candidate backend
    shared (deepcopy took ~0.7 s per state mid-episode),
  * shared backend cache: 5-1/5-2 results are cached per pallet snapshot, so
    siblings with the same pallet (e.g. after BUFFER_CURRENT) reuse them,
  * duplicate states merged in beam mode,
  * time budget per decision: when it runs out, unfinished branches are scored
    as they are; with nothing scored the rule action is returned.
"""

from collections import Counter
from dataclasses import dataclass, field, replace
import time

from pac_common import BoxStatus

import numpy as np

from .actions import from_index, to_index
from .rules import RulePolicy
from .world import PalletizingWorld, box_volume


@dataclass
class LookaheadConfig:
    horizon: int = 3            # boxes after the current one visible on the conveyor
    mode: str = "pilot"         # pilot | beam
    beam: int = 3               # beam mode: states kept per first action and level
    time_budget_s: float = 0.0  # per decision; 0 = no limit
    margin: float = 0.005       # leave the rule action only if the score is better by this
    # leaf score weights (unit: pallet-volume fractions, lower is better)
    w_void: float = 1.0         # empty volume trapped below the open pallet's surface
    w_rough: float = 0.3        # height std of the open pallet / pallet height
    w_buffer: float = 0.01      # per occupied buffer slot at the end of the window
    w_ng: float = 0.5           # per box sent to NG in the window
    w_time: float = 0.0005      # per second of robot time in the window
    # "dead pallet": share of the still-expected volume (buffer + visible boxes
    # + the order list) that has no safe spot on the open pallet; such a pallet
    # is closed soon, so it costs about dead_share * (1 - fill)
    w_dead: float = 1.0
    dead_top_skus: int = 6      # order-list SKUs checked (largest remaining volume first)

    def __post_init__(self):
        if self.horizon < 0 or self.beam < 1:
            raise ValueError("horizon must be >= 0 and beam >= 1")
        if self.mode not in ("pilot", "beam"):
            raise ValueError("mode must be pilot or beam")


def lookahead_config_from_dict(data):
    data = dict(data or {})
    known = LookaheadConfig.__dataclass_fields__
    unknown = set(data) - set(known)
    if unknown:
        raise ValueError(f"unknown lookahead keys: {sorted(unknown)}")
    return LookaheadConfig(**data)


def load_lookahead_config(path=None):
    """``config/taehyeon/lookahead.yaml`` (top key ``lookahead``)."""
    if path is None:
        return LookaheadConfig()
    from pathlib import Path

    import yaml

    data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    return lookahead_config_from_dict(data.get("lookahead", data))


class WindowWorld(PalletizingWorld):
    """A world copy that stops when the visible window is used up (the real
    world would pull box N+1 next, which the search must not know)."""

    window_end = False

    def _advance(self):
        if self.current is None and self.next_arrival >= len(self.arrivals):
            self.window_end = True
            self._options = None
            return
        super()._advance()

    @property
    def leaf(self):
        return self.done or self.window_end or not self.action_mask().any()


def window_clone(world, end):
    """Copy of ``world`` holding arrivals ``[:end]`` only (light, see module doc)."""
    c = object.__new__(WindowWorld)
    c.__dict__.update(world.__dict__)
    c.arrivals = world.arrivals[:end]
    c.window_end = False
    for name in ("buffer", "placed", "closed", "ng", "uncertain"):
        setattr(c, name, list(getattr(world, name)))
    for name in ("capacity_overrides", "placed_truth", "_boxes"):
        setattr(c, name, dict(getattr(world, name)))
    c.counts = Counter(world.counts)
    c.remaining = Counter(world.remaining)
    return c


def _child(node):
    c = object.__new__(WindowWorld)
    c.__dict__.update(node.__dict__)
    for name in ("buffer", "placed", "closed", "ng", "uncertain"):
        setattr(c, name, list(getattr(node, name)))
    for name in ("capacity_overrides", "placed_truth", "_boxes"):
        setattr(c, name, dict(getattr(node, name)))
    c.counts = Counter(node.counts)
    c.remaining = Counter(node.remaining)
    return c


def state_key(node):
    placed = tuple(sorted(
        (p.box_id, round(p.pose.x, 4), round(p.pose.y, 4), round(p.pose.z, 4), round(p.pose.yaw, 3))
        for p in node.placed
    ))
    buffer = tuple(None if e is None else e.arrival.box.box_id for e in node.buffer)
    current = None if node.current is None else node.current.box.box_id
    return (node.pallet_index, placed, buffer, current, node.next_arrival)


def leaf_score(node, root, cfg):
    """Lost volume (pallet fractions) of the window + penalties; lower is better.

    Pallets used at the end = total box volume / pallet volume + waste of the
    closed pallets + what the open pallet will waste, so the waste of pallets
    closed in the window counts fully and the open pallet's future waste is
    estimated by its trapped void and its unevenness.
    """
    waste = sum(1.0 - p.fill for p in node.closed[len(root.closed):])
    score = waste
    if node.placed:
        grid = node.heightmap()
        cell = node.config.features.heightmap_cell_m
        under = float(grid.sum()) * cell * cell
        boxes = sum(box_volume(p.size) for p in node.placed)
        score += cfg.w_void * max(0.0, under - boxes) / node.pallet_volume
        score += cfg.w_rough * float(grid.std()) / node.pallet_size.z
    if cfg.w_dead > 0 and node.placed:
        score += cfg.w_dead * dead_share(node, cfg) * (1.0 - node.fill())
    score += cfg.w_buffer * sum(e is not None for e in node.buffer)
    score += cfg.w_ng * (len(node.ng) - len(root.ng))
    score += cfg.w_time * (node.time_s - root.time_s)
    return score


def _probe_boxes(node, cfg):
    """(box, volume weight) to test: buffer and window boxes as they are,
    plus the order list's largest-volume SKUs (types and counts only)."""
    out, template = [], None
    for e in node.buffer:
        if e is not None:
            out.append((e.arrival.box, box_volume(e.arrival.box.size)))
            template = template or e.arrival.box
    if node.current is not None:
        out.append((node.current.box, box_volume(node.current.box.size)))
        template = template or node.current.box
    for a in node.arrivals[node.next_arrival:]:
        out.append((a.box, box_volume(a.box.size)))
        template = template or a.box
    if template is None and node.placed:
        template = node._box_of(node.placed[0].box_id)
    unseen = []
    for sku, n in node.remaining_by_sku().items():
        spec = node.catalog.get(sku)
        if spec is not None and n > 0:
            unseen.append((n * box_volume(spec.size), spec))
    unseen.sort(key=lambda t: -t[0])
    for vol, spec in unseen[: cfg.dead_top_skus]:
        if template is None:
            break
        probe = replace(template, box_id=f"PROBE-{spec.sku_id}", sku_id=spec.sku_id, size=spec.size,
                        weight_kg=spec.weight_kg, allowed_yaws_rad=spec.allowed_yaws_rad,
                        status=BoxStatus.READY_FOR_PICK)
        out.append((probe, vol))
    return out


def dead_share(node, cfg):
    """Volume share of the expected boxes with no safe spot on the open pallet."""
    probes = _probe_boxes(node, cfg)
    total = sum(v for _, v in probes)
    if total <= 0:
        return 0.0
    state = node.state()
    backend = node.backend()
    seen, dead = {}, 0.0
    for box, vol in probes:
        key = (box.size, round(box.weight_kg, 6), tuple(box.allowed_yaws_rad), box.box_id in node.uncertain)
        if key not in seen:
            seen[key] = not backend.candidate_set(box, state).valid
        dead += vol * seen[key]
    return dead / total


@dataclass
class SearchStats:
    decisions: int = 0
    searched: int = 0           # decisions with more than one feasible action
    changed: int = 0            # decisions where the search overrode the rule
    expansions: int = 0         # simulated steps
    timeouts: int = 0
    seconds: list = field(default_factory=list)

    def summary(self):
        s = np.array(self.seconds) if self.seconds else np.zeros(1)
        return {
            "decisions": self.decisions,
            "searched": self.searched,
            "changed": self.changed,
            "expansions": self.expansions,
            "timeouts": self.timeouts,
            "decision_s_mean": round(float(s.mean()), 3),
            "decision_s_p95": round(float(np.percentile(s, 95)), 3),
            "decision_s_max": round(float(s.max()), 3),
        }


class LookaheadPolicy:
    """``policy(world) -> HighLevelAction`` like ``RulePolicy``."""

    name = "lookahead"

    def __init__(self, hl_config, config=None):
        self.cfg = config or LookaheadConfig()
        self.rule = RulePolicy(hl_config)
        self.stats = SearchStats()

    # -- search ---------------------------------------------------------
    def _deadline_passed(self, deadline):
        return deadline is not None and time.perf_counter() > deadline

    def _step(self, node, index):
        c = _child(node)
        c.step(index)
        self.stats.expansions += 1
        return c

    def _pilot(self, node, root, deadline):
        while not node.leaf:
            if self._deadline_passed(deadline):
                self.stats.timeouts += 1
                break
            node = self._step(node, to_index(self.rule(node)))
        return leaf_score(node, root, self.cfg)

    def _beam(self, node, root, deadline):
        frontier, best = [node], float("inf")
        while frontier:
            children = {}
            for n in frontier:
                if n.leaf:
                    best = min(best, leaf_score(n, root, self.cfg))
                    continue
                for k in np.flatnonzero(n.action_mask()):
                    if self._deadline_passed(deadline):
                        self.stats.timeouts += 1
                        return min([best] + [leaf_score(x, root, self.cfg) for x in frontier])
                    c = self._step(n, int(k))
                    children.setdefault(state_key(c), c)
            ranked = sorted(children.values(), key=lambda c: leaf_score(c, root, self.cfg))
            frontier = ranked[: self.cfg.beam]
        return best

    def scores(self, world):
        """Score per feasible first action (index -> score)."""
        cfg = self.cfg
        start = time.perf_counter()
        deadline = start + cfg.time_budget_s if cfg.time_budget_s > 0 else None
        root = window_clone(world, world.next_arrival + cfg.horizon)
        out = {}
        for k in np.flatnonzero(world.action_mask()):
            child = self._step(root, int(k))
            run = self._pilot if cfg.mode == "pilot" else self._beam
            out[int(k)] = run(child, root, deadline)
        return out

    def __call__(self, world):
        start = time.perf_counter()
        rule_action = self.rule(world)
        self.stats.decisions += 1
        mask = world.action_mask()
        if mask.sum() <= 1:
            self.stats.seconds.append(time.perf_counter() - start)
            return rule_action
        self.stats.searched += 1
        scores = self.scores(world)
        rule_index = to_index(rule_action)
        best = min(scores, key=lambda k: (scores[k], k != rule_index))
        self.stats.seconds.append(time.perf_counter() - start)
        if best != rule_index and scores[best] < scores[rule_index] - self.cfg.margin:
            self.stats.changed += 1
            return from_index(best, world.slots)
        return rule_action


__all__ = [
    "LookaheadConfig",
    "LookaheadPolicy",
    "WindowWorld",
    "dead_share",
    "leaf_score",
    "load_lookahead_config",
    "lookahead_config_from_dict",
    "window_clone",
]
