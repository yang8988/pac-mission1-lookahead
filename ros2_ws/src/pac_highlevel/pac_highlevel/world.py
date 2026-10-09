"""Stage 4 world: one order stream, a buffer and a sequence of pallets.

The world is the single place where stage-4 actions are *executed in
simulation*. It is used identically by the rule policy, the PPO training
environment and the evaluation scripts (flowchart: "학습·실전 동일").

* Every option is checked with stages 5-1/5-2 (``pac_candidates``); an
  action is feasible only if the low-level placer finds a hard-mask-valid
  pose. Hard-mask checks are never skipped.
* PALLET_CLOSE and PARTIAL_REPACK are rules applied automatically when none
  of the learned actions is feasible (TXT 6/8: close only when nothing can
  be placed; repack only when it frees a place for the current box).
* A box that cannot be placed even on an empty pallet goes to NG (stage 2
  Inspection flow) and is counted, never forced.

All states here are SIMULATED; nothing talks to the robot.
"""

import math
from collections import Counter
from dataclasses import dataclass, field, replace

import numpy as np
from pac_candidates import CandidateBackend
from pac_candidates.geometry import rotated_dims
from pac_common import (
    BoxStatus,
    InventoryState,
    PalletState,
    PlacedBox,
    PlanningContext,
    SystemState,
)

from .actions import ActionType, action_count, from_index, to_index
from .config import HighLevelConfig
from .repack import plan_repack
from .value import make_value_provider


@dataclass
class Arrival:
    box: object  # measured BoxState (what the planner sees)
    truth: object = None  # true BoxState (metrics only)
    uncertain: bool = False
    # stage-2 inspection found a dented top: placeable, nothing on top
    # (capacity override 0 N from the moment the box is observed)
    damage_detected: bool = False


@dataclass
class BufferEntry:
    arrival: Arrival
    slot: int
    stored_at: int  # decision counter


@dataclass
class Option:
    """Best low-level placement for one box in the current pallet state."""

    box: object
    candidate: object = None
    valid_count: int = 0
    generated_count: int = 0
    support_ratio: float = 0.0
    z: float = 0.0
    top_after: float = 0.0
    flatness_after: float = 0.0
    future: float = 0.0

    @property
    def feasible(self):
        return self.candidate is not None


@dataclass
class PalletRecord:
    pallet_id: str
    boxes: int
    volume_m3: float
    true_volume_m3: float
    weight_kg: float
    max_top_m: float
    fill: float
    issues: dict = field(default_factory=dict)


MAX_REPACKS_PER_BOX = 2  # guard: repack attempts per waiting current box


def box_volume(size):
    return size.x * size.y * size.z


def dblf_choice(valid):
    def key(c):
        p = c.target_pose
        return (round(p.z, 6), round(p.y, 6), round(p.x, 6), c.candidate_id)

    return min(valid, key=key) if valid else None


class PalletizingWorld:
    def __init__(
        self,
        arrivals,
        pallet_size,
        catalog,
        cand_config,
        config=None,
        *,
        pallet_max_weight_kg=1000.0,
        pallet_id="PALLET",
        value_provider=None,
        placer=None,
        remaining_by_sku=None,
        capacity_overrides_n=None,
        start=True,
    ):
        """``start=False`` builds an idle world (no arrival pulled, no rule
        applied); ``runtime.HighLevelDecider`` fills it from a real snapshot."""
        self.config = config or HighLevelConfig()
        self.cand_config = cand_config
        self.catalog = dict(catalog)
        self.pallet_size = pallet_size
        self.pallet_volume = pallet_size.x * pallet_size.y * pallet_size.z
        self.max_load = float(pallet_max_weight_kg)
        self.base_pallet_id = pallet_id
        self.fixed_pallet_id = None  # real snapshot: keep the State Manager's id
        self.capacity_overrides = dict(capacity_overrides_n or {})
        self.arrivals = list(arrivals)
        self.slots = self.config.buffer.slots
        self.travel = self.config.buffer.travel_times()
        if value_provider is None:
            # built from the config so the policy contract (which records
            # features.value_provider) always matches what is computed
            feat = self.config.features
            kwargs = {"model_path": feat.value_model_path} if feat.value_provider == "donghan" else {}
            value_provider = make_value_provider(feat.value_provider, **kwargs)
        self.value_provider = value_provider
        self.placer = placer or dblf_choice
        if remaining_by_sku is None:
            remaining_by_sku = Counter(a.box.sku_id for a in self.arrivals)
        self.remaining = Counter(remaining_by_sku)
        self.next_arrival = 0
        self.current = None
        self.buffer = [None] * self.slots
        self.placed = []
        self.placed_truth = {}
        self._boxes = {}
        self.uncertain = []
        self.pallet_index = 0
        self.version = 0
        self.decisions = 0
        self.time_s = 0.0
        self.closed = []
        self.ng = []
        self.counts = Counter()
        self.done = False
        self._backend = None
        self._backend_key = None
        self._options = None
        self._empty_ok = {}
        self._advance_reward = 0.0
        self._repacks_for_current = 0
        self._pending_reward = 0.0
        if start:
            self._advance()
            self._pending_reward = self._advance_reward

    # ------------------------------------------------------------------
    # snapshots for stages 5-1/5-2
    # ------------------------------------------------------------------

    @property
    def pallet_id(self):
        if self.fixed_pallet_id is not None:
            return self.fixed_pallet_id
        return f"{self.base_pallet_id}-P{self.pallet_index}"

    def context(self):
        return PlanningContext(
            catalog=self.catalog,
            pallet_max_weight_kg=self.max_load,
            uncertain_box_ids=tuple(self.uncertain),
            buffer_capacity=self.slots,
            capacity_overrides_n=dict(self.capacity_overrides),
        )

    def backend(self):
        key = (len(self.uncertain), len(self.capacity_overrides))
        if self._backend is None or self._backend_key != key:
            self._backend = CandidateBackend(self.context(), self.cand_config)
            self._backend_key = key
        return self._backend

    def state(self, placed=None, extra=()):
        placed = self.placed if placed is None else placed
        tracked = {
            p.box_id: _placed_state(p, self._box_of(p.box_id)) for p in placed
        }
        for entry in self.buffer:
            if entry is not None:
                b = entry.arrival.box
                tracked[b.box_id] = replace(b, status=BoxStatus.BUFFERED)
        if self.current is not None:
            tracked[self.current.box.box_id] = self.current.box
        for b in extra:
            tracked[b.box_id] = b
        return SystemState(
            self.version,
            float(self.decisions),
            PalletState(self.pallet_id, self.pallet_size, tuple(placed)),
            InventoryState(tracked, self.remaining_by_sku()),
        )

    def remaining_by_sku(self):
        """Unseen boxes per SKU, or {} when the order list is not known."""
        if not self.config.features.order_list_known:
            return {}
        return {k: v for k, v in sorted(self.remaining.items()) if v > 0}

    def _box_of(self, box_id):
        return self._boxes[box_id]

    # ------------------------------------------------------------------
    # options and masks
    # ------------------------------------------------------------------

    def heightmap(self, placed=None):
        placed = self.placed if placed is None else placed
        cell = self.config.features.heightmap_cell_m
        nx = max(1, int(round(self.pallet_size.x / cell)))
        ny = max(1, int(round(self.pallet_size.y / cell)))
        grid = np.zeros((nx, ny))
        for p in placed:
            _stamp(grid, p.pose, rotated_dims(p.size, p.pose.yaw), cell)
        return grid

    def evaluate_box(self, box, state=None, grid=None):
        state = self.state() if state is None else state
        backend = self.backend()
        cset = backend.candidate_set(box, state)
        option = Option(box, valid_count=cset.valid_count, generated_count=cset.generated_count)
        chosen = self.choose_candidate(list(cset.valid), box, state)
        if chosen is None:
            return option
        verdict = backend.validate_constraints(box, chosen, state)
        if not verdict.success:  # placer returned something invalid: never execute it
            return option
        pose = chosen.target_pose
        dz = rotated_dims(box.size, pose.yaw)[2]
        grid = self.heightmap(state.pallet.boxes) if grid is None else grid.copy()
        _stamp(grid, pose, rotated_dims(box.size, pose.yaw), self.config.features.heightmap_cell_m)
        H = self.pallet_size.z
        option.candidate = chosen
        option.support_ratio = float(verdict.details["metrics"].get("support_ratio", 1.0))
        option.z = pose.z
        option.top_after = pose.z + dz
        option.flatness_after = float(max(0.0, 1.0 - grid.std() / H))
        if self.value_provider is not None:
            option.future = float(self.value_provider(self, box, chosen, state, option))
        else:
            option.future = option.flatness_after
        return option

    def choose_candidate(self, valid, box, state):
        """Low-level choice among hard-mask-valid candidates (DBLF or planner)."""
        if not valid:
            return None
        if getattr(self.placer, "wants_context", False):
            return self.placer(valid, box, state, self.backend())
        return self.placer(valid)

    def options(self):
        """Option per learned action slot (cached per world version)."""
        if self._options is not None:
            return self._options
        state = self.state()
        grid = self.heightmap()
        current = None
        if self.current is not None:
            current = self.evaluate_box(self.current.box, state, grid)
        buffered = [
            None if e is None else self.evaluate_box(e.arrival.box, state, grid) for e in self.buffer
        ]
        self._options = (current, buffered)
        return self._options

    def free_slot(self):
        free = [i for i, e in enumerate(self.buffer) if e is None]
        return min(free, key=lambda i: (self.travel[i], i)) if free else None

    def action_mask(self):
        mask = np.zeros(action_count(self.slots), dtype=bool)
        if self.done:
            return mask
        current, buffered = self.options()
        if self.current is not None:
            mask[0] = current.feasible
            mask[1] = self.free_slot() is not None
        for i, opt in enumerate(buffered):
            mask[2 + i] = opt is not None and opt.feasible
        return mask

    # ------------------------------------------------------------------
    # execution
    # ------------------------------------------------------------------

    def step(self, action):
        if isinstance(action, (int, np.integer)):
            action = from_index(action, self.slots)
        if self.done:
            raise RuntimeError("episode finished")
        mask = self.action_mask()
        if not mask[to_index(action)]:
            raise ValueError(f"masked action {action.label()}")
        reward = -self.config.reward.occupancy_weight * sum(e is not None for e in self.buffer)
        current, buffered = self.options()
        self.decisions += 1
        self.counts[action.label() if action.type != ActionType.RETRIEVE_BUFFER else "RETRIEVE_BUFFER"] += 1
        if action.type == ActionType.PLACE_CURRENT:
            reward += self._place(self.current, current.candidate)
            self.time_s += self.config.timing.place_time_s
            reward -= self.config.reward.time_weight * self.config.timing.place_time_s
            self.current = None
        elif action.type == ActionType.BUFFER_CURRENT:
            slot = self.free_slot()
            self.buffer[slot] = BufferEntry(self.current, slot, self.decisions)
            self.current = None
            self._invalidate()
            t = self.travel[slot]
            self.time_s += t
            reward -= self.config.reward.time_weight * t
        else:
            entry = self.buffer[action.slot]
            opt = buffered[action.slot]
            self.buffer[action.slot] = None
            reward += self._place(entry.arrival, opt.candidate)
            t = self.config.timing.place_time_s + self.travel[action.slot]
            self.time_s += t
            reward -= self.config.reward.time_weight * t
        reward += self._pending_reward  # rule outcomes before the first decision
        self._pending_reward = 0.0
        self._advance_reward = 0.0
        self._advance()
        reward += self._advance_reward
        return reward

    def _invalidate(self):
        self.version += 1
        self._options = None

    def _place(self, arrival, candidate):
        box = arrival.box
        pose = candidate.target_pose
        self.placed.append(PlacedBox(box.box_id, box.sku_id, box.size, box.weight_kg, pose))
        self._boxes[box.box_id] = box
        self.placed_truth[box.box_id] = arrival.truth or box
        self._invalidate()
        return self.config.reward.volume_weight * box_volume(box.size) / self.pallet_volume

    def placeable_on_empty(self, box):
        """NG test: can this box be placed on an empty pallet at all?"""
        uncertain = box.box_id in self.uncertain
        key = (box.size, round(box.weight_kg, 6), tuple(box.allowed_yaws_rad), uncertain)
        if key not in self._empty_ok:
            state = SystemState(
                0,
                0.0,
                PalletState("EMPTY", self.pallet_size, ()),
                InventoryState({box.box_id: box}, {}),
            )
            ctx = PlanningContext(
                catalog=self.catalog,
                pallet_max_weight_kg=self.max_load,
                uncertain_box_ids=(box.box_id,) if uncertain else (),
            )
            cset = CandidateBackend(ctx, self.cand_config).candidate_set(box, state)
            self._empty_ok[key] = bool(cset.valid)
        return self._empty_ok[key]

    def _pull_arrival(self):
        arrival = self.arrivals[self.next_arrival]
        self.next_arrival += 1
        self.remaining[arrival.box.sku_id] -= 1
        if self.remaining[arrival.box.sku_id] <= 0:
            del self.remaining[arrival.box.sku_id]
        if arrival.uncertain:
            self.uncertain.append(arrival.box.box_id)
        if arrival.damage_detected:
            self.capacity_overrides[arrival.box.box_id] = 0.0
        self.current = arrival
        self._repacks_for_current = 0
        self._invalidate()

    def _advance(self):
        """Apply arrivals and the rule-only actions until a decision is needed."""
        while True:
            if self.current is None and self.next_arrival < len(self.arrivals):
                self._pull_arrival()
            if self.current is None and all(e is None for e in self.buffer):
                self._finish()
                return
            if self.current is not None and not self.placeable_on_empty(self.current.box):
                self._reject(self.current)
                self.current = None
                self._invalidate()
                continue
            mask = self.action_mask()
            if mask.any():
                if not self._close_before_buffer(mask):
                    return
                self._close_pallet()
                continue
            # nothing learned is feasible -> rule actions
            if (
                self.current is not None
                and self.config.repack.enabled
                and self.placed
                and self._repacks_for_current < MAX_REPACKS_PER_BOX
            ):
                self._repacks_for_current += 1
                plan = plan_repack(self)
                if plan is not None:
                    self._apply_repack(plan)
                    continue
            if self.placed:
                self._close_pallet()
                continue
            # empty pallet and still nothing feasible
            if self.current is not None:
                self._reject(self.current)
                self.current = None
            for i, e in enumerate(self.buffer):
                if e is not None:
                    self._reject(e.arrival)
                    self.buffer[i] = None
            self._invalidate()

    def fill(self):
        return sum(box_volume(p.size) for p in self.placed) / self.pallet_volume

    def _close_before_buffer(self, mask):
        """CLOSE rule: a well-filled pallet is closed instead of forcing the
        current box into the buffer when nothing can be placed now."""
        threshold = self.config.close.fill_before_buffer
        if threshold <= 0.0 or self.current is None or not self.placed:
            return False
        if mask[0] or mask[2:].any():
            return False
        return self.fill() >= threshold

    def _reject(self, arrival):
        self.ng.append(arrival.box.box_id)
        self.counts[ActionType.REJECT_NG.value] += 1
        self._advance_reward -= self.config.reward.ng_penalty

    def _apply_repack(self, plan):
        moves, _ = plan
        by_id = {p.box_id: p for p in self.placed}
        for box_id, new_pose in moves:
            old = by_id[box_id]
            self.placed = [p for p in self.placed if p.box_id != box_id]
            self.placed.append(replace(old, pose=new_pose))
        self._invalidate()
        self.counts[ActionType.PARTIAL_REPACK.value] += 1
        self.counts["REPACK_MOVES"] += len(moves)
        t = self.config.timing.repack_move_time_s * len(moves)
        self.time_s += t
        self._advance_reward -= self.config.reward.time_weight * t

    def pallet_record(self):
        vol = sum(box_volume(p.size) for p in self.placed)
        true_vol = sum(box_volume(self.placed_truth[p.box_id].size) for p in self.placed)
        top = max((p.pose.z + rotated_dims(p.size, p.pose.yaw)[2] for p in self.placed), default=0.0)
        model = self.backend().model_for(self.state())
        issues = {k: list(v) for k, v in model.snapshot_issues().items() if v}
        return PalletRecord(
            self.pallet_id,
            len(self.placed),
            vol,
            true_vol,
            sum(p.weight_kg for p in self.placed),
            top,
            vol / self.pallet_volume,
            issues,
        )

    def _close_pallet(self):
        record = self.pallet_record()
        self.closed.append(record)
        self.counts[ActionType.PALLET_CLOSE.value] += 1
        self._advance_reward -= self.config.reward.close_waste_weight * (1.0 - record.fill)
        t = self.config.timing.pallet_change_time_s
        self.time_s += t
        self._advance_reward -= self.config.reward.time_weight * t
        self.pallet_index += 1
        self.placed = []
        self._invalidate()

    def _finish(self):
        self.done = True
        self._options = None

    # ------------------------------------------------------------------
    # reporting
    # ------------------------------------------------------------------

    def summary(self):
        pallets = list(self.closed)
        if self.placed:
            pallets.append(self.pallet_record())
        n_boxes = len(self.arrivals)
        placed = sum(p.boxes for p in pallets)
        closed_fill = [p.fill for p in self.closed]
        total_vol = sum(p.volume_m3 for p in pallets)
        return {
            "boxes": n_boxes,
            "placed": placed,
            "ng": len(self.ng),
            "pallets_used": len(pallets),
            "pallets_closed": len(self.closed),
            "closed_fill_mean": float(np.mean(closed_fill)) if closed_fill else None,
            "fill_per_pallet_used": total_vol / (len(pallets) * self.pallet_volume) if pallets else 0.0,
            # fractional pallets: closed ones + the open one's fill (order continues)
            "pallet_equivalents": len(self.closed) + (pallets[-1].fill if self.placed else 0.0),
            "time_s": self.time_s,
            "decisions": self.decisions,
            "counts": dict(self.counts),
            "safety_issues": sum(sum(len(v) for v in p.issues.values()) for p in pallets),
            "pallets": [p.__dict__ for p in pallets],
        }


def _placed_state(placed, box):
    return replace(box, pose=placed.pose, status=BoxStatus.PLACED)


def _stamp(grid, pose, dims, cell):
    dx, dy, dz = dims
    i0 = int(math.floor(pose.x / cell + 1e-6))
    j0 = int(math.floor(pose.y / cell + 1e-6))
    i1 = int(math.ceil((pose.x + dx) / cell - 1e-6))
    j1 = int(math.ceil((pose.y + dy) / cell - 1e-6))
    i0, j0 = max(0, i0), max(0, j0)
    region = grid[i0:i1, j0:j1]
    np.maximum(region, pose.z + dz, out=region)

