"""Stage-4 entry point on a real snapshot: ``SystemState`` -> one decision.

```python
decider = HighLevelDecider(
    context,                                   # PlanningContext of this cycle
    policy=load_policy("numpy", "models/highlevel_ppo.json"),
)
decision = decider.decide(state, current_box_id="B0123",
                          buffer_slots={0: "B0101", 2: "B0117"})
if decision.requires_low_level:                # PLACE_CURRENT / RETRIEVE_BUFFER
    result = planner.plan(decision.box, state)  # stage 5 (donghan) picks the pose
```

The decision is computed on a private copy of the snapshot with the same
world logic, masks, features and rules used for training (flowchart: "학습·
실전 동일"). Nothing is executed and the inputs are never modified; the
State Manager applies the outcome and sends a new snapshot (new version).

Priority (identical to the training world):
1. no current and no buffered box               -> ``action=None`` (NO_BOX)
2. current box fits no empty pallet              -> REJECT_NG (stage-2 NG flow)
3. some learned action feasible:
   pallet filled >= close.fill_before_buffer and neither the current nor a
   buffered box fits                             -> PALLET_CLOSE (rule)
   otherwise                                     -> policy picks among the mask
4. nothing feasible: PARTIAL_REPACK if a plan exists, else PALLET_CLOSE;
   on an empty pallet the unplaceable boxes go to NG.
"""

from dataclasses import dataclass, field

from pac_common import BoxState, BoxStatus

from .actions import ActionType, HighLevelAction, action_count
from .config import HighLevelConfig
from .features import feature_names, observe
from .repack import plan_repack
from .rules import GreedyPolicy, RulePolicy
from .trainer import policy_contract
from .world import MAX_REPACKS_PER_BOX, Arrival, BufferEntry, PalletizingWorld

PENDING_STATUSES = (BoxStatus.MEASURED, BoxStatus.ON_CONVEYOR, BoxStatus.READY_FOR_PICK)


@dataclass(frozen=True)
class HighLevelDecision:
    """One stage-4 decision for snapshot ``state_version``.

    ``box``: the box the action concerns (the box handed to stage 5 for
    PLACE_CURRENT / RETRIEVE_BUFFER). ``candidate``: the 5-1/5-2-valid pose
    that made the action feasible; advisory only, stage 5 chooses the final
    pose. ``repack_moves``: ((box_id, Pose3D), ...) to execute in order for
    PARTIAL_REPACK. ``mask`` and ``probabilities`` follow the policy index
    layout (0 PLACE_CURRENT, 1 BUFFER_CURRENT, 2+i RETRIEVE_BUFFER(i)).
    """

    action: HighLevelAction | None
    state_version: int
    box: BoxState | None = None
    slot: int | None = None
    candidate: object = None
    repack_moves: tuple = ()
    mask: tuple = ()
    probabilities: tuple | None = None
    decided_by: str = ""
    reason: str = ""
    diagnostics: dict = field(default_factory=dict)

    @property
    def requires_low_level(self):
        return self.action is not None and self.action.type in (
            ActionType.PLACE_CURRENT,
            ActionType.RETRIEVE_BUFFER,
        )


class LoadedPolicy:
    """Policy wrapper: ``choose(world)`` and optional ``probs(world)``."""

    def __init__(self, name, choose, probs=None):
        self.name = name
        self.choose = choose
        self.probs = probs


def load_policy(kind="numpy", path=None, *, config=None, candidate_config_name="candidates.yaml"):
    """``rule`` | ``greedy`` | ``numpy`` (MaskablePPO json) | ``sb3`` (zip).

    Learned policies are checked against the feature layout and the
    contract (buffer slots, value provider, placer, candidate config).
    """
    config = config or HighLevelConfig()
    if kind == "rule":
        return LoadedPolicy("rule", RulePolicy(config))
    if kind == "greedy":
        return LoadedPolicy("greedy", GreedyPolicy())
    contract = policy_contract(config, candidate_config_name)
    if kind == "numpy":
        from .actions import from_index
        from .ppo import MaskablePPO

        agent = MaskablePPO.load(path, feature_names=feature_names(config.buffer.slots), contract=contract)

        def choose(world):
            a, _, _ = agent.act(observe(world), world.action_mask(), deterministic=True)
            return from_index(a, world.slots)

        def probs(world):
            return agent.probs(observe(world), world.action_mask())[0]

        return LoadedPolicy("maskable_ppo_numpy", choose, probs)
    if kind == "sb3":
        from . import sb3

        model = sb3.load(path, slots=config.buffer.slots, contract=contract)
        return LoadedPolicy("maskable_ppo_sb3", sb3.chooser(model))
    raise ValueError(f"unknown policy kind {kind}")


class HighLevelDecider:
    def __init__(self, context, cand_config=None, config=None, *, policy=None, value_provider=None,
                 placer=None):
        from pac_candidates import CandidateConfig

        self.context = context
        self.cand_config = cand_config or CandidateConfig()
        self.config = config or HighLevelConfig()
        self.policy = policy or load_policy("rule", config=self.config)
        self.value_provider = value_provider
        self.placer = placer

    # ------------------------------------------------------------------
    def snapshot_world(self, state, current_box_id=None, buffer_slots=None, buffer_age=None):
        """Private world built from a real snapshot (inputs untouched)."""
        tracked = dict(state.inventory.tracked_boxes)
        current_id = self._current_id(state, tracked, current_box_id)
        slots = self._slots(state, tracked, buffer_slots, current_id)
        ctx = self.context
        world = PalletizingWorld(
            [],
            state.pallet.size,
            ctx.catalog,
            self.cand_config,
            self.config,
            pallet_max_weight_kg=ctx.pallet_max_weight_kg,
            pallet_id=state.pallet.pallet_id,
            value_provider=self.value_provider,
            placer=self.placer,
            remaining_by_sku=dict(state.inventory.remaining_by_sku),
            capacity_overrides_n=dict(ctx.capacity_overrides_n),
            start=False,
        )
        world.fixed_pallet_id = state.pallet.pallet_id
        world.version = state.state_version
        world.uncertain = list(ctx.uncertain_box_ids)
        world.placed = list(state.pallet.boxes)
        for p in world.placed:
            box = tracked.get(p.box_id) or self._box_from_placed(p)
            world._boxes[p.box_id] = box
            world.placed_truth[p.box_id] = box
        ages = dict(buffer_age or {})
        for slot, box_id in slots.items():
            box = tracked[box_id]
            arrival = Arrival(box, None, box_id in ctx.uncertain_box_ids)
            world.buffer[slot] = BufferEntry(arrival, slot, -int(ages.get(box_id, 0)))
        if current_id is not None:
            box = tracked[current_id]
            world.current = Arrival(box, None, current_id in ctx.uncertain_box_ids)
        return world

    def decide(self, state, current_box_id=None, buffer_slots=None, buffer_age=None, repack_attempts=0):
        world = self.snapshot_world(state, current_box_id, buffer_slots, buffer_age)
        version = state.state_version
        n_actions = action_count(world.slots)
        diag = {
            "policy": self.policy.name,
            "contract": policy_contract(self.config),
            "buffer_slots": {i: e.arrival.box.box_id for i, e in enumerate(world.buffer) if e is not None},
            "pallet_fill": world.fill(),
            "robot_validation": "NOT_CHECKED",
        }
        if world.current is None and all(e is None for e in world.buffer):
            return HighLevelDecision(None, version, mask=(False,) * n_actions, decided_by="rule",
                                     reason="NO_BOX", diagnostics=diag)
        current = world.current
        if current is not None and not world.placeable_on_empty(current.box):
            return HighLevelDecision(HighLevelAction(ActionType.REJECT_NG), version, box=current.box,
                                     mask=(False,) * n_actions, decided_by="rule:ng",
                                     reason="FITS_NO_EMPTY_PALLET", diagnostics=diag)
        mask = world.action_mask()
        options_current, options_buffered = world.options()
        if mask.any():
            if world._close_before_buffer(mask):
                return HighLevelDecision(HighLevelAction(ActionType.PALLET_CLOSE), version,
                                         mask=tuple(bool(m) for m in mask), decided_by="rule:close",
                                         reason="FILL_BEFORE_BUFFER", diagnostics=diag)
            action = self.policy.choose(world)
            probs = None
            if self.policy.probs is not None:
                probs = tuple(float(v) for v in self.policy.probs(world))
            box, slot, cand = None, None, None
            if action.type == ActionType.PLACE_CURRENT:
                box, cand = current.box, options_current.candidate
            elif action.type == ActionType.BUFFER_CURRENT:
                box, slot = current.box, world.free_slot()
            else:
                slot = action.slot
                entry = world.buffer[slot]
                box = world.state().inventory.tracked_boxes[entry.arrival.box.box_id]  # status BUFFERED
                cand = options_buffered[slot].candidate
            return HighLevelDecision(action, version, box=box, slot=slot, candidate=cand,
                                     mask=tuple(bool(m) for m in mask), probabilities=probs,
                                     decided_by="policy:" + self.policy.name, reason="POLICY",
                                     diagnostics=diag)
        # nothing learned is feasible -> rule actions
        if (current is not None and self.config.repack.enabled and world.placed
                and repack_attempts < MAX_REPACKS_PER_BOX):
            plan = plan_repack(world)
            if plan is not None:
                moves, cand = plan
                return HighLevelDecision(HighLevelAction(ActionType.PARTIAL_REPACK), version,
                                         box=current.box, candidate=cand, repack_moves=tuple(moves),
                                         mask=tuple(bool(m) for m in mask), decided_by="rule:repack",
                                         reason="NO_FEASIBLE_ACTION", diagnostics=diag)
        if world.placed:
            return HighLevelDecision(HighLevelAction(ActionType.PALLET_CLOSE), version,
                                     mask=tuple(bool(m) for m in mask), decided_by="rule:close",
                                     reason="NO_FEASIBLE_ACTION", diagnostics=diag)
        # empty pallet and still nothing fits
        stuck = current.box if current is not None else next(
            e.arrival.box for e in world.buffer if e is not None
        )
        return HighLevelDecision(HighLevelAction(ActionType.REJECT_NG), version, box=stuck,
                                 mask=tuple(bool(m) for m in mask), decided_by="rule:ng",
                                 reason="FITS_NO_EMPTY_PALLET", diagnostics=diag)

    # ------------------------------------------------------------------
    @staticmethod
    def _current_id(state, tracked, current_box_id):
        on_pallet = {b.box_id for b in state.pallet.boxes}
        if current_box_id is not None:
            box = tracked.get(current_box_id)
            if box is None:
                raise ValueError(f"current box {current_box_id} is not tracked")
            if current_box_id in on_pallet or box.status not in PENDING_STATUSES:
                raise ValueError(f"current box {current_box_id} has status {box.status.value}")
            return current_box_id
        pending = sorted(k for k, b in tracked.items() if b.status in PENDING_STATUSES and k not in on_pallet)
        if len(pending) > 1:
            raise ValueError(f"several pending boxes {pending}: pass current_box_id")
        return pending[0] if pending else None

    def _slots(self, state, tracked, buffer_slots, current_id):
        buffered = sorted(k for k, b in tracked.items() if b.status == BoxStatus.BUFFERED)
        n = self.config.buffer.slots
        if buffer_slots is None:
            if len(buffered) > n:
                raise ValueError(f"{len(buffered)} buffered boxes but only {n} slots")
            return dict(enumerate(buffered))
        slots = {int(k): v for k, v in buffer_slots.items()}
        if any(not 0 <= k < n for k in slots):
            raise ValueError(f"buffer slot outside 0..{n - 1}")
        if sorted(slots.values()) != buffered:
            raise ValueError("buffer_slots must list exactly the BUFFERED tracked boxes")
        if current_id in slots.values():
            raise ValueError("current box cannot also be buffered")
        return slots

    def _box_from_placed(self, p):
        spec = self.context.catalog.get(p.sku_id)
        yaws = tuple(spec.allowed_yaws_rad) if spec is not None else (p.pose.yaw,)
        return BoxState(p.box_id, p.sku_id, p.size, p.weight_kg, p.pose, yaws, BoxStatus.PLACED, 1.0, 0.0,
                        "STATE_SNAPSHOT")


def as_dict(decision):
    """JSON-friendly summary (logs, ROS message bridging)."""
    a = decision.action
    pose = decision.candidate.target_pose if decision.candidate is not None else None
    return {
        "state_version": decision.state_version,
        "action": None if a is None else a.type.value,
        "slot": decision.slot,
        "box_id": None if decision.box is None else decision.box.box_id,
        "requires_low_level": decision.requires_low_level,
        "feasible_pose": None if pose is None else
        {"x": pose.x, "y": pose.y, "z": pose.z, "yaw": pose.yaw},
        "repack_moves": [{"box_id": b, "x": p.x, "y": p.y, "z": p.z, "yaw": p.yaw}
                         for b, p in decision.repack_moves],
        "mask": list(decision.mask),
        "probabilities": None if decision.probabilities is None else list(decision.probabilities),
        "decided_by": decision.decided_by,
        "reason": decision.reason,
    }


__all__ = ["HighLevelDecider", "HighLevelDecision", "LoadedPolicy", "as_dict", "load_policy"]
