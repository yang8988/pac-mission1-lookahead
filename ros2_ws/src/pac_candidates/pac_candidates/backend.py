"""Team-facing backend: v0.2 callbacks for stages 5-1 and 5-2.

```python
backend = CandidateBackend(context, load_candidate_config(path))
planner = PlacementPlanner(
    context=context,
    generate_candidates=backend.generate_candidates,    # 5-1
    validate_constraints=backend.validate_constraints,  # 5-2
    ...
)
```

Both callbacks are pure functions of their inputs (no state mutation, no
robot I/O). Derived pallet geometry and validation verdicts are cached per
snapshot because stage 5-5 rollouts call them thousands of times.
"""

from collections import Counter, OrderedDict
from dataclasses import replace
import time

from pac_common import (
    BoxStatus,
    PlacementCandidate,
    Pose3D,
    RejectCode as R,
    ValidationResult,
)

from .candidate_generation import (
    containing_ems,
    deduplicate,
    extreme_points,
    raw_candidates,
)
from .config import CandidateConfig
from .hard_mask import evaluate
from .pallet_model import PalletModel, box_tolerance
from .reports import CandidateInfo, CandidateSet, GenerationReport

INVALID_BOX_STATUSES = (BoxStatus.REJECTED, BoxStatus.FAILED, BoxStatus.PLACED)


class _Counter:
    def __init__(self):
        self.hits = 0
        self.misses = 0


class _Lru:
    def __init__(self, size):
        self.size = size
        self.data = OrderedDict()
        self.hits = 0
        self.misses = 0

    def get(self, key):
        value = self.data.get(key)
        if value is None:
            self.misses += 1
            return None
        self.hits += 1
        self.data.move_to_end(key)
        return value

    def put(self, key, value):
        self.data[key] = value
        self.data.move_to_end(key)
        while len(self.data) > self.size:
            self.data.popitem(last=False)


def state_key(state):
    pallet = state.pallet
    return (pallet.pallet_id, pallet.size, pallet.boxes)


def geometry_key(box, uncertain):
    """Everything the geometric checks depend on (NOT the box ID), so that
    look-ahead boxes of the same SKU share cached results."""
    return (box.size, box.weight_kg, box.allowed_yaws_rad, uncertain)


def pose_key(pose):
    return (pose.frame_id, pose.x, pose.y, pose.z, pose.roll, pose.pitch, pose.yaw)


class CandidateBackend:
    def __init__(self, context=None, config=None):
        self.context = context
        self.config = config or CandidateConfig()
        # Only models live in the LRU. Verdicts and candidate lists are stored
        # on their PalletModel, so evicting a model frees everything derived
        # from that snapshot (no cache pins an evicted model).
        self._models = _Lru(self.config.cache_size)
        self._verdicts = _Counter()
        self._generations = _Counter()
        self._last = (None, None)

    # ------------------------------------------------------------------
    # Derived geometry
    # ------------------------------------------------------------------

    def model_for(self, state):
        last_state, last_model = self._last
        if state is last_state or (
            last_state is not None and state.pallet is last_state.pallet
        ):
            return last_model
        key = state_key(state)
        model = self._models.get(key)
        if model is None:
            model = PalletModel(state, self.context, self.config)
            self._models.put(key, model)
        self._last = (state, model)
        return model

    def _uncertain(self, box_id):
        return self.context is not None and box_id in self.context.uncertain_box_ids

    # ------------------------------------------------------------------
    # 5-2 Hard mask
    # ------------------------------------------------------------------

    def _identity_failure(self, box, candidate, state):
        if candidate.base_state_version != state.state_version:
            return ValidationResult(False, (R.STALE_PLAN,), {"reasons": ("STALE_PLAN",)})
        if candidate.box_id != box.box_id:
            return ValidationResult(
                False, (R.INVALID_STATE,), {"reasons": ("BOX_ID_MISMATCH",)}
            )
        if any(b.box_id == box.box_id for b in state.pallet.boxes):
            return ValidationResult(
                False, (R.INVALID_STATE,), {"reasons": ("ALREADY_PLACED",)}
            )
        if box.status in INVALID_BOX_STATUSES:
            return ValidationResult(
                False, (R.INVALID_STATE,), {"reasons": ("BOX_STATUS_" + box.status.value,)}
            )
        if candidate.target_pose.frame_id != "pallet":
            return ValidationResult(
                False, (R.INVALID_STATE,), {"reasons": ("FRAME_NOT_PALLET",)}
            )
        if self._uncertain_rejects(box, state):
            return ValidationResult(
                False, (R.SENSOR_UNCERTAIN,), {"reasons": ("UNCERTAIN_POLICY_REJECT",)}
            )
        return None

    def _uncertain_rejects(self, box, state):
        """``uncertain_policy: reject`` and an uncertain box is involved."""
        if self.config.uncertainty.uncertain_policy != "reject" or self.context is None:
            return False
        ids = self.context.uncertain_box_ids
        return box.box_id in ids or any(b.box_id in ids for b in state.pallet.boxes)

    def _verdict_entry(self, box, pose, state):
        model = self.model_for(state)
        uncertain = self._uncertain(box.box_id)
        key = (geometry_key(box, uncertain), pose_key(pose))
        cached = model.verdict_cache.get(key)
        if cached is not None:
            self._verdicts.hits += 1
            return cached
        self._verdicts.misses += 1
        outcome = evaluate(
            model,
            box,
            pose,
            is_uncertain=uncertain,
            collect_all=self.config.collect_all_reasons,
        )
        entry = [None, outcome, None]  # [unused, outcome, lazy ValidationResult]
        model.verdict_cache[key] = entry
        return entry

    @staticmethod
    def _result(entry):
        if entry[2] is None:
            outcome = entry[1]
            if outcome.codes:
                entry[2] = ValidationResult(
                    False,
                    outcome.codes,
                    {"reasons": outcome.reasons, "metrics": dict(outcome.metrics)},
                )
            else:
                entry[2] = ValidationResult(
                    True,
                    details={"evidence": outcome.evidence, "metrics": dict(outcome.metrics)},
                )
        return entry[2]

    def evaluate_pose(self, box, pose, state):
        """Raw ``MaskOutcome`` (codes, reasons, metrics, evidence); cached."""
        return self._verdict_entry(box, pose, state)[1]

    def validate_constraints(self, box, candidate, state):
        """v0.2 ``validate_constraints(box, candidate, state)``."""
        failure = self._identity_failure(box, candidate, state)
        if failure is not None:
            return failure
        try:
            return self._result(self._verdict_entry(box, candidate.target_pose, state))
        except (ValueError, KeyError, ArithmeticError) as error:
            return ValidationResult(
                False,
                (R.INVALID_STATE,),
                {"reasons": ("INVALID_INPUT",), "error": str(error)},
            )

    # ------------------------------------------------------------------
    # 5-1 Candidate generation
    # ------------------------------------------------------------------

    def _kept_raws(self, box, state, model):
        """Deduplicated raw candidates; cached per (snapshot, box geometry)."""
        uncertain = self._uncertain(box.box_id)
        key = geometry_key(box, uncertain)
        cached = model.generation_cache.get(key)
        if cached is not None:
            self._generations.hits += 1
            return cached
        self._generations.misses += 1
        gen = self.config.generation
        raws = raw_candidates(model, box, self.config)
        if gen.dedup_mode == "off":
            kept = raws
        else:
            check = None
            if gen.dedup_mode == "mask_aware":
                def check(raw):
                    pose = Pose3D("pallet", raw.x, raw.y, raw.z, yaw=raw.yaw)
                    return not self.evaluate_pose(box, pose, state).codes

            elif gen.dedup_mode == "support_aware":
                def check(raw):
                    return raw.proxy_ok

            dist = gen.dedup_distance_m
            if gen.dedup_distance_ratio > 0:
                dist = gen.dedup_distance_ratio * min(box.size.x, box.size.y)
            kept, _ = deduplicate(raws, dist, model.height_tol, check)
        if gen.order == "likely_valid_first":
            # Stable: priority order inside each group. Consumers that only
            # look at the first N candidates (e.g. 5-3 inventory probes) then
            # see feasible positions first.
            kept = sorted(kept, key=lambda r: not r.proxy_ok)
        if gen.max_candidates:
            kept = kept[: gen.max_candidates]
        resolved = []
        for raw in kept:
            if raw.ems is None:
                raw = replace(raw, ems=containing_ems(model, raw.expanded, raw.z))
            resolved.append(raw)
        kept = tuple(resolved)
        n_ep = len(extreme_points(model)) if gen.use_extreme_points else 0
        model.generation_cache[key] = (kept, len(raws), n_ep)
        return kept, len(raws), n_ep

    def generate_with_report(self, box, state):
        started = time.perf_counter()
        model = self.model_for(state)
        gen = self.config.generation
        if self._uncertain_rejects(box, state):
            kept, raw_count, n_ep = [], 0, 0  # every candidate would be rejected anyway
        else:
            kept, raw_count, n_ep = self._kept_raws(box, state, model)
        tol = box_tolerance(self.config.uncertainty, self._uncertain(box.box_id))
        margin = model.half_gap + tol
        candidates = []
        infos = {}
        for index, raw in enumerate(kept):
            candidate_id = f"S{state.state_version:04d}-{box.box_id}-C{index:03d}"
            candidates.append(
                PlacementCandidate(
                    candidate_id,
                    box.box_id,
                    Pose3D("pallet", raw.x, raw.y, raw.z, yaw=raw.yaw),
                    state.state_version,
                )
            )
            lower = upper = None
            if raw.ems is not None:
                r = raw.ems.rect
                lower = (r.x0 + margin, r.y0 + margin, raw.ems.level)
                upper = (r.x1 - margin, r.y1 - margin, raw.ems.top)
            infos[candidate_id] = CandidateInfo(
                candidate_id,
                raw.source,
                raw.anchor,
                raw.yaw,
                raw.dims,
                lower,
                upper,
            )
        return GenerationReport(
            state_version=state.state_version,
            box_id=box.box_id,
            candidates=tuple(candidates),
            infos=infos,
            raw_count=raw_count,
            ems_count=len(model.ems()) if gen.use_ems else 0,
            extreme_point_count=n_ep,
            yaws_rad=tuple(sorted({r.yaw for r in kept})),
            elapsed_sec=time.perf_counter() - started,
        )

    def generate_candidates(self, box, state):
        """v0.2 ``generate_candidates(box, state) -> list[PlacementCandidate]``."""
        return list(self.generate_with_report(box, state).candidates)

    # ------------------------------------------------------------------
    # 5-1 + 5-2 together (what stage 5-3 consumes)
    # ------------------------------------------------------------------

    def candidate_set(self, box, state):
        generation = self.generate_with_report(box, state)
        started = time.perf_counter()
        valid = []
        rejected = {}
        codes = Counter()
        reasons = Counter()
        for candidate in generation.candidates:
            verdict = self.validate_constraints(box, candidate, state)
            if verdict.success:
                valid.append(candidate)
                continue
            rejected[candidate.candidate_id] = verdict
            codes.update(code.value for code in verdict.codes)
            reasons.update(
                {r.split(":", 1)[0] for r in verdict.details.get("reasons", ())}
            )
        return CandidateSet(
            generation=generation,
            valid=tuple(valid),
            rejected=rejected,
            code_counts=dict(codes),
            reason_counts=dict(reasons),
            mask_elapsed_sec=time.perf_counter() - started,
            snapshot_issues=self.model_for(state).snapshot_issues(),
        )

    def context_with_ems(self, context, generation):
        """PlanningContext with ``ems_upper_by_candidate`` from 5-1.

        Stage 5-3 then marks these candidates ``EMS_SUPPLIED`` (real EMS
        extents) instead of the footprint-column proxy.
        """
        upper = dict(context.ems_upper_by_candidate)
        for candidate_id, info in generation.infos.items():
            if info.ems_upper_m is not None:
                x, y, z = info.ems_upper_m
                upper[candidate_id] = Pose3D("pallet", x, y, z)
        return replace(context, ems_upper_by_candidate=upper)

    def cache_stats(self):
        return {
            "model_hits": self._models.hits,
            "model_misses": self._models.misses,
            "verdict_hits": self._verdicts.hits,
            "verdict_misses": self._verdicts.misses,
            "generation_hits": self._generations.hits,
            "generation_misses": self._generations.misses,
        }
