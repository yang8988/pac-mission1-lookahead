"""Replay one scenario through stages 5-1/5-2 and record everything."""

from collections import Counter
from dataclasses import dataclass, replace
import math
import random
import statistics
import time

from pac_common import (
    BoxStatus,
    InventoryState,
    PalletState,
    PlacedBox,
    PlanningContext,
    Size3D,
    SystemState,
    plain,
)

from pac_candidates import CandidateBackend
from pac_candidates.geometry import rotated_dims

from .observation import observe
from .policies import choose
from .scenario_source import stack_height_limit
from .strength import PROFILE_ORDER, draw as draw_strength, footprint_capacity_summary, true_overloads

SCENE_SCHEMA = "pac-common-v0.2+planning-v1"
LABEL_SCHEMA = 1
TRUE_OVERLAP_TOL_M = 0.0005


@dataclass
class EpisodeResult:
    scenario_id: str
    family: str
    split: str
    placed: list
    unplaced: list
    steps: list  # per-step summary dicts
    final_state: object
    metrics: dict
    strength: object = None
    final_context: object = None


def candidate_rows(candidate_set):
    rows = []
    infos = candidate_set.generation.infos
    for cand in candidate_set.generation.candidates:
        info = infos[cand.candidate_id]
        verdict = candidate_set.rejected.get(cand.candidate_id)
        pose = cand.target_pose
        row = {
            "candidate_id": cand.candidate_id,
            "pose": {"x": pose.x, "y": pose.y, "z": pose.z, "yaw": pose.yaw},
            "source": info.source,
            "anchor": info.anchor,
            "ems_upper_m": info.ems_upper_m,
            "valid": verdict is None,
            "reject_codes": [] if verdict is None else [c.value for c in verdict.codes],
            "reasons": [] if verdict is None else list(verdict.details.get("reasons", ())),
        }
        rows.append(row)
    return rows


def attach_evidence(rows, backend, box, state, candidates):
    by_id = {c.candidate_id: c for c in candidates}
    for row in rows:
        if not row["valid"]:
            continue
        cand = by_id[row["candidate_id"]]
        verdict = backend.validate_constraints(box, cand, state)
        row["evidence"] = plain(verdict.details["evidence"])
        metrics = verdict.details["metrics"]
        row["metrics"] = {
            k: plain(metrics[k])
            for k in ("support_ratio", "lbcp_margin_m", "max_load_ratio", "pallet_cog_after_m")
            if k in metrics
        }


def true_geometry_violations(placed, truths, pallet_size):
    """Interpenetration / protrusion of the TRUE boxes at the planned centres.

    Planning used measured sizes; the robot places the box centre. This checks
    whether the clearance/tolerance delta absorbed the measurement error.
    """
    boxes = []
    for p in placed:
        dx, dy, dz = rotated_dims(p.size, p.pose.yaw)
        cx, cy = p.pose.x + dx / 2, p.pose.y + dy / 2
        t = truths[p.box_id]
        tx, ty, tz = rotated_dims(t.size, p.pose.yaw)
        boxes.append(
            (p.box_id, (cx - tx / 2, cy - ty / 2, p.pose.z), (cx + tx / 2, cy + ty / 2, p.pose.z + tz))
        )
    overlaps = []
    protrusions = []
    for i, (a, alo, ahi) in enumerate(boxes):
        if (
            alo[0] < -TRUE_OVERLAP_TOL_M
            or alo[1] < -TRUE_OVERLAP_TOL_M
            or ahi[0] > pallet_size.x + TRUE_OVERLAP_TOL_M
            or ahi[1] > pallet_size.y + TRUE_OVERLAP_TOL_M
        ):
            protrusions.append(a)
        for b, blo, bhi in boxes[i + 1 :]:
            pen = [min(ahi[k], bhi[k]) - max(alo[k], blo[k]) for k in range(3)]
            # vertical contact of stacked boxes is not penetration; height
            # error is resolved by the z of the next placement in reality.
            if pen[0] > TRUE_OVERLAP_TOL_M and pen[1] > TRUE_OVERLAP_TOL_M and pen[2] > 0.01:
                overlaps.append((a, b, round(min(pen[0], pen[1]), 5)))
    return overlaps, protrusions


def select_pallet_xy(spec, vcfg, family_index=0):
    """Pallet footprint for this scenario (covers pallet-spec changes).

    Round robin over the configured sizes *within each scenario family*, so
    every family (e.g. late_heavy) is tested on every footprint and size is
    not confounded with family or with the strength profile (which cycles
    over the global scenario index).
    """
    sizes = vcfg.pallet.sizes_m
    if not sizes:
        return spec.pallet_xy
    return sizes[family_index % len(sizes)]


def select_strength(spec, vcfg, scenario_index):
    cfg = vcfg.strength
    if not cfg.enabled:
        return None
    names = [n for n in PROFILE_ORDER if cfg.profiles.get(n, 0) > 0]
    rng = random.Random(f"{vcfg.seed}:{spec.scenario_id}:strength")
    if cfg.assignment == "round_robin":
        profile = names[scenario_index % len(names)]
    else:
        total = sum(cfg.profiles[n] for n in names)
        value = rng.random() * total
        profile = names[-1]
        for n in names:
            value -= cfg.profiles[n]
            if value <= 0:
                profile = n
                break
    return draw_strength(
        spec.arrivals,
        profile,
        rng,
        damage_factor=cfg.damage_factor,
        detect_probability=cfg.detect_probability,
    )


def run_episode(
    spec,
    catalog,
    cand_config,
    vcfg,
    split,
    scene_sink=None,
    label_sink=None,
    planner_factory=None,
    run_id="virtual",
    scenario_index=0,
    family_index=0,
):
    rng = random.Random(f"{vcfg.seed}:{spec.scenario_id}")
    policy_rng = random.Random(f"{vcfg.seed}:{spec.scenario_id}:policy")
    strength = select_strength(spec, vcfg, scenario_index)
    height = stack_height_limit(spec, vcfg)
    pallet_xy = select_pallet_xy(spec, vcfg, family_index)
    pallet_size = Size3D(pallet_xy[0], pallet_xy[1], height)
    max_load = (
        float(spec.max_load_kg) if spec.max_load_kg is not None else vcfg.pallet.default_max_load_kg
    )
    limit = vcfg.episode.max_steps or len(spec.arrivals)
    # the unseen pool is the (possibly truncated) stream actually replayed
    remaining = Counter(b.sku_id for b in spec.arrivals[:limit])
    placed = []
    tracked = {}
    truths = {}
    uncertain = []
    overrides = {}
    unplaced = []
    steps = []
    for step, truth in enumerate(spec.arrivals[:limit]):
        remaining[truth.sku_id] -= 1  # now observed -> leaves the unseen pool
        obs = observe(truth, rng, vcfg.observation, float(step))
        box = obs.box
        truths[box.box_id] = truth
        if obs.uncertain:
            uncertain.append(box.box_id)
        if strength is not None and box.box_id in strength.detected:
            # stage-2 inspection flags the dented top: placeable, nothing on top
            overrides[box.box_id] = 0.0
        context = PlanningContext(
            catalog=catalog,
            pallet_max_weight_kg=max_load,
            capacity_overrides_n=dict(overrides),
            uncertain_box_ids=tuple(uncertain),
        )
        state = SystemState(
            step,
            float(step),
            PalletState(spec.pallet_id, pallet_size, tuple(placed)),
            InventoryState(
                {**tracked, box.box_id: box},
                {k: v for k, v in sorted(remaining.items()) if v > 0},
            ),
        )
        backend = CandidateBackend(context, cand_config)
        started = time.perf_counter()
        cset = backend.candidate_set(box, state)
        total_sec = time.perf_counter() - started

        planner_call = None
        if planner_factory is not None:
            planner_call = planner_factory(backend, context, cset, box, state)
        chosen, decision = choose(vcfg.episode.policy, cset, policy_rng, vcfg.episode, planner_call)

        if scene_sink is not None and step % vcfg.episode.scene_every == 0:
            scene_sink(
                {
                    "schema_version": SCENE_SCHEMA,
                    "scenario_id": f"{spec.scenario_id}-T{step:03d}",
                    "current_box_id": box.box_id,
                    "state": plain(state),
                    "context": plain(context),
                    "source": {
                        "generator_scenario_id": spec.scenario_id,
                        "scenario_family": spec.family,
                        "split": split,
                        "step": step,
                        "state_kind": "SIMULATED",
                        "strength_profile": strength.profile if strength else None,
                        "pallet_xy_m": list(pallet_xy),
                        "producer": "taehyeon.virtual_data",
                    },
                },
                split,
            )
        if label_sink is not None:
            rows = candidate_rows(cset)
            attach_evidence(rows, backend, box, state, cset.generation.candidates)
            label_sink(
                {
                    "schema_version": LABEL_SCHEMA,
                    "run_id": run_id,
                    "scenario_id": spec.scenario_id,
                    "scenario_family": spec.family,
                    "split": split,
                    "step": step,
                    "state_version": state.state_version,
                    "box_id": box.box_id,
                    "box": plain(box),
                    "box_uncertain": obs.uncertain,
                    "box_damage_detected": box.box_id in overrides,
                    "strength_profile": strength.profile if strength else None,
                    "pallet_box_count": len(placed),
                    "raw_count": cset.generation.raw_count,
                    "generated_count": cset.generated_count,
                    "valid_count": cset.valid_count,
                    "masked_count": cset.masked_count,
                    "code_counts": dict(cset.code_counts),
                    "reason_counts": dict(cset.reason_counts),
                    "ems_count": cset.generation.ems_count,
                    "generation_sec": cset.generation.elapsed_sec,
                    "mask_sec": cset.mask_elapsed_sec,
                    "candidates": rows,
                    "chosen_candidate_id": chosen.candidate_id if chosen else None,
                    "decision": decision,
                    "seed": vcfg.seed,
                },
                spec.scenario_id,
            )
        steps.append(
            {
                "step": step,
                "generated": cset.generated_count,
                "valid": cset.valid_count,
                "raw": cset.generation.raw_count,
                "reason_counts": dict(cset.reason_counts),
                "code_counts": dict(cset.code_counts),
                "total_sec": total_sec,
                "generation_sec": cset.generation.elapsed_sec,
                "mask_sec": cset.mask_elapsed_sec,
                "decision": decision,
            }
        )
        if chosen is None:
            unplaced.append(box.box_id)
            continue
        pose = chosen.target_pose
        placed.append(PlacedBox(box.box_id, box.sku_id, box.size, box.weight_kg, pose))
        tracked[box.box_id] = replace(box, pose=pose, status=BoxStatus.PLACED)

    final_state = SystemState(
        len(spec.arrivals),
        float(len(spec.arrivals)),
        PalletState(spec.pallet_id, pallet_size, tuple(placed)),
        InventoryState(dict(tracked), {}),
    )
    # Re-check the final pallet with the SAME context the mask used during
    # the episode (catalog capacities, 0 N damage overrides, uncertain boxes).
    final_context = PlanningContext(
        catalog=catalog,
        pallet_max_weight_kg=max_load,
        capacity_overrides_n=dict(overrides),
        uncertain_box_ids=tuple(uncertain),
    )
    model = CandidateBackend(final_context, cand_config).model_for(final_state)
    overlaps, protrusions = true_geometry_violations(placed, truths, pallet_size)
    true_volume = sum(
        truths[p.box_id].size.x * truths[p.box_id].size.y * truths[p.box_id].size.z for p in placed
    )
    all_volume = sum(t.size.x * t.size.y * t.size.z for t in spec.arrivals[:limit])
    top = model.max_top
    gens = [s["generation_sec"] + s["mask_sec"] for s in steps]
    strength_metrics = {"strength_profile": None}
    if strength is not None:
        over, worst = true_overloads(final_state, strength, cand_config)
        on_detected = [
            b.box_id
            for b in placed
            if any(
                c.supporter_id in strength.detected
                for c in model.contacts.get(b.box_id, ())
            )
        ]
        strength_metrics = {
            "strength_profile": strength.profile,
            "true_capacity_kg": footprint_capacity_summary(strength, spec.arrivals[:limit]),
            "true_overloaded_boxes": over,
            "true_max_load_ratio": worst if math.isfinite(worst) else 1e6,
            "damaged_boxes": len(strength.damaged),
            "damaged_detected": len(strength.detected),
            "boxes_on_detected_damaged": on_detected,
        }
    metrics = {
        **strength_metrics,
        "pallet_xy_m": list(pallet_xy),
        "boxes": min(limit, len(spec.arrivals)),
        "placed": len(placed),
        "unplaced": len(unplaced),
        "placed_ratio": len(placed) / max(1, min(limit, len(spec.arrivals))),
        "placed_volume_ratio": true_volume / all_volume if all_volume else 0.0,
        "pallet_volume_utilization": true_volume / (pallet_size.x * pallet_size.y * pallet_size.z),
        "bounding_density": true_volume / (pallet_size.x * pallet_size.y * top) if top > 0 else 0.0,
        "max_height_m": top,
        "total_weight_kg": sum(p.weight_kg for p in placed),
        "snapshot_issues": {k: list(v) for k, v in model.snapshot_issues().items()},
        "true_overlaps": overlaps,
        "true_protrusions": protrusions,
        "uncertain_boxes": len(uncertain),
        "mean_step_sec": statistics.fmean(gens) if gens else 0.0,
        "max_step_sec": max(gens) if gens else 0.0,
        "mean_generated": statistics.fmean(s["generated"] for s in steps) if steps else 0.0,
        "mean_valid": statistics.fmean(s["valid"] for s in steps) if steps else 0.0,
    }
    if math.isnan(metrics["mean_step_sec"]):
        metrics["mean_step_sec"] = 0.0
    return EpisodeResult(
        spec.scenario_id, spec.family, split, placed, unplaced, steps, final_state, metrics, strength,
        final_context,
    )
