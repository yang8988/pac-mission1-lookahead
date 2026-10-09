"""Consistency checks for a virtual-data output directory."""

from collections import defaultdict
import json
from pathlib import Path
import random

from pac_common import BoxStatus, PlacementCandidate, Pose3D
from pac_common.adapters import context_from_json, state_from_json

from pac_candidates import CandidateBackend, config_from_dict

FORBIDDEN_SCENE_KEYS = (
    "arrival_events",
    "ground_truth",
    "arrival_index",
    "true_capacity_n",
    "damaged_box_ids",
)


def _scan_keys(value, found):
    if isinstance(value, dict):
        for k, v in value.items():
            if k in FORBIDDEN_SCENE_KEYS:
                found.add(k)
            _scan_keys(v, found)
    elif isinstance(value, list):
        for v in value:
            _scan_keys(v, found)


def validate_output(out_dir, recheck_rows=30, seed=0):
    out = Path(out_dir)
    failures = []
    manifest_path = out / "manifest.json"
    if not manifest_path.exists():
        return ["missing manifest.json"]
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    cand_cfg = config_from_dict(_config_dict(manifest["candidate_config"]))

    split_of = defaultdict(set)
    scene_paths = sorted((out / "scenes").rglob("S*-T*.json"))
    if len(scene_paths) != manifest["scene_count"]:
        failures.append("scene count mismatch with manifest")
    for path in scene_paths:
        data = json.loads(path.read_text(encoding="utf-8"))
        found = set()
        _scan_keys(data, found)
        if found:
            failures.append(f"future/ground-truth leakage {sorted(found)} in {path.name}")
        try:
            state = state_from_json(data["state"])
            context_from_json(data["context"])
        except (KeyError, TypeError, ValueError) as error:
            failures.append(f"scene adapter failure {path.name}: {error}")
            continue
        box = state.inventory.tracked_boxes.get(data["current_box_id"])
        if box is None or box.status != BoxStatus.READY_FOR_PICK:
            failures.append(f"current box missing/not pending in {path.name}")
        if any(b.box_id == data["current_box_id"] for b in state.pallet.boxes):
            failures.append(f"current box already on pallet in {path.name}")
        split_of[data["source"]["generator_scenario_id"]].add(path.parent.name)
    for scenario, splits in split_of.items():
        if len(splits) > 1:
            failures.append(f"scenario {scenario} appears in several splits {sorted(splits)}")

    rows = []
    for path in sorted((out / "candidate_sets").glob("*.jsonl")):
        with path.open(encoding="utf-8") as f:
            for line in f:
                rows.append(json.loads(line))
    for row in rows:
        ids = [c["candidate_id"] for c in row["candidates"]]
        if len(ids) != len(set(ids)):
            failures.append(f"duplicate candidate ids {row['scenario_id']} step {row['step']}")
        valid_ids = {c["candidate_id"] for c in row["candidates"] if c["valid"]}
        if row["valid_count"] != len(valid_ids):
            failures.append(f"valid_count mismatch {row['scenario_id']} step {row['step']}")
        chosen = row["chosen_candidate_id"]
        if chosen is not None and chosen not in valid_ids:
            failures.append(f"chosen candidate not valid {row['scenario_id']} step {row['step']}")
        for c in row["candidates"]:
            if c["valid"] and "evidence" not in c:
                failures.append(f"valid candidate without evidence {c['candidate_id']}")
            if not c["valid"] and not c["reject_codes"]:
                failures.append(f"invalid candidate without codes {c['candidate_id']}")

    # Determinism: recompute a sample of label rows from their scenes.
    scenes = {p.stem: p for p in scene_paths}
    rng = random.Random(seed)
    sample = [r for r in rows if f"{r['scenario_id']}-T{r['step']:03d}" in scenes]
    rng.shuffle(sample)
    for row in sample[:recheck_rows]:
        data = json.loads(scenes[f"{row['scenario_id']}-T{row['step']:03d}"].read_text(encoding="utf-8"))
        state = state_from_json(data["state"])
        context = context_from_json(data["context"])
        box = state.inventory.tracked_boxes[data["current_box_id"]]
        backend = CandidateBackend(context, cand_cfg)
        for c in row["candidates"]:
            pose = c["pose"]
            cand = PlacementCandidate(
                c["candidate_id"],
                box.box_id,
                Pose3D("pallet", pose["x"], pose["y"], pose["z"], yaw=pose["yaw"]),
                state.state_version,
            )
            verdict = backend.validate_constraints(box, cand, state)
            if verdict.success != c["valid"]:
                failures.append(f"re-validation differs for {c['candidate_id']}")
        regenerated = [x.candidate_id for x in backend.generate_candidates(box, state)]
        if regenerated != [c["candidate_id"] for c in row["candidates"]]:
            failures.append(f"re-generation differs {row['scenario_id']} step {row['step']}")

    for path in sorted((out / "episodes").glob("*.json")):
        ep = json.loads(path.read_text(encoding="utf-8"))
        state = state_from_json(ep["final_state"])
        final_context = (
            context_from_json(ep["final_context"]) if ep.get("final_context") else None
        )
        issues = CandidateBackend(final_context, cand_cfg).model_for(state).snapshot_issues()
        if issues:
            failures.append(f"final pallet has stability/load issues in {path.name}: {issues}")
        if ep["metrics"]["true_overlaps"]:
            failures.append(f"true-size interpenetration in {path.name}")
        if ep["metrics"].get("boxes_on_detected_damaged"):
            failures.append(f"box stacked on a detected damaged box in {path.name}")
    return failures


def _config_dict(raw):
    """``plain(CandidateConfig)`` -> mapping accepted by ``config_from_dict``."""
    return json.loads(json.dumps(raw))
