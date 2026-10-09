#!/usr/bin/env python3
"""Cross-check 5-2 verdicts with jaesung's PyBullet simulator.

    python tools/virtual_data/scripts/physics_crosscheck.py OUT_DIR --steps 40 --per-step 2
Requires pybullet and pac_simulation (scripts/taehyeon/fetch_team_deps.sh).
"""

import argparse
import json
from pathlib import Path
import random
import sys

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "scripts" / "taehyeon"))
import team_paths  # noqa: E402

team_paths.bootstrap()
sim_root = team_paths.add_optional("pac_simulation")

from pac_common import Pose3D, Size3D  # noqa: E402
from pac_common.adapters import state_from_json  # noqa: E402
from virtual_data.physics_check import check_candidate  # noqa: E402

STABILITY_CODES = {"LOW_SUPPORT", "COG_VIOLATION"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    parser.add_argument("--steps", type=int, default=40)
    parser.add_argument("--per-step", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--collision-model", default="solid", choices=("solid", "slatted"))
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    if sim_root is None:
        parser.error("pac_simulation not found; run scripts/taehyeon/fetch_team_deps.sh")
    import pac_simulation.ahead_sim as sim

    out = args.output
    truths = {}
    for path in (out / "episodes").glob("*.json"):
        ep = json.loads(path.read_text(encoding="utf-8"))
        for bid, s in ep["true_sizes"].items():
            truths[bid] = Size3D(**s)
    rows = []
    for path in sorted((out / "candidate_sets").glob("*.jsonl")):
        rows += [json.loads(line) for line in path.open(encoding="utf-8")]
    rng = random.Random(args.seed)
    rows = [r for r in rows if r["step"] >= 3]
    rng.shuffle(rows)
    results = {"valid": [], "rejected_stability": []}
    for row in rows[: args.steps]:
        scene = out / "scenes" / row["split"] / f"{row['scenario_id']}-T{row['step']:03d}.json"
        data = json.loads(scene.read_text(encoding="utf-8"))
        state = state_from_json(data["state"])
        box = state.inventory.tracked_boxes[data["current_box_id"]]
        valid = [c for c in row["candidates"] if c["valid"]]
        unstable = [
            c for c in row["candidates"]
            if not c["valid"] and set(c["reject_codes"]) <= STABILITY_CODES
            and not any(r.startswith("FLOATING") for r in c["reasons"])
        ]
        for group, pool in (("valid", valid), ("rejected_stability", unstable)):
            for c in rng.sample(pool, min(args.per_step, len(pool))):
                pose = Pose3D("pallet", c["pose"]["x"], c["pose"]["y"], c["pose"]["z"], yaw=c["pose"]["yaw"])
                outcome = check_candidate(
                    sim, state.pallet.boxes, truths, state.pallet.size, box,
                    truths.get(box.box_id, box.size), pose, collision_model=args.collision_model,
                )
                results[group].append(
                    {"scene": scene.name, "candidate_id": c["candidate_id"], "reasons": c["reasons"],
                     "stable": outcome.stable, "shift_m": outcome.new_box_shift_m,
                     "tilt_deg": outcome.new_box_tilt_deg, "existing_shift_m": outcome.worst_existing_shift_m,
                     "insertion_error": outcome.insertion_error,
                     "support_ratio": (c.get("metrics") or {}).get("support_ratio")}
                )
    summary = {}
    for group, items in results.items():
        n = len(items)
        stable = sum(1 for i in items if i["stable"])
        summary[group] = {"n": n, "physically_stable": stable,
                          "stable_ratio": stable / n if n else None,
                          "insertion_errors": sum(1 for i in items if i["insertion_error"])}
    summary["collision_model"] = args.collision_model
    print(json.dumps(summary, indent=1))
    for i in results["valid"]:
        if not i["stable"]:
            print("  valid-but-moved:", i)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps({"summary": summary, "results": results}, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
