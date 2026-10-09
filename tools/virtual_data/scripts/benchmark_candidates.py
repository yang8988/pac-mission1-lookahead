#!/usr/bin/env python3
"""Benchmark 5-1/5-2 on virtual-data scenes against a dense grid oracle.

    python tools/virtual_data/scripts/benchmark_candidates.py \
        --scenes tools/virtual_data/output/run1/scenes --sample 60 --step 0.02
"""

import argparse
import json
from pathlib import Path
import random
import statistics
import sys
import time

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "scripts" / "taehyeon"))
import team_paths  # noqa: E402

team_paths.bootstrap()

from pac_common.adapters import context_from_json, state_from_json  # noqa: E402
from pac_candidates import CandidateBackend, load_candidate_config  # noqa: E402
from virtual_data.oracle import compare  # noqa: E402


def load_scene(path):
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    state = state_from_json(data["state"])
    context = context_from_json(data["context"])
    return data, state.inventory.tracked_boxes[data["current_box_id"]], state, context


def run(scene_paths, cfg, step):
    rows = []
    for path in scene_paths:
        data, box, state, context = load_scene(path)
        backend = CandidateBackend(context, cfg)
        t = time.perf_counter()
        cset = backend.candidate_set(box, state)
        elapsed = time.perf_counter() - t
        row = compare(backend, box, state, cset, step)
        row.update(scene=Path(path).name, elapsed_ms=1e3 * elapsed, boxes=len(state.pallet.boxes))
        rows.append(row)
    return rows


def report(rows):
    feasible = [r for r in rows if r["grid_feasible"] > 0]
    gaps = [r["best_top_gap"] for r in rows if r["best_top_gap"] is not None]
    return {
        "scenes": len(rows),
        "scenes_with_feasible_grid_pose": len(feasible),
        "recall_any": (
            sum(1 for r in feasible if r["candidate_valid"] > 0) / len(feasible) if feasible else 1.0
        ),
        "missed_all": sum(r["missed_all"] for r in rows),
        "candidate_only": sum(r["candidate_only"] for r in rows),
        "best_top_gap_mean_mm": 1e3 * statistics.fmean(gaps) if gaps else 0.0,
        "best_top_gap_max_mm": 1e3 * max(gaps) if gaps else 0.0,
        "best_top_equal_or_better": sum(1 for g in gaps if g <= 1e-6) / len(gaps) if gaps else 1.0,
        "mean_generated": statistics.fmean(r["candidate_generated"] for r in rows),
        "mean_valid": statistics.fmean(r["candidate_valid"] for r in rows),
        "mean_ms": statistics.fmean(r["elapsed_ms"] for r in rows),
        "p95_ms": sorted(r["elapsed_ms"] for r in rows)[int(0.95 * (len(rows) - 1))],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenes", type=Path, required=True)
    parser.add_argument("--sample", type=int, default=60)
    parser.add_argument("--step", type=float, default=0.02)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--candidate-config", type=Path, default=REPO / "config/taehyeon/candidates.yaml")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    paths = sorted(args.scenes.rglob("S*-T*.json"))
    random.Random(args.seed).shuffle(paths)
    paths = sorted(paths[: args.sample])
    rows = run(paths, load_candidate_config(args.candidate_config), args.step)
    summary = report(rows)
    print(json.dumps(summary, indent=1))
    for r in rows:
        if r["missed_all"] or (r["best_top_gap"] or 0) > 0.05:
            print("  check:", r["scene"], {k: r[k] for k in ("grid_feasible", "candidate_valid", "best_top_gap")})
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps({"summary": summary, "rows": rows}, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
