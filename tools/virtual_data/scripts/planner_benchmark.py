#!/usr/bin/env python3
"""Run donghan's planner (5-3~5-6) with taehyeon's 5-1/5-2 on virtual scenes.

    python tools/virtual_data/scripts/planner_benchmark.py --scenes OUT/scenes --sample 30
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
if team_paths.add_optional("pac_planning") is None:
    sys.exit("pac_planning not found; run scripts/taehyeon/fetch_team_deps.sh")

from pac_planning import PlacementPlanner, PlannerConfig  # noqa: E402
from pac_planning.demo import scene_from_file  # noqa: E402
from pac_candidates import CandidateBackend, load_candidate_config  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenes", type=Path, required=True)
    parser.add_argument("--sample", type=int, default=30)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--fixed-work", action="store_true", help="disable the 1 s soft budget")
    parser.add_argument("--candidate-config", type=Path, default=REPO / "config/taehyeon/candidates.yaml")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    cfg = load_candidate_config(args.candidate_config)
    paths = sorted(args.scenes.rglob("S*-T*.json"))
    random.Random(args.seed).shuffle(paths)
    rows = []
    for path in sorted(paths[: args.sample]):
        _, box, state, context = scene_from_file(path)
        backend = CandidateBackend(context, cfg)
        report = backend.generate_with_report(box, state)
        ctx = backend.context_with_ems(context, report)
        planner = PlacementPlanner(
            context=ctx,
            config=PlannerConfig(),
            generate_candidates=backend.generate_candidates,
            validate_constraints=backend.validate_constraints,
        )
        t = time.perf_counter()
        result = planner.plan(box, state, list(report.candidates), seed=7,
                              use_time_budget=not args.fixed_work)
        elapsed = time.perf_counter() - t
        d = result.diagnostics
        top = result.ranked[0] if result.ranked else None
        if top is not None:
            assert backend.validate_constraints(box, top, state).success
        rows.append({
            "scene": path.name, "pallet_boxes": len(state.pallet.boxes),
            "generated": d["generated_count"], "valid": d["valid_count"],
            "ranked": len(result.ranked), "time_sec": elapsed,
            "completed_scenarios": d.get("completed_scenarios", 0),
            "degradation": list(d.get("degradation", ())),
            "budget_exceeded": bool(d.get("budget_exceeded", False)),
            "ems_supplied": sum(e.features.geometry_source == "EMS_SUPPLIED" for e in result.evaluations),
            "evaluations": len(result.evaluations),
            "top": None if top is None else {"id": top.candidate_id, "score": top.score},
        })
        print(f"{path.name:18s} boxes={len(state.pallet.boxes):2d} valid={d['valid_count']:3d} "
              f"t={elapsed * 1e3:6.0f}ms scen={rows[-1]['completed_scenarios']} degr={rows[-1]['degradation']}")
    planned = [r for r in rows if r["valid"]]
    times = sorted(r["time_sec"] for r in planned)
    summary = {
        "scenes": len(rows), "scenes_with_valid": len(planned),
        "budget": "fixed-work" if args.fixed_work else "1s soft budget",
        "time_mean_ms": 1e3 * statistics.fmean(times) if times else 0,
        "time_p95_ms": 1e3 * times[int(0.95 * (len(times) - 1))] if times else 0,
        "time_max_ms": 1e3 * max(times) if times else 0,
        "budget_exceeded": sum(r["budget_exceeded"] for r in planned),
        "degraded": sum(bool(r["degradation"]) for r in planned),
        "full_7_scenarios": sum(r["completed_scenarios"] == 7 for r in planned),
        "ems_supplied_ratio": (sum(r["ems_supplied"] for r in planned)
                               / max(1, sum(r["evaluations"] for r in planned))),
    }
    print(json.dumps(summary, indent=1))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps({"summary": summary, "rows": rows}, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
