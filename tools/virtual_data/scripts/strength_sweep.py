#!/usr/bin/env python3
"""Assumed carton capacity (planner) vs TRUE strength scenarios (hidden).

For each assumed McKee safety factor the virtual data is regenerated and the
true crush risk per strength profile is reported. This chooses a capacity
assumption without measured box data.

    python tools/virtual_data/scripts/strength_sweep.py --dataset GEN_OUT \
        --safety-factors 1.5 2 3 4 6 8 --report docs/taehyeon/reports/strength_sweep.json
"""

import argparse
from dataclasses import replace
import json
from pathlib import Path
import sys
import tempfile

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "scripts" / "taehyeon"))
import team_paths  # noqa: E402

team_paths.bootstrap()

from pac_candidates import load_candidate_config  # noqa: E402
from virtual_data import generate, load_virtual_config  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--safety-factors", type=float, nargs="+", default=[1.5, 2, 3, 4, 6, 8])
    parser.add_argument("--candidate-config", type=Path, default=REPO / "config/taehyeon/candidates.yaml")
    parser.add_argument("--virtual-config", type=Path, default=REPO / "config/taehyeon/virtual_data.yaml")
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    base = load_candidate_config(args.candidate_config)
    vcfg = load_virtual_config(args.virtual_config)
    rows = []
    for sf in args.safety_factors:
        lm = replace(base.constraints.load_model, safety_factor=sf)
        cfg = replace(base, constraints=replace(base.constraints, load_model=lm))
        with tempfile.TemporaryDirectory() as tmp:
            _, summary = generate(args.dataset, Path(tmp), cfg, vcfg, repo_root=REPO,
                                  write_labels=False, write_scenes=False)
        for profile, f in summary["strength_profiles"].items():
            rows.append({"safety_factor": sf, "profile": profile, **{k: f[k] for k in (
                "scenarios", "placed_ratio", "pallet_volume_utilization", "mean_max_height_m",
                "true_overloaded_boxes", "true_overload_rate", "scenarios_with_true_overload",
                "true_max_load_ratio", "boxes_on_detected_damaged")}})
        a = summary["families"]["ALL"]
        rows.append({"safety_factor": sf, "profile": "ALL", "scenarios": a["scenarios"],
                     "placed_ratio": a["placed_ratio"],
                     "pallet_volume_utilization": a["pallet_volume_utilization"],
                     "mean_max_height_m": a["mean_max_height_m"],
                     "true_overloaded_boxes": a["true_overloaded_boxes"],
                     "true_overload_rate": a["true_overload_rate"],
                     "scenarios_with_true_overload": a["scenarios_with_true_overload"],
                     "true_max_load_ratio": a["true_max_load_ratio"],
                     "boxes_on_detected_damaged": a["boxes_on_detected_damaged"]})
        print(f"SF={sf}: placed {100 * a['placed_ratio']:.1f}% util {a['pallet_volume_utilization']:.3f} "
              f"true overloads {a['true_overloaded_boxes']} max ratio {a['true_max_load_ratio']:.2f}", flush=True)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(rows, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
