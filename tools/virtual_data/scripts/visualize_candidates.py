#!/usr/bin/env python3
"""Render 5-1/5-2 decisions of planner scenes as 2D top-view SVGs.

    python tools/virtual_data/scripts/visualize_candidates.py SCENE.json [...] --output DIR
Hover a footprint in a browser to see its candidate ID, z and mask reasons.
"""

import argparse
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "scripts" / "taehyeon"))
import team_paths  # noqa: E402

team_paths.bootstrap()

import json  # noqa: E402

from pac_common.adapters import context_from_json, state_from_json  # noqa: E402
from pac_candidates import CandidateBackend, load_candidate_config  # noqa: E402
from virtual_data.policies import dblf_key  # noqa: E402
from virtual_data.visualize import decision_svg  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("scenes", type=Path, nargs="+")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--candidate-config", type=Path, default=REPO / "config/taehyeon/candidates.yaml")
    args = parser.parse_args()
    from dataclasses import replace

    # every failing check is listed (verdicts are identical to runtime mode)
    cfg = replace(load_candidate_config(args.candidate_config), collect_all_reasons=True)
    args.output.mkdir(parents=True, exist_ok=True)
    for path in args.scenes:
        data = json.loads(path.read_text(encoding="utf-8"))
        state = state_from_json(data["state"])
        context = context_from_json(data["context"])
        box = state.inventory.tracked_boxes[data["current_box_id"]]
        cset = CandidateBackend(context, cfg).candidate_set(box, state)
        chosen = min(cset.valid, key=dblf_key).candidate_id if cset.valid else None
        svg = decision_svg(state, box, cset, chosen, title=f"{data['scenario_id']}: 5-1 candidates / 5-2 hard mask")
        target = args.output / (path.stem + ".svg")
        target.write_text(svg, encoding="utf-8")
        print(f"{target}  {cset.summary()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
