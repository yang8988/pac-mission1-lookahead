"""Generate the evaluation dataset with jaesung's generator (read-only copy in .deps).

Defaults reproduce ``dataset80_x40_s8`` used in the reports: 40 scenarios per
family (240 in total, split 168/36/36), 80 boxes per scenario, seed 20261008.

    bash scripts/taehyeon/fetch_team_deps.sh
    python tools/highlevel/scripts/make_dataset.py
"""

import argparse
from pathlib import Path

from _common import REPO
import team_paths
from virtual_data.scenario_source import run_generator


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=REPO / "tools/highlevel/output/dataset80_x40_s8")
    parser.add_argument("--per-family", type=int, default=40)
    parser.add_argument("--boxes", type=int, default=80)
    parser.add_argument("--seed", type=int, default=20261008)
    args = parser.parse_args()
    if (args.output / "manifest.json").exists():
        print(f"exists: {args.output}")
        return
    generator = team_paths.locate("generator")
    common = team_paths.locate("pac_common")
    if generator is None or common is None:
        raise SystemExit("run scripts/taehyeon/fetch_team_deps.sh first")
    print(run_generator(generator, common, args.output, "sample", args.per_family,
                        seed=args.seed, boxes_per_scenario=args.boxes))


if __name__ == "__main__":
    main()
