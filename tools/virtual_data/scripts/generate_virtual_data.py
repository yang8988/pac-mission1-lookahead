#!/usr/bin/env python3
"""Generate virtual data for stages 5-1/5-2.

Examples::

    # use an existing jaesung-generator output directory
    python tools/virtual_data/scripts/generate_virtual_data.py \
        --dataset path/to/generated --output tools/virtual_data/output/run1

    # or let this script run jaesung's generator first (sample: 2 per family)
    python tools/virtual_data/scripts/generate_virtual_data.py \
        --run-generator sample --sample-per-family 2 --output tools/virtual_data/output/run1
"""

import argparse
from pathlib import Path
import sys
import time

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "scripts" / "taehyeon"))
import team_paths  # noqa: E402

team_paths.bootstrap()

from pac_candidates import load_candidate_config  # noqa: E402
from virtual_data import generate, load_virtual_config  # noqa: E402
from virtual_data.scenario_source import run_generator  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", type=Path, help="existing generator output directory")
    parser.add_argument("--run-generator", choices=("sample", "benchmark"))
    parser.add_argument("--sample-per-family", type=int, default=2)
    parser.add_argument("--generator-seed", type=int)
    parser.add_argument("--boxes-per-scenario", type=int,
                        help="override the generator's boxes_per_scenario (copy of its config)")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--candidate-config", type=Path, default=REPO / "config/taehyeon/candidates.yaml")
    parser.add_argument("--virtual-config", type=Path, default=REPO / "config/taehyeon/virtual_data.yaml")
    parser.add_argument("--policy", choices=("dblf", "random", "mixed", "planner"))
    parser.add_argument("--scenarios", nargs="*", help="limit to these scenario IDs")
    parser.add_argument("--max-scenarios", type=int, default=0)
    parser.add_argument("--no-labels", action="store_true")
    parser.add_argument("--no-scenes", action="store_true")
    args = parser.parse_args()

    dataset = args.dataset
    if args.run_generator:
        gen_root = team_paths.locate("generator")
        common = team_paths.locate("pac_common")
        if gen_root is None:
            parser.error("jaesung generator not found; run scripts/taehyeon/fetch_team_deps.sh")
        dataset = args.output / "source_dataset"
        print(run_generator(gen_root, common, dataset, args.run_generator,
                            args.sample_per_family, args.generator_seed,
                            boxes_per_scenario=args.boxes_per_scenario))
    if dataset is None:
        parser.error("--dataset or --run-generator is required")

    cand_cfg = load_candidate_config(args.candidate_config)
    vcfg = load_virtual_config(args.virtual_config)
    if args.policy:
        from dataclasses import replace
        vcfg = replace(vcfg, episode=replace(vcfg.episode, policy=args.policy))
        if args.policy == "planner":
            team_paths.add_optional("pac_planning")

    started = time.perf_counter()

    def progress(done, total, result):
        m = result.metrics
        print(f"[{done}/{total}] {result.scenario_id} {result.family:12s} placed "
              f"{m['placed']}/{m['boxes']} util={m['pallet_volume_utilization']:.3f} "
              f"H={m['max_height_m']:.2f} step={1e3 * m['mean_step_sec']:.1f}ms", flush=True)

    manifest, summary = generate(
        dataset, args.output, cand_cfg, vcfg,
        scenario_ids=set(args.scenarios) if args.scenarios else None,
        repo_root=REPO, max_scenarios=args.max_scenarios,
        write_labels=not args.no_labels, write_scenes=not args.no_scenes, progress=progress,
    )
    print(f"done: {manifest['scenario_count']} scenarios, {manifest['scene_count']} scenes "
          f"in {time.perf_counter() - started:.1f}s -> {args.output}")
    print((args.output / "analysis" / "summary.md").read_text(encoding="utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
