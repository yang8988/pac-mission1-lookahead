"""Shared setup for the stage-4 scripts."""

import os
import sys
from pathlib import Path

# Worker processes each run single-threaded BLAS; nested BLAS threads in 4+
# forked workers oversubscribe the CPU (sb3 vector step 170 -> 128 ms).
for _var in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "scripts" / "taehyeon"))
import team_paths

team_paths.bootstrap()

from pac_candidates import load_candidate_config
from pac_highlevel import load_highlevel_config, load_lookahead_config
from virtual_data import load_virtual_config
from virtual_data.environment import apply_environment, load_environment
from virtual_data.scenario_source import load_dataset, run_generator

DEFAULTS = {
    "candidate_config": REPO / "config/taehyeon/candidates.yaml",
    "virtual_config": REPO / "config/taehyeon/virtual_data.yaml",
    "highlevel_config": REPO / "config/taehyeon/highlevel.yaml",
    "lookahead_config": REPO / "config/taehyeon/lookahead.yaml",
    "environment": REPO / "config/environment.yaml",
}


def add_common_args(parser):
    parser.add_argument("--dataset", type=Path, help="jaesung generator output")
    parser.add_argument("--run-generator", type=int, metavar="PER_FAMILY",
                        help="generate a dataset first (sample mode, N scenarios per family)")
    parser.add_argument("--boxes-per-scenario", type=int, default=80)
    parser.add_argument("--work-dir", type=Path, default=REPO / "tools/highlevel/output")
    for key, path in DEFAULTS.items():
        parser.add_argument("--" + key.replace("_", "-"), type=Path, default=path)


def load_all(args):
    dataset_dir = args.dataset
    if args.run_generator:
        dataset_dir = args.work_dir / f"dataset{args.boxes_per_scenario}_x{args.run_generator}"
        if not (dataset_dir / "manifest.json").exists():
            print(run_generator(team_paths.locate("generator"), team_paths.locate("pac_common"),
                                dataset_dir, "sample", args.run_generator,
                                boxes_per_scenario=args.boxes_per_scenario))
    if dataset_dir is None:
        raise SystemExit("--dataset or --run-generator is required")
    env = load_environment(args.environment)
    cand, vcfg, hl, _ = apply_environment(
        env,
        load_candidate_config(args.candidate_config),
        load_virtual_config(args.virtual_config),
        load_highlevel_config(args.highlevel_config),
    )
    return load_dataset(dataset_dir), cand, vcfg, hl


def load_lookahead(args):
    """Look-ahead config with the environment (N, decision budget) written in."""
    env = load_environment(args.environment)
    return apply_environment(env, la=load_lookahead_config(args.lookahead_config))[3]


def load_env(args):
    return load_environment(args.environment)
