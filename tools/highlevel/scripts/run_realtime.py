"""Timed run of one scenario: moving conveyor, planning during robot motion.

    python tools/highlevel/scripts/run_realtime.py --dataset tools/highlevel/output/dataset80_x40_s8 \
        --split test --episode 0 --output reports/realtime_test0.json

Writes the timeline (events with pallet/buffer/conveyor snapshots) for the
3D viewer and the Gazebo replay; prints makespan, robot idle time and replans.
"""

import argparse
import json
from pathlib import Path

from _common import add_common_args, load_all, load_env, load_lookahead

from pac_highlevel import RulePolicy
from pac_highlevel.lookahead import LookaheadPolicy
from pac_highlevel.placement import make_placer
from pac_highlevel.realtime import ConveyorConfig, TimedRun
from virtual_data.highlevel import split_ids, world_factory


def main():
    parser = argparse.ArgumentParser()
    add_common_args(parser)
    parser.add_argument("--policy", choices=("lookahead", "rule"), default="lookahead")
    parser.add_argument("--split", default="test")
    parser.add_argument("--episode", type=int, default=0)
    parser.add_argument("--interval", type=float, default=0.0,
                        help="seconds between boxes entering the camera view (0 = environment file)")
    parser.add_argument("--capacity", type=int, default=0, help="boxes on the conveyor (0 = horizon + 1)")
    parser.add_argument("--compute-scale", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=4242)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    dataset, cand, vcfg, hl = load_all(args)
    la = load_lookahead(args)
    env = load_env(args)
    interval = args.interval or env.conveyor_interval_s or 6.0
    placer = make_placer(la.placer) if args.policy == "lookahead" else None
    world = world_factory(dataset, split_ids(dataset, args.split), cand, vcfg, hl,
                          shuffle_seed=args.seed, placer=placer)(args.episode)
    policy = LookaheadPolicy(hl, la) if args.policy == "lookahead" else RulePolicy(hl)
    horizon = la.horizon if args.policy == "lookahead" else 0
    conveyor = ConveyorConfig(interval, args.capacity or la.horizon + 1, args.compute_scale)
    out = TimedRun(world, policy, conveyor, horizon=horizon).run()
    out["policy"] = args.policy
    out["placer"] = la.placer if args.policy == "lookahead" else "dblf"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out), encoding="utf-8")
    print(json.dumps({k: out[k] for k in ("summary", "makespan_s", "robot_idle_s", "replans")}, indent=1))


if __name__ == "__main__":
    main()
