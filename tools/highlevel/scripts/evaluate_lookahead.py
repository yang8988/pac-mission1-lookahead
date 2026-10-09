"""Paired comparison: Rule vs look-ahead search over the next N boxes.

Both policies play the SAME box streams (same pallets, noise and order) of a
dataset split; reported per variant: pallets (pallet equivalents), fill,
robot time, NG, safety issues, decision time and the paired difference with
Rule (better / equal / worse, sign test, 95 % interval).

    python tools/highlevel/scripts/evaluate_lookahead.py \
        --dataset tools/highlevel/output/dataset80_x40_s8 --split val \
        --variant N3=horizon=3 --variant N5=horizon=5 --output docs/taehyeon/reports/lookahead_val.json
"""

import argparse
import json
from math import comb
import multiprocessing as mp
from pathlib import Path
import statistics
import time

from _common import add_common_args, load_all

from pac_highlevel import RulePolicy, run_policy
from pac_highlevel.lookahead import LookaheadPolicy, lookahead_config_from_dict
from virtual_data.highlevel import split_ids, world_factory

_JOB = None
KEEP = ("pallet_equivalents", "fill_per_pallet_used", "time_s", "ng", "safety_issues", "pallets_used")


def _parse_variant(text):
    name, _, body = text.partition("=")
    params = {}
    for item in filter(None, body.split(",")):
        key, _, value = item.partition("=")
        default = getattr(lookahead_config_from_dict({}), key)
        params[key] = type(default)(value) if not isinstance(default, str) else value
    return name, params


def _episode(i):
    factory, hl, variants = _JOB
    row = {"episode": i}
    t = time.perf_counter()
    out = run_policy(factory(i), RulePolicy(hl))
    row["rule"] = {k: out[k] for k in KEEP} | {"wall_s": round(time.perf_counter() - t, 1)}
    for name, params in variants:
        policy = LookaheadPolicy(hl, lookahead_config_from_dict(params))
        t = time.perf_counter()
        out = run_policy(factory(i), policy)
        row[name] = {k: out[k] for k in KEEP} | {"wall_s": round(time.perf_counter() - t, 1),
                                                 "search": policy.stats.summary()}
    return row


def sign_test(better, worse):
    n = better + worse
    if n == 0:
        return 1.0
    k = min(better, worse)
    return min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n)


def summarize(rows, name):
    m = lambda key: statistics.fmean(r[name][key] for r in rows)  # noqa: E731
    out = {
        "pallets_mean": round(m("pallet_equivalents"), 4),
        "fill_mean": round(m("fill_per_pallet_used"), 4),
        "robot_time_s_mean": round(m("time_s"), 1),
        "ng_mean": round(m("ng"), 3),
        "safety_issues": int(sum(r[name]["safety_issues"] for r in rows)),
    }
    if name != "rule":
        d = [r[name]["pallet_equivalents"] - r["rule"]["pallet_equivalents"] for r in rows]
        better = sum(x < -1e-9 for x in d)
        worse = sum(x > 1e-9 for x in d)
        mean = statistics.fmean(d)
        se = statistics.stdev(d) / len(d) ** 0.5 if len(d) > 1 else 0.0
        out["vs_rule"] = {
            "pallets_diff_mean": round(mean, 4),
            "ci95": [round(mean - 1.96 * se, 4), round(mean + 1.96 * se, 4)],
            "better": better, "equal": len(d) - better - worse, "worse": worse,
            "sign_test_p": round(sign_test(better, worse), 4),
        }
        s = [r[name]["search"] for r in rows]
        dec = sum(x["decisions"] for x in s)
        out["search"] = {
            "decision_s_mean": round(statistics.fmean(x["decision_s_mean"] for x in s), 3),
            "decision_s_p95_max": max(x["decision_s_p95"] for x in s),
            "decision_s_max": max(x["decision_s_max"] for x in s),
            "changed_share": round(sum(x["changed"] for x in s) / max(1, dec), 3),
            "timeouts": sum(x["timeouts"] for x in s),
        }
    return out


def main():
    parser = argparse.ArgumentParser()
    add_common_args(parser)
    parser.add_argument("--split", default="val")
    parser.add_argument("--passes", type=int, default=1)
    parser.add_argument("--limit", type=int, default=0, help="first N episodes only")
    parser.add_argument("--variant", action="append", default=[],
                        help="NAME=key=value,key=value (LookaheadConfig fields)")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=4242)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    dataset, cand, vcfg, hl = load_all(args)
    variants = [_parse_variant(v) for v in args.variant] or [("N3", {"horizon": 3})]
    specs = split_ids(dataset, args.split)
    n = len(specs) * args.passes
    if args.limit:
        n = min(n, args.limit)
    global _JOB
    _JOB = (world_factory(dataset, specs, cand, vcfg, hl, shuffle_seed=args.seed), hl, variants)
    rows = []
    with mp.get_context("fork").Pool(args.workers) as pool:
        for row in pool.imap_unordered(_episode, range(n)):
            rows.append(row)
            print(json.dumps({k: (v["pallet_equivalents"] if isinstance(v, dict) else v) for k, v in row.items()}),
                  flush=True)
    rows.sort(key=lambda r: r["episode"])
    report = {
        "split": args.split, "episodes": len(rows), "seed": args.seed,
        "variants": {name: params for name, params in variants},
        "results": {name: summarize(rows, name) for name in ["rule"] + [v[0] for v in variants]},
        "rows": rows,
    }
    print(json.dumps(report["results"], indent=2))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
