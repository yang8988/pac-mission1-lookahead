"""End-to-end virtual data generation and its summary report.

Output layout (``OUT/``)::

    manifest.json                      config, sources, counts, git commit
    scenes/<split>/Sxxxx-Tyyy.json     donghan planner scenes (pac-common-v0.2+planning-v1)
    candidate_sets/Sxxxx.jsonl         5-1/5-2 labels per decision step
    episodes/Sxxxx.json                placements, final pallet, metrics
    analysis/summary.json|md           aggregate statistics per family
    analysis/steps.csv                 one row per decision step
"""

from collections import Counter, defaultdict
import csv
from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path
import statistics
import subprocess

from pac_common import plain

from .episode import run_episode
from .scenario_source import build_catalog, load_dataset

GENERATOR_NAME = "taehyeon.virtual_data"
VERSION = "1.0.0"


def _dump(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")


def _git_commit(root):
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True, check=True
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return "UNKNOWN"


def make_planner_factory(seed):
    """Policy hook running donghan's planner (5-3~5-6) on our candidates."""
    from pac_planning import PlacementPlanner, PlannerConfig

    def factory(backend, context, cset, box, state):
        def call(valid):
            ctx = backend.context_with_ems(context, cset.generation)
            planner = PlacementPlanner(
                context=ctx,
                config=PlannerConfig(),
                generate_candidates=backend.generate_candidates,
                validate_constraints=backend.validate_constraints,
            )
            try:
                result = planner.plan(
                    box, state, list(cset.generation.candidates), seed=seed, use_time_budget=False
                )
            except ValueError:
                return None
            return result.ranked[0] if result.ranked else None

        return call

    return factory


def generate(dataset_dir, out_dir, cand_config, vcfg, scenario_ids=None, repo_root=None,
             max_scenarios=0, write_labels=True, write_scenes=True, progress=None):
    dataset = load_dataset(dataset_dir)
    # Labels list every mask reason (verdicts are identical in both modes).
    from dataclasses import replace as _replace

    cand_config = _replace(cand_config, collect_all_reasons=True)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    catalog = build_catalog(dataset.sku_ranges, vcfg, cand_config.constraints.load_model)
    planner_factory = make_planner_factory(vcfg.seed) if vcfg.episode.policy == "planner" else None
    run_id = f"{GENERATOR_NAME}-{vcfg.episode.policy}-{vcfg.seed}"

    label_files = {}

    def label_sink(row, scenario_id):
        f = label_files.get(scenario_id)
        if f is None:
            path = out / "candidate_sets" / f"{scenario_id}.jsonl"
            path.parent.mkdir(parents=True, exist_ok=True)
            f = label_files[scenario_id] = path.open("w", encoding="utf-8")
        f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")

    scene_index = []

    def scene_sink(scene, split):
        path = out / "scenes" / split / f"{scene['scenario_id']}.json"
        _dump(path, scene)
        scene_index.append({"scene": str(path.relative_to(out)), "split": split, **scene["source"]})

    episodes = []
    dataset_index = {s.scenario_id: k for k, s in enumerate(dataset.scenarios)}
    family_index = {}
    seen_per_family = Counter()
    for s in dataset.scenarios:  # ordinal of each scenario inside its family
        family_index[s.scenario_id] = seen_per_family[s.family]
        seen_per_family[s.family] += 1
    specs = [s for s in dataset.scenarios if not scenario_ids or s.scenario_id in scenario_ids]
    if max_scenarios:
        specs = specs[:max_scenarios]
    try:
        for i, spec in enumerate(specs):
            split = dataset.split_of(spec.scenario_id)
            result = run_episode(
                spec,
                catalog,
                cand_config,
                vcfg,
                split,
                scene_sink=scene_sink if write_scenes else None,
                label_sink=label_sink if write_labels else None,
                planner_factory=planner_factory,
                run_id=run_id,
                # position in the FULL dataset, so filtering keeps assignments
                scenario_index=dataset_index[spec.scenario_id],
                family_index=family_index[spec.scenario_id],
            )
            episodes.append(result)
            _dump(
                out / "episodes" / f"{spec.scenario_id}.json",
                {
                    "scenario_id": spec.scenario_id,
                    "scenario_family": spec.family,
                    "split": split,
                    "metrics": result.metrics,
                    "unplaced_box_ids": result.unplaced,
                    "final_state": plain(result.final_state),
                    "final_context": plain(result.final_context),
                    "true_sizes": {
                        b.box_id: plain(b.size) for b in spec.arrivals
                    },
                    "true_capacity_n": (
                        dict(result.strength.true_capacity_n) if result.strength else None
                    ),
                    "damaged_box_ids": list(result.strength.damaged) if result.strength else [],
                    "damage_detected_box_ids": (
                        list(result.strength.detected) if result.strength else []
                    ),
                },
            )
            if progress:
                progress(i + 1, len(specs), result)
    finally:
        for f in label_files.values():
            f.close()

    summary = summarize(episodes)
    _dump(out / "analysis" / "summary.json", summary)
    (out / "analysis" / "summary.md").write_text(summary_markdown(summary), encoding="utf-8")
    write_steps_csv(out / "analysis" / "steps.csv", episodes)
    _dump(out / "scenes" / "index.json", scene_index)
    manifest = {
        "schema_version": 1,
        "generator": GENERATOR_NAME,
        "version": VERSION,
        "run_id": run_id,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(repo_root or Path.cwd()),
        "source_dataset": {
            "path": str(Path(dataset_dir).resolve()),
            "run_id": dataset.manifest.get("run_id"),
            "generator_version": dataset.manifest.get("generator_version"),
            "git_commit": dataset.manifest.get("git_commit"),
        },
        "scenario_count": len(episodes),
        "scene_count": len(scene_index),
        "label_step_count": sum(len(e.steps) for e in episodes) if write_labels else 0,
        "virtual_config": asdict(vcfg),
        "candidate_config": plain(cand_config),
        "catalog": plain(catalog),
        "units": {"length": "m", "mass": "kg", "force": "N", "time": "s", "angle": "rad"},
        "contract": {
            "scene_schema": "pac-common-v0.2+planning-v1",
            "pose": "AABB minimum corner after yaw, pallet frame, z=0 loading surface",
            "pallet_size_z": "max stack height above the loading surface",
            "future_order_in_scene": False,
            "state_kind": "SIMULATED",
        },
    }
    _dump(out / "manifest.json", manifest)
    return manifest, summary


def _pct(values, q):
    if not values:
        return 0.0
    ordered = sorted(values)
    k = min(len(ordered) - 1, max(0, int(round(q * (len(ordered) - 1)))))
    return ordered[k]


def _group_stats(eps):
    steps = [s for e in eps for s in e.steps]
    reasons = Counter()
    codes = Counter()
    for s in steps:
        reasons.update(s["reason_counts"])
        codes.update(s["code_counts"])
    total_masked = sum(s["generated"] - s["valid"] for s in steps)
    step_times = [s["generation_sec"] + s["mask_sec"] for s in steps]
    placed_boxes = sum(e.metrics["placed"] for e in eps)
    over = sum(len(e.metrics.get("true_overloaded_boxes", ())) for e in eps)
    mean = lambda xs: statistics.fmean(xs) if xs else 0.0  # noqa: E731
    return {
        "scenarios": len(eps),
        "steps": len(steps),
        "mean_raw_candidates": mean([s["raw"] for s in steps]),
        "mean_generated": mean([s["generated"] for s in steps]),
        "mean_valid": mean([s["valid"] for s in steps]),
        "valid_ratio": sum(s["valid"] for s in steps) / max(1, sum(s["generated"] for s in steps)),
        "no_valid_steps": sum(1 for s in steps if s["valid"] == 0),
        "placed_ratio": mean([e.metrics["placed_ratio"] for e in eps]),
        "pallet_volume_utilization": mean([e.metrics["pallet_volume_utilization"] for e in eps]),
        "bounding_density": mean([e.metrics["bounding_density"] for e in eps]),
        "mean_max_height_m": mean([e.metrics["max_height_m"] for e in eps]),
        "snapshot_issue_boxes": sum(len(e.metrics["snapshot_issues"]) for e in eps),
        "true_overlaps": sum(len(e.metrics["true_overlaps"]) for e in eps),
        "true_protrusions": sum(len(e.metrics["true_protrusions"]) for e in eps),
        "true_overloaded_boxes": over,
        "true_overload_rate": over / max(1, placed_boxes),
        "scenarios_with_true_overload": sum(
            1 for e in eps if e.metrics.get("true_overloaded_boxes")
        ),
        "true_max_load_ratio": max(
            [e.metrics.get("true_max_load_ratio", 0.0) for e in eps], default=0.0
        ),
        "boxes_on_detected_damaged": sum(
            len(e.metrics.get("boxes_on_detected_damaged", ())) for e in eps
        ),
        "step_time_mean_ms": 1e3 * mean(step_times),
        "step_time_p95_ms": 1e3 * _pct(step_times, 0.95),
        "step_time_max_ms": 1e3 * max(step_times) if step_times else 0,
        "reason_share_of_masked": {k: v / max(1, total_masked) for k, v in reasons.most_common()},
        "reject_code_counts": dict(codes.most_common()),
    }


def summarize(episodes):
    by_family = defaultdict(list)
    by_strength = defaultdict(list)
    by_pallet = defaultdict(list)
    for e in episodes:
        by_family[e.family].append(e)
        xy = e.metrics.get("pallet_xy_m")
        if xy:
            by_pallet[f"{xy[0]:.2f}x{xy[1]:.2f}"].append(e)
        if e.metrics.get("strength_profile"):
            by_strength[e.metrics["strength_profile"]].append(e)
    by_family["ALL"] = list(episodes)
    return {
        "families": {k: _group_stats(v) for k, v in sorted(by_family.items())},
        "strength_profiles": {k: _group_stats(v) for k, v in sorted(by_strength.items())},
        "pallet_sizes": {k: _group_stats(v) for k, v in sorted(by_pallet.items())},
    }


def summary_markdown(summary):
    fams = summary["families"]
    lines = [
        "# Virtual data summary (stage 5-1/5-2)",
        "",
        "| family | scen | steps | gen | valid | valid% | no-valid | placed% | util | density | H(m) | issues | true overlap | ms mean/p95 |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for name, f in fams.items():
        lines.append(
            f"| {name} | {f['scenarios']} | {f['steps']} | {f['mean_generated']:.1f} | "
            f"{f['mean_valid']:.1f} | {100 * f['valid_ratio']:.0f} | {f['no_valid_steps']} | "
            f"{100 * f['placed_ratio']:.0f} | {f['pallet_volume_utilization']:.3f} | "
            f"{f['bounding_density']:.3f} | {f['mean_max_height_m']:.2f} | "
            f"{f['snapshot_issue_boxes']} | {f['true_overlaps']} | "
            f"{f['step_time_mean_ms']:.1f}/{f['step_time_p95_ms']:.1f} |"
        )
    if summary.get("strength_profiles"):
        lines += [
            "",
            "## True carton strength (hidden from the planner)",
            "",
            "| strength | scen | placed% | util | H(m) | true-overloaded boxes | rate | scenarios w/ overload | max true load ratio | on detected-damaged |",
            "|---|---|---|---|---|---|---|---|---|---|",
        ]
        for name, f in summary["strength_profiles"].items():
            lines.append(
                f"| {name} | {f['scenarios']} | {100 * f['placed_ratio']:.0f} | "
                f"{f['pallet_volume_utilization']:.3f} | {f['mean_max_height_m']:.2f} | "
                f"{f['true_overloaded_boxes']} | {100 * f['true_overload_rate']:.2f}% | "
                f"{f['scenarios_with_true_overload']} | {f['true_max_load_ratio']:.2f} | "
                f"{f['boxes_on_detected_damaged']} |"
            )
    if summary.get("pallet_sizes"):
        lines += [
            "",
            "## Pallet footprints (m)",
            "",
            "| pallet | scen | steps | valid | no-valid | placed% | util | H(m) | issues | true overlap | true protrusion |",
            "|---|---|---|---|---|---|---|---|---|---|---|",
        ]
        for name, f in summary["pallet_sizes"].items():
            lines.append(
                f"| {name} | {f['scenarios']} | {f['steps']} | {f['mean_valid']:.1f} | "
                f"{f['no_valid_steps']} | {100 * f['placed_ratio']:.0f} | "
                f"{f['pallet_volume_utilization']:.3f} | {f['mean_max_height_m']:.2f} | "
                f"{f['snapshot_issue_boxes']} | {f['true_overlaps']} | {f['true_protrusions']} |"
            )
    allf = fams.get("ALL", {})
    lines += ["", "## Mask reasons (share of masked candidates, ALL)", ""]
    for k, v in allf.get("reason_share_of_masked", {}).items():
        lines.append(f"- {k}: {100 * v:.1f}%")
    return "\n".join(lines) + "\n"


def write_steps_csv(path, episodes):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["scenario_id", "family", "split", "step", "raw", "generated", "valid",
             "generation_ms", "mask_ms", "decision"]
        )
        for e in episodes:
            for s in e.steps:
                writer.writerow(
                    [e.scenario_id, e.family, e.split, s["step"], s["raw"], s["generated"],
                     s["valid"], round(1e3 * s["generation_sec"], 3), round(1e3 * s["mask_sec"], 3),
                     s["decision"]]
                )
