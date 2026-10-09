"""One environment file for the whole algorithm (``config/environment.yaml``).

Everything that describes the cell and the goods lives here: pallet
footprints, cargo height and load, buffer slots, how many conveyor boxes the
camera sees, robot timings and the packaging of each SKU. The algorithm
configs (candidates / highlevel / lookahead / virtual data) keep only
algorithm settings; ``apply_environment`` writes the environment into them,
so a new site only needs a new environment file.

Keys left out (or 0 / empty) keep what the other configs or the data say.
"""

from dataclasses import dataclass, field, replace
from pathlib import Path


@dataclass(frozen=True)
class Environment:
    pallet_sizes_m: tuple = ()          # [[x, y], ...]; empty = virtual_data / data
    max_stack_height_m: float = 0.0     # cargo height above the deck; 0 = data
    max_load_kg: float = 0.0            # 0 = virtual_data default
    buffer_slots: int = -1              # -1 = highlevel.yaml
    visible_boxes: int = -1             # conveyor boxes after the current one the camera sees (N)
    conveyor_interval_s: float = 0.0
    cycle_s: float = 0.0                # pick-and-place cycle of the robot
    buffer_travel_s: float = 0.0
    pallet_change_s: float = 0.0
    repack_move_s: float = 0.0
    decision_budget_ratio: float = -1.0  # search time per decision = ratio x cycle
    orientation: str = ""               # "sku": each SKU's allowed yaws | "config": candidates.yaml yaw_set
    packaging_default: str = ""
    packaging: dict = field(default_factory=dict)
    rated_top_load_n: dict = field(default_factory=dict)


def environment_from_dict(data):
    d = dict((data or {}).get("environment", data or {}))
    pallet = d.get("pallet", {}) or {}
    conveyor = d.get("conveyor", {}) or {}
    robot = d.get("robot", {}) or {}
    catalog = d.get("catalog", {}) or {}
    return Environment(
        pallet_sizes_m=tuple(tuple(float(v) for v in xy) for xy in pallet.get("sizes_m", ()) or ()),
        max_stack_height_m=float(pallet.get("max_stack_height_m", 0.0) or 0.0),
        max_load_kg=float(pallet.get("max_load_kg", 0.0) or 0.0),
        buffer_slots=int((d.get("buffer", {}) or {}).get("slots", -1)),
        visible_boxes=int(conveyor.get("visible_boxes", -1)),
        conveyor_interval_s=float(conveyor.get("interval_s", 0.0) or 0.0),
        cycle_s=float(robot.get("cycle_s", 0.0) or 0.0),
        buffer_travel_s=float(robot.get("buffer_travel_s", 0.0) or 0.0),
        pallet_change_s=float(robot.get("pallet_change_s", 0.0) or 0.0),
        repack_move_s=float(robot.get("repack_move_s", 0.0) or 0.0),
        decision_budget_ratio=float(robot.get("decision_budget_ratio", -1.0)),
        orientation=str(catalog.get("orientation", "") or ""),
        packaging_default=str(catalog.get("packaging_default", "") or ""),
        packaging=dict(catalog.get("packaging", {}) or {}),
        rated_top_load_n=dict(catalog.get("rated_top_load_n", {}) or {}),
    )


def load_environment(path=None):
    if path is None or not Path(path).exists():
        return Environment()
    import yaml

    return environment_from_dict(yaml.safe_load(Path(path).read_text(encoding="utf-8")))


def apply_environment(env, cand=None, vcfg=None, hl=None, la=None):
    """Return (cand, vcfg, hl, la) with the environment written in."""
    if cand is not None and env.orientation == "sku":
        cand = replace(cand, generation=replace(cand.generation, yaw_set_rad=()))
    if vcfg is not None:
        pallet = vcfg.pallet
        if env.pallet_sizes_m:
            pallet = replace(pallet, sizes_m=env.pallet_sizes_m)
        if env.max_stack_height_m > 0:
            pallet = replace(pallet, max_stack_height_m=env.max_stack_height_m)
        if env.max_load_kg > 0:
            pallet = replace(pallet, default_max_load_kg=env.max_load_kg)
        catalog = vcfg.catalog
        if env.packaging_default or env.packaging or env.rated_top_load_n:
            catalog = replace(
                catalog,
                packaging_default=env.packaging_default or catalog.packaging_default,
                packaging={**catalog.packaging, **env.packaging},
                rated_top_load_n={**catalog.rated_top_load_n, **env.rated_top_load_n},
            )
        vcfg = replace(vcfg, pallet=pallet, catalog=catalog)
    if hl is not None:
        buffer, timing = hl.buffer, hl.timing
        if env.buffer_slots >= 0:
            buffer = replace(buffer, slots=env.buffer_slots, travel_time_s=())
        if env.buffer_travel_s > 0:
            buffer = replace(buffer, base_travel_time_s=env.buffer_travel_s)
        if env.cycle_s > 0:
            timing = replace(timing, place_time_s=env.cycle_s)
        if env.pallet_change_s > 0:
            timing = replace(timing, pallet_change_time_s=env.pallet_change_s)
        if env.repack_move_s > 0:
            timing = replace(timing, repack_move_time_s=env.repack_move_s)
        hl = replace(hl, buffer=buffer, timing=timing)
    if la is not None:
        if env.visible_boxes >= 0:
            la = replace(la, horizon=env.visible_boxes)
        if env.decision_budget_ratio >= 0:
            la = replace(la, time_budget_ratio=env.decision_budget_ratio, time_budget_s=0.0)
    return cand, vcfg, hl, la


__all__ = ["Environment", "apply_environment", "environment_from_dict", "load_environment"]
