"""Virtual cell for the 1 -> 8 runtime loop (``pac_runtime``) from one
generator scenario: same pallet sizes, height limit, catalog and hidden box
strength (damage) as the 5-1/5-2 virtual data, plus field anomalies:

- ``spec_mismatch_probability``: the box really differs from its SKU size
  (wrong carton): one dimension scaled by 0.85 or 1.15
- ``missing_probability``: an ordered box never arrives (order list keeps it)
"""

from collections import Counter
from dataclasses import replace
import random

from pac_common import Size3D

from .episode import select_pallet_xy, select_strength
from .scenario_source import build_catalog, stack_height_limit


def runtime_cell(spec, dataset, cand_config, vcfg, *, family_index=0, scenario_index=0,
                 spec_mismatch_probability=0.02, missing_probability=0.02, seed=0, pallet_xy=None):
    from pac_runtime import CellSpec, FieldBox

    rng = random.Random(f"{vcfg.seed}:{spec.scenario_id}:runtime:{seed}")
    strength = select_strength(spec, vcfg, scenario_index)
    damaged = set(strength.damaged) if strength is not None else set()
    capacity = dict(strength.true_capacity_n) if strength is not None else {}
    stream = []
    for truth in spec.arrivals:
        if rng.random() < missing_probability:
            continue
        if rng.random() < spec_mismatch_probability:
            axis = rng.choice("xyz")
            factor = rng.choice((0.85, 1.15))
            s = truth.size
            truth = replace(truth, size=Size3D(*(round(getattr(s, a) * (factor if a == axis else 1.0), 4)
                                                 for a in "xyz")))
        stream.append(FieldBox(truth, truth.box_id in damaged, capacity.get(truth.box_id)))
    xy = pallet_xy or select_pallet_xy(spec, vcfg, family_index)
    catalog = build_catalog(dataset.sku_ranges, vcfg, cand_config.constraints.load_model)
    max_load = float(spec.max_load_kg) if spec.max_load_kg is not None else vcfg.pallet.default_max_load_kg
    return CellSpec(
        stream=tuple(stream),
        expected_by_sku=dict(Counter(b.sku_id for b in spec.arrivals)),
        pallet_size=Size3D(xy[0], xy[1], stack_height_limit(spec, vcfg)),
        catalog=catalog,
        weight_ranges={k: (r.weight_min_kg, r.weight_max_kg) for k, r in dataset.sku_ranges.items()},
        pallet_max_weight_kg=max_load,
        pallet_prefix=spec.pallet_id,
    )
