"""Stage-4 episodes from the virtual data (jaesung's scenarios).

Builds a :class:`pac_highlevel.PalletizingWorld` from one generator
scenario with the same measurement model, pallet-size round robin and
height limit as the 5-1/5-2 virtual data, so stage 4 is trained and
evaluated on exactly the same box streams.
"""

import random

from pac_common import Size3D
from pac_highlevel import Arrival, PalletizingWorld

from .episode import select_pallet_xy, select_strength
from .observation import observe
from .scenario_source import build_catalog, stack_height_limit


def family_indices(dataset):
    seen = {}
    out = {}
    for spec in dataset.scenarios:
        out[spec.scenario_id] = seen.get(spec.family, 0)
        seen[spec.family] = out[spec.scenario_id] + 1
    return out


def world_from_spec(
    spec,
    dataset,
    cand_config,
    vcfg,
    hl_config,
    *,
    family_index=0,
    scenario_index=0,
    episode_seed=0,
    value_provider=None,
    placer=None,
    pallet_xy=None,
):
    """Fresh world for one scenario. ``episode_seed`` varies the noise only."""
    rng = random.Random(f"{vcfg.seed}:{spec.scenario_id}:hl:{episode_seed}")
    # same hidden strength draw as the 5-1/5-2 virtual data: damaged boxes
    # found by inspection get a 0 N capacity override when they arrive
    strength = select_strength(spec, vcfg, scenario_index)
    detected = strength.detected if strength is not None else ()
    arrivals = []
    for step, truth in enumerate(spec.arrivals):
        obs = observe(truth, rng, vcfg.observation, float(step))
        arrivals.append(Arrival(obs.box, truth, obs.uncertain, obs.box.box_id in detected))
    xy = pallet_xy or select_pallet_xy(spec, vcfg, family_index)
    size = Size3D(xy[0], xy[1], stack_height_limit(spec, vcfg))
    max_load = (
        float(spec.max_load_kg) if spec.max_load_kg is not None else vcfg.pallet.default_max_load_kg
    )
    catalog = build_catalog(dataset.sku_ranges, vcfg, cand_config.constraints.load_model)
    return PalletizingWorld(
        arrivals,
        size,
        catalog,
        cand_config,
        hl_config,
        pallet_max_weight_kg=max_load,
        pallet_id=spec.pallet_id,
        value_provider=value_provider,
        placer=placer,
    )


def split_ids(dataset, split):
    ids = set(dataset.splits.get(split, ()))
    return [s for s in dataset.scenarios if s.scenario_id in ids]


def world_factory(dataset, specs, cand_config, vcfg, hl_config, *, shuffle_seed=0, vary_pallet=True,
                  value_provider=None, placer=None):
    """``make_world(i)``: episode i cycles over ``specs`` in a seeded order.

    Each pass uses new measurement noise, and with ``vary_pallet`` the pallet
    footprint also rotates over the configured sizes, so the policy sees every
    scenario on every footprint.
    """
    specs = list(specs)
    if not specs:
        raise ValueError("no scenarios for this split")
    fam = family_indices(dataset)
    position = {s.scenario_id: i for i, s in enumerate(dataset.scenarios)}
    sizes = list(vcfg.pallet.sizes_m) or [None]

    def make_world(i):
        epoch, k = divmod(int(i), len(specs))
        order = list(range(len(specs)))
        random.Random(f"{shuffle_seed}:{epoch}").shuffle(order)
        spec = specs[order[k]]
        xy = None
        if vary_pallet and sizes[0] is not None:
            xy = sizes[(fam[spec.scenario_id] + epoch) % len(sizes)]
        return world_from_spec(
            spec, dataset, cand_config, vcfg, hl_config,
            family_index=fam[spec.scenario_id], scenario_index=position[spec.scenario_id],
            episode_seed=epoch, pallet_xy=xy,
            value_provider=value_provider, placer=placer,
        )

    return make_world
