"""Fixed-size observation for the stage-4 policy.

Layout (all roughly in [0, 1]); the names are stored in the policy file and
checked on load so training and deployment use the same features::

    global (8)        fill, max top, weight, box count, free slots,
                      arrivals left, unseen volume, pallets closed
    inventory (4)     unseen count, mean unseen volume, mean unseen weight,
                      heavy share of unseen boxes
    current (6 + K)   present, sorted dims (3), weight, volume, + option K
    slot i (6 + K)    occupied, age, travel time, weight, volume, uncertain
                      + option K                                     x S

Option features (K = 7), per action choice ("선택지별"): feasible, z,
top after, support ratio, valid candidate count, flatness after, future
value from the value provider (proxy or donghan's 5-4 value head).
"""

import numpy as np

OPTION_NAMES = ("feasible", "z", "top_after", "support", "n_valid", "flatness_after", "future")
GLOBAL_NAMES = (
    "fill", "max_top", "weight", "box_count", "free_slots", "arrivals_left", "unseen_volume",
    "pallets_closed",
)
INVENTORY_NAMES = ("unseen_count", "unseen_mean_volume", "unseen_mean_weight", "unseen_heavy_share")
BOX_NAMES = ("present", "dim_a", "dim_b", "dim_c", "weight", "volume")
SLOT_NAMES = ("occupied", "age", "travel", "weight", "volume", "uncertain")
WEIGHT_REF_KG = 30.0
HEAVY_KG = 15.0


def feature_names(slots):
    names = [f"g.{n}" for n in GLOBAL_NAMES] + [f"inv.{n}" for n in INVENTORY_NAMES]
    names += [f"cur.{n}" for n in BOX_NAMES] + [f"cur.opt.{n}" for n in OPTION_NAMES]
    for i in range(slots):
        names += [f"s{i}.{n}" for n in SLOT_NAMES] + [f"s{i}.opt.{n}" for n in OPTION_NAMES]
    return tuple(names)


def _option(opt, H):
    if opt is None or not opt.feasible:
        return [0.0] * len(OPTION_NAMES)
    return [
        1.0,
        opt.z / H,
        opt.top_after / H,
        opt.support_ratio,
        min(1.0, opt.valid_count / 30.0),
        opt.flatness_after,
        float(np.clip(opt.future, -1.0, 2.0)),
    ]


def _box(box, V, Lref):
    if box is None:
        return [0.0] * len(BOX_NAMES)
    dims = sorted((box.size.x, box.size.y, box.size.z), reverse=True)
    return [
        1.0,
        *(d / Lref for d in dims),
        box.weight_kg / WEIGHT_REF_KG,
        box.size.x * box.size.y * box.size.z / V * 20.0,
    ]


def observe(world):
    p = world.pallet_size
    V = world.pallet_volume
    H = p.z
    Lref = max(p.x, p.y)
    placed_vol = sum(b.size.x * b.size.y * b.size.z for b in world.placed)
    top = max(
        (b.pose.z + (b.size.z) for b in world.placed), default=0.0
    )  # yaw-only rotations keep z
    weight = sum(b.weight_kg for b in world.placed)
    free = sum(e is None for e in world.buffer)
    total = max(1, len(world.arrivals))
    unseen = []
    for sku, n in world.remaining_by_sku().items():
        spec = world.catalog.get(sku)
        if spec is not None and n > 0:
            unseen.append((n, spec.size.x * spec.size.y * spec.size.z, spec.weight_kg))
    count = sum(n for n, _, _ in unseen)
    unseen_vol = sum(n * v for n, v, _ in unseen)
    g = [
        placed_vol / V,
        top / H,
        weight / world.max_load,
        min(1.0, len(world.placed) / 100.0),
        free / max(1, world.slots),
        (len(world.arrivals) - world.next_arrival) / total if world.config.features.order_list_known else 0.0,
        min(2.0, unseen_vol / V),
        min(1.0, len(world.closed) / 5.0),
    ]
    inv = [
        min(1.0, count / 100.0),
        (unseen_vol / count / V * 20.0) if count else 0.0,
        (sum(n * w for n, _, w in unseen) / count / WEIGHT_REF_KG) if count else 0.0,
        (sum(n for n, _, w in unseen if w >= HEAVY_KG) / count) if count else 0.0,
    ]
    current, buffered = world.options() if not world.done else (None, [None] * world.slots)
    x = g + inv
    x += _box(world.current.box if world.current else None, V, Lref) + _option(current, H)
    max_travel = max(world.travel) if world.travel else 1.0
    for i, entry in enumerate(world.buffer):
        if entry is None:
            x += [0.0] * (len(SLOT_NAMES) + len(OPTION_NAMES))
            continue
        b = entry.arrival.box
        x += [
            1.0,
            min(1.0, (world.decisions - entry.stored_at) / 20.0),
            world.travel[i] / max_travel,
            b.weight_kg / WEIGHT_REF_KG,
            b.size.x * b.size.y * b.size.z / V * 20.0,
            1.0 if entry.arrival.uncertain else 0.0,
        ]
        x += _option(buffered[i], H)
    return np.asarray(x, dtype=np.float32)
