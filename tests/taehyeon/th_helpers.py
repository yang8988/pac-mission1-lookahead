"""Shared helpers for taehyeon stage 5-1/5-2 tests.

Teammates' ``pac_common`` is located via scripts/taehyeon/team_paths.py (merged
monorepo location, nested upload, or ``scripts/taehyeon/fetch_team_deps.sh``).
"""

import math
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts" / "taehyeon"))

import team_paths  # noqa: E402

team_paths.bootstrap()
PLANNING = team_paths.add_optional("pac_planning")

from pac_common import (  # noqa: E402
    BoxState,
    BoxStatus,
    InventoryState,
    PalletState,
    PlacedBox,
    PlacementCandidate,
    PlanningContext,
    Pose3D,
    Size3D,
    SkuSpec,
    SystemState,
)

HALF_PI = math.pi / 2
PALLET = Size3D(1.1, 1.1, 1.35)


def make_box(box_id="N", size=(0.4, 0.3, 0.2), weight=5.0, yaws=(0.0, HALF_PI), sku="K"):
    return BoxState(
        box_id,
        sku,
        Size3D(*size),
        weight,
        Pose3D("conveyor", 0.0, 0.0, 0.0),
        tuple(yaws),
        BoxStatus.READY_FOR_PICK,
        1.0,
        0.0,
        "test",
    )


def placed(box_id, x, y, z, size=(0.4, 0.3, 0.2), weight=5.0, yaw=0.0, sku="K"):
    return PlacedBox(box_id, sku, Size3D(*size), weight, Pose3D("pallet", x, y, z, yaw=yaw))


def make_state(boxes=(), version=0, pallet=PALLET, tracked=None):
    return SystemState(
        version,
        0.0,
        PalletState("P001", pallet, tuple(boxes)),
        InventoryState(tracked or {}, {}),
    )


def make_context(capacity_n=1000.0, pallet_max=1000.0, overrides=None, uncertain=(), skus=("K",)):
    catalog = {
        s: SkuSpec(s, Size3D(0.4, 0.3, 0.2), 5.0, (0.0, HALF_PI), capacity_n) for s in skus
    }
    return PlanningContext(
        catalog=catalog,
        pallet_max_weight_kg=pallet_max,
        capacity_overrides_n=overrides or {},
        uncertain_box_ids=tuple(uncertain),
    )


def candidate(box, x, y, z, yaw=0.0, version=0, cid="T"):
    return PlacementCandidate(cid, box.box_id, Pose3D("pallet", x, y, z, yaw=yaw), version)
