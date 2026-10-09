"""Cross-check the 5-2 hard mask with jaesung's PyBullet simulator.

For sampled decision steps the current pallet (TRUE sizes at the planned
centres) is built in ``pac_simulation.ahead_sim.BulletPalletWorld`` and
settled. Then one candidate is dropped and the world is simulated again:

* stable   : every box stays within ``max_shift_m`` and ``max_tilt_deg``
* unstable : the new box or any existing box moved/tilted beyond that

Comparing hard-mask-valid candidates with candidates rejected for
LBCP/support reasons measures whether the mask is (a) safe -- valid ones are
physically stable -- and (b) meaningful -- rejected ones often are not.
The simulator is a rigid-body development model (friction, slatted deck
geometry are placeholders); see jaesung's README.
"""

from dataclasses import dataclass
import math

from pac_candidates.geometry import rotated_dims


@dataclass(frozen=True)
class PhysicsOutcome:
    stable: bool
    new_box_shift_m: float
    new_box_tilt_deg: float
    worst_existing_shift_m: float
    insertion_error: str | None


def _make_world(sim_module, collision_model, pallet_size=None):
    pallet_kwargs = {"collision_model": collision_model}
    if pallet_size is not None:  # match the scene's footprint (T11, T12, ...)
        pallet_kwargs.update(length_m=pallet_size.x, width_m=pallet_size.y)
    cfg = sim_module.SimulatorConfig(pallet=sim_module.PalletConfig(**pallet_kwargs))
    from pac_simulation.ahead_sim.world import BulletPalletWorld

    return BulletPalletWorld(cfg)


def _spec(sim_module, box_id, size, yaw, center, mass):
    return sim_module.BoxSpec(
        box_id=box_id,
        size_m=(size.x, size.y, size.z),
        mass_kg=max(0.05, mass),
        target_position_m=center,
        yaw_rad=yaw,
        source="taehyeon_physics_check",
    )


def _center(pose, planned_size, true_size, pallet_size, true_boxes):
    """Planned centre in x/y; z = drop onto the TRUE tops below.

    The robot places the planned centre and releases on the real surface, so
    height measurement errors of lower boxes shift z, not interpenetrate.
    ``true_boxes``: list of (x0, y0, x1, y1, top) already in the world
    (simulator frame).
    """
    dx, dy, _ = rotated_dims(planned_size, pose.yaw)
    tx, ty, tz = rotated_dims(true_size, pose.yaw)
    cx = pose.x + dx / 2 - pallet_size.x / 2
    cy = pose.y + dy / 2 - pallet_size.y / 2
    x0, y0, x1, y1 = cx - tx / 2, cy - ty / 2, cx + tx / 2, cy + ty / 2
    z = 0.0
    for bx0, by0, bx1, by1, top in true_boxes:
        if min(x1, bx1) - max(x0, bx0) > 1e-6 and min(y1, by1) - max(y0, by0) > 1e-6:
            z = max(z, top)
    true_boxes.append((x0, y0, x1, y1, z + tz))
    return (cx, cy, z + tz / 2)


def _tilt_deg(quat):
    import pybullet as p

    roll, pitch, _ = p.getEulerFromQuaternion(quat)
    return math.degrees(max(abs(roll), abs(pitch)))


def check_candidate(
    sim_module,
    placed,
    truths,
    pallet_size,
    box,
    true_size,
    pose,
    settle_s=1.0,
    observe_s=1.5,
    max_shift_m=0.01,
    max_tilt_deg=2.0,
    collision_model="solid",
):
    import pybullet as p

    world = _make_world(sim_module, collision_model, pallet_size)
    true_boxes = []
    try:
        for b in sorted(placed, key=lambda q: q.pose.z):
            t = truths.get(b.box_id, b.size)
            center = _center(b.pose, b.size, t, pallet_size, true_boxes)
            try:
                world.add_box(_spec(sim_module, b.box_id, t, b.pose.yaw, center, b.weight_kg))
            except ValueError as error:
                return PhysicsOutcome(False, 0.0, 0.0, 0.0, "BASE:" + str(error))
        hz = world.config.physics.physics_hz
        world.step(int(settle_s * hz))
        before = {
            bid: p.getBasePositionAndOrientation(rec.body_id, physicsClientId=world.client_id)[0]
            for bid, rec in world.boxes.items()
        }
        target = _center(pose, box.size, true_size, pallet_size, true_boxes)
        try:
            body = world.add_box(_spec(sim_module, box.box_id, true_size, pose.yaw, target, box.weight_kg))
        except ValueError as error:
            return PhysicsOutcome(False, math.inf, math.inf, 0.0, str(error))
        world.step(int(observe_s * hz))
        pos, quat = p.getBasePositionAndOrientation(body, physicsClientId=world.client_id)
        shift = math.dist(pos[:2], target[:2])
        drop = target[2] - pos[2]
        new_shift = max(shift, abs(drop) - 0.003)
        tilt = _tilt_deg(quat)
        worst = 0.0
        for bid, old in before.items():
            now, q = p.getBasePositionAndOrientation(world.boxes[bid].body_id, physicsClientId=world.client_id)
            worst = max(worst, math.dist(now, old), 0.0 if _tilt_deg(q) < max_tilt_deg else math.inf)
        stable = new_shift <= max_shift_m and tilt <= max_tilt_deg and worst <= max_shift_m
        return PhysicsOutcome(stable, new_shift, tilt, worst, None)
    finally:
        world.close()
