"""PARTIAL_REPACK rule (flowchart: Partial Repack, Advanced).

Trigger (applied by the world): no learned action is feasible, i.e. the
current box has no valid candidate and no buffered box can be placed.

Targets: only *accessible* boxes of the current pallet (nothing rests on
them), so moving one never disturbs another box.

Search: uniform-cost / A* with a zero (admissible) heuristic over move
sequences, cheapest first (fewer moves). One move = pick an accessible box
and put it on another hard-mask-valid pose of the same pallet; the goal test
is "the current box now has a valid candidate". Bounded by ``max_moves`` and
``max_nodes`` and accepted only if the net gain (current box volume minus
the time cost of the moves) is at least ``min_gain``. Each move and the
final placement are re-validated by 5-2 in the order they are executed.
The MCTS variant mentioned in the flowchart is not implemented.
"""

from collections import deque
from dataclasses import replace

from pac_common import BoxStatus

from pac_candidates.geometry import rotated_dims


def accessible_ids(world, placed):
    model = world.backend().model_for(world.state(placed))
    supporters = {c.supporter_id for contacts in model.contacts.values() for c in contacts}
    return [p.box_id for p in placed if p.box_id not in supporters]


def _relocations(world, placed, box_id, limit=3):
    moving = next(p for p in placed if p.box_id == box_id)
    rest = [p for p in placed if p.box_id != box_id]
    box = replace(world._box_of(box_id), status=BoxStatus.READY_FOR_PICK)
    state = world.state(rest, extra=(box,))
    cset = world.backend().candidate_set(box, state)
    old_fp = _footprint(moving.pose, moving.size)
    # a move must change where the box is; rotating in place counts when it
    # changes the footprint, an identical footprint is not a move
    valid = [c for c in cset.valid if _footprint(c.target_pose, moving.size) != old_fp]
    valid.sort(key=lambda c: (round(c.target_pose.z, 6), round(c.target_pose.y, 6), round(c.target_pose.x, 6)))
    return rest, moving, valid[:limit]


def plan_repack(world):
    cfg = world.config.repack
    if world.current is None or cfg.max_moves <= 0:
        return None
    gain = world.config.reward.volume_weight * _volume(world.current.box.size) / world.pallet_volume
    current = world.current.box
    queue = deque([(tuple(world.placed), ())])
    expanded = 0
    while queue and expanded < cfg.max_nodes:
        placed, moves = queue.popleft()
        if len(moves) >= cfg.max_moves:
            continue
        moved = {m[0] for m in moves}
        for box_id in accessible_ids(world, list(placed)):
            if box_id in moved or expanded >= cfg.max_nodes:
                continue
            rest, moving, targets = _relocations(world, list(placed), box_id)
            for target in targets:
                expanded += 1
                new_placed = tuple(rest) + (replace(moving, pose=target.target_pose),)
                plan = moves + ((box_id, target.target_pose),)
                cost = (
                    world.config.reward.time_weight
                    * world.config.timing.repack_move_time_s
                    * len(plan)
                )
                state = world.state(list(new_placed))
                cset = world.backend().candidate_set(current, state)
                if cset.valid and gain - cost >= cfg.min_gain:
                    chosen = world.choose_candidate(list(cset.valid), current, state)
                    if chosen is not None and world.backend().validate_constraints(
                        current, chosen, state
                    ).success:
                        return plan, chosen
                queue.append((new_placed, plan))
                if expanded >= cfg.max_nodes:
                    break
    return None


def _footprint(pose, size):
    dx, dy, _ = rotated_dims(size, pose.yaw)
    return tuple(round(v, 6) for v in (pose.x, pose.y, pose.z, dx, dy))


def _volume(size):
    return size.x * size.y * size.z
