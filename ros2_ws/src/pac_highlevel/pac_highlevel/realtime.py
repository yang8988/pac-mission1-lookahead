"""Timed (real-time) run of the stage-4/5 decision loop with a moving conveyor.

The plain simulator is turn based. Here time flows:

  conveyor  box i enters the camera's view at arrival[i]: one box every
            ``interval_s``, but only ``capacity`` boxes fit on the conveyor,
            so a box waits upstream until the robot has picked an earlier one.
  planner   the decision for the next box starts as soon as the robot STARTS
            executing the previous action (on the predicted state = the state
            after that action) and sees the boxes that have arrived by then:
            the current box plus up to N after it. If another box arrives
            while the planner is still computing and the window was not full,
            the planner restarts with the new window ("replan").
            Compute time is measured and added to the timeline (``compute_scale``
            converts it, e.g. 0.5 for a machine twice as fast).
  robot     executes the decision when it is free AND the decision is ready
            AND the box is on the conveyor; any wait is logged as idle time.
            Action durations come from the simulator (pick-and-place, buffer
            travel, repack moves, pallet change).

Every event carries a snapshot (pallet boxes, buffer, visible conveyor boxes)
so a viewer can replay it, and the robot commands needed by the Gazebo replay.
"""

from dataclasses import dataclass
import time

from pac_candidates.geometry import rotated_dims

from .lookahead import LookaheadPolicy


@dataclass
class ConveyorConfig:
    interval_s: float = 6.0     # one box enters the camera view every interval (if there is room)
    capacity: int = 6           # boxes on the conveyor incl. the pick position (current + N visible)
    compute_scale: float = 1.0  # timeline seconds per measured compute second


def _box_row(b):
    return {"box_id": b.box_id, "sku": b.sku_id, "size": [b.size.x, b.size.y, b.size.z],
            "weight": round(b.weight_kg, 3)}


def _placed_rows(world):
    rows = []
    for p in world.placed:
        dx, dy, dz = rotated_dims(p.size, p.pose.yaw)
        rows.append({"box_id": p.box_id, "sku": p.sku_id, "min": [p.pose.x, p.pose.y, p.pose.z],
                     "dims": [dx, dy, dz], "yaw": p.pose.yaw, "weight": round(p.weight_kg, 3)})
    return rows


class TimedRun:
    def __init__(self, world, policy, conveyor=None, horizon=None):
        self.world = world
        self.policy = policy
        self.cfg = conveyor or ConveyorConfig()
        if horizon is None:
            horizon = policy.cfg.horizon if isinstance(policy, LookaheadPolicy) else 0
        self.horizon = int(horizon)
        self.arrival = {}   # index -> time the box entered the view
        self.picked = {}    # index -> time the robot took it off the conveyor
        self.events = []

    # -- conveyor ---------------------------------------------------------
    def arrival_time(self, i):
        """Arrival of box i, or None if it depends on a pick that has not happened."""
        if i in self.arrival:
            return self.arrival[i]
        prev = 0.0 if i == 0 else self.arrival_time(i - 1)
        if prev is None:
            return None
        t = prev + (self.cfg.interval_s if i > 0 else 0.0)
        j = i - self.cfg.capacity
        if j >= 0:
            if j not in self.picked:
                return None
            t = max(t, self.picked[j])
        self.arrival[i] = t
        return t

    def arrived_by(self, t, start, stop):
        """Indices in [start, stop) that are on the conveyor at time t."""
        out = []
        for i in range(start, min(stop, len(self.world.arrivals))):
            a = self.arrival_time(i)
            if a is None or a > t + 1e-9:
                break
            out.append(i)
        return out

    def next_arrival_after(self, t, start, stop):
        for i in range(start, min(stop, len(self.world.arrivals))):
            a = self.arrival_time(i)
            if a is None:
                return None
            if a > t + 1e-9:
                return a
        return None

    # -- loop -------------------------------------------------------------
    def _decide(self, plan_start):
        w = self.world
        first = w.next_arrival  # index of the first box after the current one
        replans, compute = 0, 0.0
        while True:
            seen = self.arrived_by(plan_start, first, first + self.horizon)
            end = first + len(seen)
            t0 = time.perf_counter()
            if isinstance(self.policy, LookaheadPolicy):
                action = self.policy(w, window_end=end)
            else:
                action = self.policy(w)
            c = (time.perf_counter() - t0) * self.cfg.compute_scale
            compute += c
            ready = plan_start + c
            if len(seen) < self.horizon:
                nxt = self.next_arrival_after(plan_start, first, first + self.horizon)
                if nxt is not None and nxt < ready and replans < self.horizon:
                    plan_start, replans = nxt, replans + 1
                    continue
            return action, ready, seen, replans, compute

    def run(self):
        w = self.world
        robot_free = 0.0
        plan_start = 0.0
        while not w.done:
            cur = w.current
            cur_index = w.next_arrival - 1 if cur is not None else None
            if cur_index is not None:
                plan_start = max(plan_start, self.arrival_time(cur_index))
            action, ready, seen, replans, compute = self._decide(plan_start)
            box_ready = self.arrival_time(cur_index) if cur_index is not None else 0.0
            start = max(robot_free, ready, box_ready)
            idle = max(0.0, start - robot_free)
            visible = [_box_row(w.arrivals[i].box) for i in seen]
            closed_before = len(w.closed)
            before_t = w.time_s
            before = {p.box_id: p.pose for p in w.placed}
            label = action.label()
            box = None
            if label == "PLACE_CURRENT" or label == "BUFFER_CURRENT":
                box = cur.box
                self.picked[cur_index] = start
            elif action.slot is not None:
                box = w.buffer[action.slot].arrival.box
            w.step(action)
            duration = w.time_s - before_t
            placed_now = [p for p in w.placed if p.box_id not in before]
            moved = [p.box_id for p in w.placed if p.box_id in before and p.pose != before[p.box_id]]
            target = None
            for p in placed_now:
                if box is not None and p.box_id == box.box_id:
                    dx, dy, dz = rotated_dims(p.size, p.pose.yaw)
                    target = {"min": [p.pose.x, p.pose.y, p.pose.z], "dims": [dx, dy, dz], "yaw": p.pose.yaw}
            self.events.append({
                "t_start": round(start, 3), "t_end": round(start + duration, 3), "idle_s": round(idle, 3),
                "plan_ready": round(ready, 3), "compute_s": round(compute, 3), "replans": replans,
                "action": label, "box": _box_row(box) if box is not None else None, "target": target,
                "pallet_closed": len(w.closed) > closed_before,
                "pallet_index": w.pallet_index, "pallet_size": [w.pallet_size.x, w.pallet_size.y, w.pallet_size.z],
                "moved": moved,
                "visible": visible,
                "current": _box_row(w.current.box) if w.current is not None else None,
                "buffer": [None if e is None else _box_row(e.arrival.box) for e in w.buffer],
                "pallet": _placed_rows(w),
                "fill": round(w.fill(), 4),
            })
            robot_free = start + duration
            plan_start = start  # pipelining: plan the next box while this one is moved
        s = w.summary()
        idle = sum(e["idle_s"] for e in self.events)
        return {
            "summary": {k: s[k] for k in ("boxes", "placed", "ng", "pallets_used", "pallets_closed",
                                          "closed_fill_mean", "pallet_equivalents", "safety_issues")},
            "makespan_s": round(robot_free, 1),
            "robot_idle_s": round(idle, 1),
            "replans": sum(e["replans"] for e in self.events),
            "conveyor": self.cfg.__dict__,
            "horizon": self.horizon,
            "events": self.events,
            "closed_pallets": [p.__dict__ for p in w.closed],
        }


__all__ = ["ConveyorConfig", "TimedRun"]
