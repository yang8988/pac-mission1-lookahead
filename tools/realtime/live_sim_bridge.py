#!/usr/bin/env python3
"""Replay a real-time timeline on jaesung's AHEAD Live Physics Simulator.

    python3 tools/realtime/live_sim_bridge.py reports/realtime_test0.json --speed 2

The simulator (``scripts/run_ahead_simulator.py`` in the team monorepo,
PyBullet + three.js at http://127.0.0.1:4173) exposes, in
``pac_simulation/ahead_sim/server.py``:

    GET  /api/state        full snapshot; ``robot`` = {state, queue, completed, log, ...}
    POST /api/robot_place  queue one placement for the HDR50-22 robot cell
                           (box delivered to PICK, picked by suction, released)
    POST /api/place        place the box directly (no robot)
    POST /api/reset        empty world, robot cell reloaded

Payload (``models.BoxSpec.from_mapping``): ``id``, ``size_m`` (box frame),
``mass_kg`` (> 0), ``target_position_m`` = box CENTRE in the simulator
frame (pallet centre at the origin, deck top z = 0), ``yaw_rad``.
The timeline uses the pallet frame (corner origin, z = 0 on the deck, min
corner of the rotated AABB), so for a timeline pallet X x Y the pallet
frame origin is (-X/2, -Y/2, 0) in the simulator frame. A timeline pallet
that differs from the simulator pallet (1.1 x 1.1 m by default) is centred
on it and a warning is printed.

What the simulator cannot show: the buffer (BUFFER_CURRENT is only
logged), partial-repack moves (no move/remove endpoint; logged) and a
pallet swap (there is no swap endpoint: a closed pallet is replaced by
``/api/reset``, which also reloads the robot cell).

Only the standard library is used for HTTP (urllib).
"""

from dataclasses import dataclass, field
import argparse
import json
from pathlib import Path
import sys
import time
import urllib.error
import urllib.request

sys.path.insert(0, str(Path(__file__).resolve().parent))

from rt_timeline import (  # noqa: E402
    Pacer,
    action_kind,
    load_timeline,
    lost_placement,
    ordered_events,
    placement_of,
    status_line,
    summary_text,
)

SIM_PALLET_XY = (1.10, 1.10)   # live simulator default footprint (config/default.yaml pallet)
MIN_MASS_KG = 0.05


# ---------------------------------------------------------------------- pure
def sim_center(placement, pallet_xy):
    """Pallet-frame placement -> box centre in the simulator frame."""
    cx, cy, cz = placement.center
    return (cx - pallet_xy[0] / 2, cy - pallet_xy[1] / 2, cz)


def sim_payload(placement, pallet_xy, source="ahead_realtime"):
    """JSON body for POST /api/robot_place or /api/place."""
    size, yaw = placement.size_and_yaw()
    return {
        "id": placement.box_id,
        "size_m": [round(float(v), 5) for v in size],
        "mass_kg": round(max(MIN_MASS_KG, float(placement.weight)), 4),
        "target_position_m": [round(float(v), 5) for v in sim_center(placement, pallet_xy)],
        "yaw_rad": round(float(yaw), 6),
        "source": source,
    }


def event_payload(event, pallet_xy=None):
    """Payload for the event's placement, or ``None`` if nothing is placed."""
    placement = placement_of(event)
    if placement is None:
        return None
    pallet_xy = pallet_xy or tuple(event["pallet_size"][:2])
    return sim_payload(placement, pallet_xy, source=f"ahead:{event['action']}")


def pallet_warnings(timeline_size, sim_pallet, tol=1e-3):
    """Differences between the timeline pallet (x, y, stack height) and the
    simulator pallet (``state['pallet']``)."""
    out = []
    sx, sy = float(sim_pallet.get("length_m", SIM_PALLET_XY[0])), float(sim_pallet.get("width_m", SIM_PALLET_XY[1]))
    tx, ty = float(timeline_size[0]), float(timeline_size[1])
    if abs(tx - sx) > tol or abs(ty - sy) > tol:
        out.append(f"timeline pallet {tx:.3f} x {ty:.3f} m differs from the simulator pallet {sx:.3f} x {sy:.3f} m; "
                   "the layout is centred on the simulator pallet"
                   + (" and overhangs it" if tx > sx + tol or ty > sy + tol else ""))
    if len(timeline_size) > 2 and sim_pallet.get("max_height_m") is not None:
        th, sh = float(timeline_size[2]), float(sim_pallet["max_height_m"])
        if th > sh + tol:
            out.append(f"timeline stack height {th:.3f} m exceeds the simulator limit {sh:.3f} m "
                       "(start the simulator with --max-height)")
    return out


def robot_ready(state):
    """(ready, robot dict). Ready = no task running and nothing queued."""
    robot = state.get("robot")
    if robot is None:
        return True, None
    return robot.get("state") == "READY" and not robot.get("queue"), robot


def robot_result(robot, box_id):
    """Outcome of ``box_id`` from the robot snapshot: completed row, a FAILED
    log line, or ``None`` while unknown."""
    for row in robot.get("completed", []):
        if row.get("id") == box_id:
            return f"placed: xy err {row.get('xy_error_mm')} mm, z err {row.get('z_error_mm')} mm, " \
                   f"tilt {row.get('tilt_deg')} deg"
    for line in robot.get("log", []):
        if line.startswith(f"{box_id}:") and "FAILED" in line:
            return line
    return None


# ---------------------------------------------------------------------- HTTP
class SimClient:
    """Minimal urllib client for the live simulator."""

    def __init__(self, url="http://127.0.0.1:4173", timeout=10.0):
        self.url = url.rstrip("/")
        self.timeout = timeout

    def call(self, path, payload=None):
        data = json.dumps(payload).encode() if payload is not None else None
        req = urllib.request.Request(self.url + path, data=data, headers={"Content-Type": "application/json"},
                                     method="POST" if payload is not None else "GET")
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as r:
                return json.loads(r.read().decode())
        except urllib.error.HTTPError as exc:  # the server answers 400 with {"ok": false, "error": ...}
            try:
                return json.loads(exc.read().decode())
            except Exception:
                return {"ok": False, "error": f"HTTP {exc.code}"}

    def state(self):
        return self.call("/api/state")

    def place(self, payload, robot=True):
        return self.call("/api/robot_place" if robot else "/api/place", payload)

    def reset(self):
        return self.call("/api/reset", {})


class DryRunClient:
    """Stands in for the simulator (``--dry-run``): prints the requests and
    reports an idle robot, so the replay can be checked without PyBullet."""

    def __init__(self, out=print):
        self.out = out
        self.sent = []

    def state(self):
        return {"pallet": {"length_m": SIM_PALLET_XY[0], "width_m": SIM_PALLET_XY[1], "max_height_m": 1.5},
                "robot": {"state": "READY", "queue": [], "completed": [{"id": p["id"], "xy_error_mm": 0.0,
                                                                         "z_error_mm": 0.0, "tilt_deg": 0.0}
                                                                        for _, p in self.sent if p],
                          "log": []}}

    def place(self, payload, robot=True):
        self.sent.append(("robot_place" if robot else "place", payload))
        self.out(f"    -> POST /api/{'robot_place' if robot else 'place'} {json.dumps(payload)}")
        return {"ok": True}

    def reset(self):
        self.sent.append(("reset", None))
        self.out("    -> POST /api/reset")
        return {"ok": True}


# ---------------------------------------------------------------------- replay
@dataclass
class BridgeStats:
    sent: int = 0
    placed: int = 0
    failed: list = field(default_factory=list)
    skipped: list = field(default_factory=list)
    resets: int = 0
    max_lag_s: float = 0.0


class LiveSimBridge:
    def __init__(self, client, pacer=None, use_robot=True, ready_timeout_s=180.0, poll_s=0.5,
                 out=print, sleep=time.sleep, clock=time.monotonic):
        self.client = client
        self.pacer = pacer or Pacer()
        self.use_robot = use_robot
        self.ready_timeout_s = ready_timeout_s
        self.poll_s = poll_s
        self.out = out
        self.sleep = sleep
        self.clock = clock
        self.stats = BridgeStats()
        self.pallet_xy = None

    def check_pallet(self, timeline):
        state = self.client.state()
        if self.use_robot and state.get("robot") is None:
            self.out("WARNING: robot cell not available in the simulator; using /api/place (direct placement)")
            self.use_robot = False
        sizes = {tuple(e["pallet_size"]) for e in timeline["events"]}
        for size in sorted(sizes):
            for w in pallet_warnings(size, state.get("pallet", {})):
                self.out(f"WARNING: {w}")
        return state

    def wait_ready(self, box_id=None):
        """Poll /api/state until the robot is idle; returns the outcome text."""
        if not self.use_robot:
            return None
        start = self.clock()
        while True:
            ready, robot = robot_ready(self.client.state())
            if robot is None:
                return None
            if ready:
                return robot_result(robot, box_id) if box_id else None
            if self.clock() - start > self.ready_timeout_s:
                return f"TIMEOUT after {self.ready_timeout_s:.0f} s (robot state {robot.get('state')})"
            self.sleep(self.poll_s)

    def step(self, event, index, total):
        lag = self.pacer.wait_until(event["t_start"])
        self.stats.max_lag_s = max(self.stats.max_lag_s, lag)
        notes = []
        payload = event_payload(event, self.pallet_xy)
        kind = action_kind(event["action"])
        if payload is not None:
            reply = self.client.place(payload, robot=self.use_robot)
            self.stats.sent += 1
            if not reply.get("ok", False):
                notes.append(f"REJECTED by simulator: {reply.get('error')}")
                self.stats.failed.append(payload["id"])
                payload = None
        elif lost_placement(event):
            notes.append("pose not in timeline (pallet closed in this step): box not sent")
            self.stats.skipped.append(event["box"]["box_id"])
        elif kind == "BUFFER_CURRENT":
            notes.append("to buffer (not shown in the simulator)")
        if event.get("moved"):
            notes.append(f"repack of {len(event['moved'])} boxes not reproduced (no move endpoint)")
        if lag > 0.5:
            notes.append(f"behind schedule {lag:.1f} s")
        self.out(status_line(event, index, total, "; ".join(notes)))
        if payload is not None:
            result = self.wait_ready(payload["id"])
            if result is not None:
                self.out(f"    robot: {result}")
                if result.startswith("placed"):
                    self.stats.placed += 1
                else:
                    self.stats.failed.append(payload["id"])
            elif not self.use_robot:
                self.stats.placed += 1
        if event.get("pallet_closed"):
            self.wait_ready()
            self.client.reset()
            self.stats.resets += 1
            self.out(f"    pallet {event.get('pallet_index', 0) - 1} closed -> POST /api/reset "
                     "(the simulator has no pallet-swap endpoint; robot cell reloaded)")

    def run(self, timeline, reset=True, start=0, limit=None):
        self.out(summary_text(timeline))
        self.check_pallet(timeline)
        if reset:
            self.client.reset()
            self.out("POST /api/reset (start from an empty pallet)")
        events = ordered_events(timeline)[start:]
        if limit is not None:
            events = events[:limit]
        for i, event in enumerate(events):
            self.step(event, i, len(events))
        self.wait_ready()
        s = self.stats
        self.out(f"done: sent {s.sent}, placed {s.placed}, failed {len(s.failed)} {s.failed or ''}, "
                 f"skipped {len(s.skipped)} {s.skipped or ''}, resets {s.resets}, max lag {s.max_lag_s:.1f} s")
        return s


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("timeline", type=Path, help="timeline JSON from run_realtime.py")
    ap.add_argument("--url", default="http://127.0.0.1:4173")
    ap.add_argument("--speed", type=float, default=1.0, help="timeline seconds per wall second (0 = no pacing)")
    ap.add_argument("--direct", action="store_true", help="POST /api/place (no robot motion)")
    ap.add_argument("--no-reset", action="store_true", help="do not POST /api/reset before the replay")
    ap.add_argument("--start", type=int, default=0, help="first event index")
    ap.add_argument("--limit", type=int, default=None, help="number of events to replay")
    ap.add_argument("--ready-timeout", type=float, default=180.0, help="max wait for the robot per box (s)")
    ap.add_argument("--poll", type=float, default=0.5, help="robot status poll period (s)")
    ap.add_argument("--dry-run", action="store_true", help="print the requests, no simulator needed")
    return ap.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    timeline = load_timeline(args.timeline)
    client = DryRunClient() if args.dry_run else SimClient(args.url)
    if not args.dry_run:
        try:
            client.state()
        except (urllib.error.URLError, OSError) as exc:
            print(f"ERROR: simulator not reachable at {args.url} ({exc}). Start it with "
                  "'python3 scripts/run_ahead_simulator.py' in the team repo.")
            return 2
    bridge = LiveSimBridge(client, Pacer(speed=args.speed), use_robot=not args.direct,
                           ready_timeout_s=args.ready_timeout, poll_s=args.poll)
    stats = bridge.run(timeline, reset=not args.no_reset, start=args.start, limit=args.limit)
    return 0 if not stats.failed else 1


if __name__ == "__main__":
    sys.exit(main())
