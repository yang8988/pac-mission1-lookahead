"""Shared helpers for the real-time demo connectors (taehyeon).

The input is the timeline JSON written by
``tools/highlevel/scripts/run_realtime.py`` (``pac_highlevel.realtime.TimedRun``):
one event per robot action with ``t_start``/``t_end`` in timeline seconds,
the planning figures (``compute_s``, ``replans``, ``idle_s``), the action, the
box, its target (pallet frame: corner origin, z = 0 on the deck, min corner of
the rotated AABB) and snapshots of the pallet, buffer and visible conveyor.

This module has no ROS / HTTP dependency: it loads and orders the events,
derives what a connector needs (placement, conveyor window) and formats the
live terminal line. ``Pacer`` maps timeline seconds to wall-clock seconds.
"""

from dataclasses import dataclass, field
import json
import math
from pathlib import Path
import time

PLACING = ("PLACE_CURRENT", "RETRIEVE_BUFFER")
FROM_CONVEYOR = ("PLACE_CURRENT", "BUFFER_CURRENT")


def load_timeline(path):
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if "events" not in data or not isinstance(data["events"], list):
        raise ValueError(f"{path}: not a realtime timeline (no 'events' list)")
    return data


def ordered_events(timeline):
    """Events in time order (stable for equal start times)."""
    events = list(timeline["events"])
    return [e for _, e in sorted(enumerate(events), key=lambda ie: (float(ie[1]["t_start"]), ie[0]))]


def action_kind(action):
    """``RETRIEVE_BUFFER(2)`` -> ``RETRIEVE_BUFFER``."""
    return str(action).split("(", 1)[0]


def buffer_slot(action):
    """Slot index of ``RETRIEVE_BUFFER(i)``, else ``None``."""
    text = str(action)
    if "(" in text and text.endswith(")"):
        return int(text[text.index("(") + 1:-1])
    return None


def is_placing(event):
    return action_kind(event["action"]) in PLACING


def quarter_turns(yaw):
    return int(round(float(yaw) / (math.pi / 2))) % 4


def rotated_size(size, yaw):
    """AABB dims of ``size`` turned by ``yaw`` about z (multiples of 90 deg)."""
    sx, sy, sz = size
    return (sy, sx, sz) if quarter_turns(yaw) % 2 else (sx, sy, sz)


@dataclass
class Placement:
    """One box put on the pallet, in the pallet frame of the timeline."""
    box_id: str
    sku: str
    size: tuple           # box size as measured (x, y, z), before the yaw turn
    weight: float
    min_corner: tuple     # min corner of the rotated AABB
    dims: tuple           # rotated AABB dims
    yaw: float

    @property
    def center(self):
        return tuple(m + d / 2 for m, d in zip(self.min_corner, self.dims))

    def size_and_yaw(self, tol=2e-3):
        """(size, yaw) whose turn reproduces ``dims``; falls back to (dims, 0)
        when the box was tipped or the yaw is not a multiple of 90 deg."""
        on_grid = abs(self.yaw - quarter_turns(self.yaw) * math.pi / 2) < 1e-3
        turned = rotated_size(self.size, self.yaw)
        if on_grid and all(abs(a - b) <= tol for a, b in zip(turned, self.dims)):
            return tuple(self.size), float(self.yaw)
        return tuple(self.dims), 0.0


def placement_of(event):
    """Placement of the event's box, or ``None`` (buffering, or the target
    was not recorded: the box went onto a pallet closed in the same step)."""
    target, box = event.get("target"), event.get("box")
    if not target or not box or not is_placing(event):
        return None
    return Placement(box["box_id"], box.get("sku", ""), tuple(box["size"]), float(box.get("weight", 0.0)),
                     tuple(target["min"]), tuple(target["dims"]), float(target.get("yaw", 0.0)))


def snapshot_placements(event):
    """Pallet snapshot after the event as ``Placement`` rows (size = dims
    turned back by the yaw)."""
    out = []
    for row in event.get("pallet", []):
        dims = tuple(row["dims"])
        out.append(Placement(row["box_id"], row.get("sku", ""), rotated_size(dims, row.get("yaw", 0.0)),
                             float(row.get("weight", 0.0)), tuple(row["min"]), dims, float(row.get("yaw", 0.0))))
    return out


def lost_placement(event):
    """True when the box was placed but its pose is not in the timeline
    (``TimedRun`` records the target after the pallet close emptied it)."""
    return is_placing(event) and not event.get("target") and bool(event.get("pallet_closed"))


def conveyor_window(event, previous=None):
    """Boxes on the conveyor when the action starts: the current box at the
    pick position, then the visible ones upstream (``event['visible']``).

    The current box is the event's box for PLACE/BUFFER_CURRENT; for a
    buffer retrieval it is the box left current by the previous event."""
    rows = []
    if action_kind(event["action"]) in FROM_CONVEYOR and event.get("box"):
        rows.append(event["box"])
    elif previous is not None and previous.get("current"):
        rows.append(previous["current"])
    seen = {r["box_id"] for r in rows}
    rows.extend(r for r in event.get("visible", []) if r["box_id"] not in seen)
    return rows


def _short(box_id):
    return str(box_id).rsplit("-", 1)[-1]


def _size_text(size):
    return "x".join(f"{v:.2f}" for v in size)


def status_line(event, index=None, total=None, extra=""):
    """One terminal line per step: time, action, box, conveyor window,
    planning figures, fill."""
    box = event.get("box") or {}
    head = f"[{index + 1:>3}/{total}] " if index is not None and total else ""
    window = ", ".join(f"{_short(r['box_id'])} {_size_text(r['size'])}" for r in event.get("visible", [])) or "-"
    buf = sum(1 for b in event.get("buffer", []) if b)
    parts = [
        f"{head}t={float(event['t_start']):7.1f}s",
        f"{event['action']:<19}",
        f"{box.get('box_id', '-')} {box.get('sku', '')} {_size_text(box['size']) if box.get('size') else ''}".rstrip(),
        f"| view[{len(event.get('visible', []))}]: {window}",
        f"| compute {float(event.get('compute_s', 0.0)):.3f}s replans {event.get('replans', 0)}"
        f" idle {float(event.get('idle_s', 0.0)):.2f}s",
        f"| buffer {buf} | pallet {event.get('pallet_index', 0)} fill {100 * float(event.get('fill', 0.0)):.1f}%",
    ]
    if event.get("pallet_closed"):
        parts.append("| PALLET CLOSED")
    if event.get("moved"):
        parts.append(f"| repack moved {len(event['moved'])}")
    if extra:
        parts.append(f"| {extra}")
    return " ".join(parts)


def summary_text(timeline):
    s = timeline.get("summary", {})
    return (f"policy {timeline.get('policy', '?')}/{timeline.get('placer', '?')}, horizon {timeline.get('horizon')}, "
            f"{len(timeline['events'])} events, placed {s.get('placed')}/{s.get('boxes')}, "
            f"pallets {s.get('pallets_used')} (closed {s.get('pallets_closed')}), "
            f"makespan {timeline.get('makespan_s')} s, robot idle {timeline.get('robot_idle_s')} s, "
            f"replans {timeline.get('replans')}")


@dataclass
class Pacer:
    """Timeline seconds -> wall clock: event at ``t`` is due at
    ``start + (t - t0) / speed``. ``speed <= 0`` means as fast as possible."""
    speed: float = 1.0
    clock: object = time.monotonic
    sleep: object = time.sleep
    t0: float | None = None
    start: float | None = None
    _log: list = field(default_factory=list)

    def due(self, t):
        if self.t0 is None:
            self.t0, self.start = float(t), self.clock()
        if self.speed <= 0:
            return self.clock()
        return self.start + (float(t) - self.t0) / self.speed

    def wait_until(self, t):
        """Sleep until timeline time ``t``; returns the lag (s, >= 0) when
        the connector is already behind schedule."""
        due = self.due(t)
        now = self.clock()
        if due > now:
            self.sleep(due - now)
            return 0.0
        return now - due


__all__ = ["PLACING", "FROM_CONVEYOR", "load_timeline", "ordered_events", "action_kind", "buffer_slot",
           "is_placing", "rotated_size", "Placement", "placement_of", "snapshot_placements", "lost_placement",
           "conveyor_window", "status_line", "summary_text", "Pacer"]
