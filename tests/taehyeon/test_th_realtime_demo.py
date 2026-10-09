"""Real-time demo connectors (tools/realtime): timeline helpers, live
simulator bridge and Gazebo replay core, without ROS / PyBullet / network."""

import copy
import json
import math
from pathlib import Path
import sys

import pytest

REPO = Path(__file__).resolve().parents[2]
RT = REPO / "tools" / "realtime"
sys.path.insert(0, str(RT))

import gazebo_replay as gzr  # noqa: E402
import live_sim_bridge as lsb  # noqa: E402
import rt_timeline as tl  # noqa: E402

FIXTURE = RT / "fixtures" / "timeline_sample.json"


def sample():
    return tl.load_timeline(FIXTURE)


def event(action="PLACE_CURRENT", box_id="B1", size=(0.4, 0.3, 0.2), weight=5.0, target=((0.0, 0.0, 0.0), 0.0),
          t=0.0, pallet=(), closed=False, moved=(), visible=(), current=None, pallet_size=(1.2, 0.8, 1.35)):
    box = {"box_id": box_id, "sku": "S1", "size": list(size), "weight": weight}
    tgt = None
    if target is not None:
        corner, yaw = target
        tgt = {"min": list(corner), "dims": list(tl.rotated_size(size, yaw)), "yaw": yaw}
    return {"t_start": t, "t_end": t + 8.0, "idle_s": 0.0, "plan_ready": t, "compute_s": 0.01, "replans": 0,
            "action": action, "box": box, "target": tgt, "pallet_closed": closed, "pallet_index": 0,
            "pallet_size": list(pallet_size), "moved": list(moved), "visible": list(visible), "current": current,
            "buffer": [None, None], "pallet": list(pallet), "fill": 0.1}


def row(box_id, corner, dims, yaw=0.0, weight=5.0):
    return {"box_id": box_id, "sku": "S1", "min": list(corner), "dims": list(dims), "yaw": yaw, "weight": weight}


# ---------------------------------------------------------------------- timeline
def test_fixture_loads_in_time_order():
    data = sample()
    events = tl.ordered_events(data)
    assert len(events) == 12
    assert [e["t_start"] for e in events] == sorted(e["t_start"] for e in events)
    shuffled = dict(data, events=list(reversed(data["events"])))
    assert tl.ordered_events(shuffled) == events


def test_action_parsing():
    assert tl.action_kind("RETRIEVE_BUFFER(2)") == "RETRIEVE_BUFFER"
    assert tl.buffer_slot("RETRIEVE_BUFFER(2)") == 2 and tl.buffer_slot("PLACE_CURRENT") is None


def test_placement_size_and_yaw():
    p = tl.placement_of(event(size=(0.27, 0.18, 0.15), target=((0.1, 0.2, 0.0), math.pi / 2)))
    assert p.dims == (0.18, 0.27, 0.15)
    assert p.center == pytest.approx((0.19, 0.335, 0.075))
    size, yaw = p.size_and_yaw()
    assert size == (0.27, 0.18, 0.15) and yaw == pytest.approx(math.pi / 2)
    # tipped box: dims do not match a turn of the size -> dims with yaw 0
    tipped = tl.Placement("B", "S", (0.4, 0.3, 0.2), 1.0, (0, 0, 0), (0.4, 0.2, 0.3), 0.0)
    assert tipped.size_and_yaw() == ((0.4, 0.2, 0.3), 0.0)
    assert tl.placement_of(event(action="BUFFER_CURRENT", target=None)) is None


def test_lost_placement_on_pallet_close():
    closed = [e for e in sample()["events"] if e["pallet_closed"]]
    assert closed and all(tl.lost_placement(e) for e in closed if e["target"] is None)
    assert not tl.lost_placement(event())


def test_conveyor_window_and_status_line():
    vis = [{"box_id": "S-B7", "sku": "K", "size": [0.3, 0.2, 0.1], "weight": 1.0}]
    ev = event(visible=vis)
    window = tl.conveyor_window(ev)
    assert [r["box_id"] for r in window] == ["B1", "S-B7"]
    prev = event(current={"box_id": "B9", "sku": "K", "size": [0.2, 0.2, 0.2], "weight": 1.0})
    retrieve = event(action="RETRIEVE_BUFFER(0)", visible=vis)
    assert [r["box_id"] for r in tl.conveyor_window(retrieve, prev)] == ["B9", "S-B7"]
    line = tl.status_line(dict(ev, pallet_closed=True), 0, 3, "note")
    for part in ("t=", "PLACE_CURRENT", "B1", "view[1]: B7 0.30x0.20x0.10", "compute 0.010s", "replans 0",
                 "idle 0.00s", "fill 10.0%", "PALLET CLOSED", "note"):
        assert part in line


def test_pacer_scales_time():
    now = [100.0]
    slept = []

    def sleep(s):
        slept.append(s)
        now[0] += s

    pacer = tl.Pacer(speed=2.0, clock=lambda: now[0], sleep=sleep)
    assert pacer.wait_until(10.0) == 0.0          # first event starts now
    assert pacer.wait_until(14.0) == 0.0 and slept == [pytest.approx(2.0)]
    now[0] += 5.0                                   # connector fell behind
    assert pacer.wait_until(16.0) == pytest.approx(4.0)
    fast = tl.Pacer(speed=0, clock=lambda: now[0], sleep=sleep)
    fast.wait_until(0.0)
    assert fast.wait_until(1e6) == 0.0 and len(slept) == 1


# ---------------------------------------------------------------------- live simulator bridge
def test_sim_center_frame():
    p = tl.placement_of(event(target=((0.0, 0.0, 0.0), 0.0)))
    # pallet frame origin is (-X/2, -Y/2) in the simulator frame; target is the box centre
    assert lsb.sim_center(p, (1.1, 1.1)) == pytest.approx((-0.35, -0.4, 0.1))
    p2 = tl.placement_of(event(target=((0.7, 0.8, 0.2), 0.0)))
    assert lsb.sim_center(p2, (1.1, 1.1)) == pytest.approx((0.35, 0.4, 0.3))


def test_sim_payload_matches_boxspec_contract():
    ev = event(size=(0.27, 0.18, 0.15), weight=0.0, target=((0.0, 0.0, 0.3), math.pi / 2))
    payload = lsb.event_payload(ev, (1.1, 1.1))
    assert payload["id"] == "B1" and payload["size_m"] == [0.27, 0.18, 0.15]
    assert payload["yaw_rad"] == pytest.approx(math.pi / 2)
    assert payload["mass_kg"] > 0                      # BoxSpec rejects mass <= 0
    assert payload["target_position_m"] == pytest.approx([0.09 - 0.55, 0.135 - 0.55, 0.375])
    json.dumps(payload)
    assert lsb.event_payload(event(action="BUFFER_CURRENT", target=None)) is None
    # default: the event's own pallet size (1.2 x 0.8)
    assert lsb.event_payload(event())["target_position_m"][:2] == pytest.approx([0.2 - 0.6, 0.15 - 0.4])


def test_pallet_warnings():
    sim = {"length_m": 1.1, "width_m": 1.1, "max_height_m": 1.5}
    assert lsb.pallet_warnings((1.1, 1.1, 1.5), sim) == []
    w = lsb.pallet_warnings((1.2, 0.8, 1.35), sim)
    assert len(w) == 1 and "overhangs" in w[0]
    assert any("stack height" in x for x in lsb.pallet_warnings((1.1, 1.1, 1.8), sim))


def test_robot_status_helpers():
    assert lsb.robot_ready({"robot": {"state": "READY", "queue": []}})[0]
    assert not lsb.robot_ready({"robot": {"state": "lift", "queue": []}})[0]
    assert not lsb.robot_ready({"robot": {"state": "READY", "queue": ["B2"]}})[0]
    robot = {"completed": [{"id": "B1", "xy_error_mm": 1.5, "z_error_mm": -2.0, "tilt_deg": 0.1}],
             "log": ["B2: FAILED to plan (PICK approach unreachable)"]}
    assert lsb.robot_result(robot, "B1").startswith("placed")
    assert "FAILED" in lsb.robot_result(robot, "B2")
    assert lsb.robot_result(robot, "B3") is None


class FakeSim:
    """Robot cell that needs two polls per box."""

    def __init__(self, reject=()):
        self.calls, self.pending, self.done, self.reject = [], [], [], set(reject)

    def state(self):
        self.calls.append(("state",))
        robot = {"state": "READY", "queue": [], "completed": list(self.done), "log": []}
        if self.pending:
            box = self.pending.pop(0)
            robot.update(state="lift", queue=[])
            self.done.append({"id": box, "xy_error_mm": 0.0, "z_error_mm": 0.0, "tilt_deg": 0.0})
        return {"pallet": {"length_m": 1.1, "width_m": 1.1, "max_height_m": 1.5}, "robot": robot}

    def place(self, payload, robot=True):
        self.calls.append(("place", payload["id"], robot))
        if payload["id"] in self.reject:
            return {"ok": False, "error": "duplicate box id"}
        self.pending.append(payload["id"])
        return {"ok": True}

    def reset(self):
        self.calls.append(("reset",))
        return {"ok": True}


def test_bridge_replays_fixture():
    data = sample()
    sim = FakeSim(reject={"S0190-B003"})
    lines = []
    bridge = lsb.LiveSimBridge(sim, tl.Pacer(speed=0), out=lines.append, sleep=lambda s: None)
    stats = bridge.run(data)
    places = [c[1] for c in sim.calls if c[0] == "place"]
    expected = [e["box"]["box_id"] for e in tl.ordered_events(data) if tl.placement_of(e)]
    assert places == expected
    assert stats.sent == len(expected) and stats.failed == ["S0190-B003"]
    assert stats.placed == len(expected) - 1
    resets = [i for i, c in enumerate(sim.calls) if c[0] == "reset"]
    assert len(resets) == 1 + sum(e["pallet_closed"] for e in data["events"])
    assert stats.skipped == [e["box"]["box_id"] for e in data["events"] if tl.lost_placement(e)]
    assert any("WARNING" in line and "1.200 x 0.800" in line for line in lines)
    assert sum("robot: placed" in line for line in lines) == stats.placed


def test_bridge_falls_back_to_direct_without_robot_cell():
    class NoRobot(FakeSim):
        def state(self):
            return {"pallet": {"length_m": 1.1, "width_m": 1.1}, "robot": None}

    sim = NoRobot()
    bridge = lsb.LiveSimBridge(sim, tl.Pacer(speed=0), out=lambda s: None)
    bridge.run({"events": [event(pallet_size=(1.1, 1.1, 1.5))]}, reset=False)
    assert sim.calls == [("place", "B1", False)] and bridge.stats.placed == 1


def test_bridge_dry_run_cli(capsys):
    assert lsb.main([str(FIXTURE), "--dry-run", "--speed", "0", "--limit", "3"]) == 0
    out = capsys.readouterr().out
    assert out.count("POST /api/robot_place") == 3 and "[  3/3]" in out


# ---------------------------------------------------------------------- Gazebo replay core
@pytest.fixture(scope="module")
def team():
    try:
        gd = gzr.load_team()
        config = gzr.default_robot_config()
    except Exception as exc:  # pragma: no cover - team checkout missing
        pytest.skip(f"team packages not available: {exc}")
    if config is None:  # pragma: no cover
        pytest.skip("robot_check_gazebo.yaml not found")
    return gd, gzr.make_robot(config)


def test_conveyor_layout_upstream():
    rows = [{"box_id": "A", "size": [0.4, 0.3, 0.2]}, {"box_id": "B", "size": [0.2, 0.2, 0.1]}]
    out = gzr.conveyor_layout(rows, (0.0, 1.2, 0.9), gap=0.1)
    (sa, pa, _), (sb, pb, _) = out["conv_A"], out["conv_B"]
    assert pa == (0.0, 1.2, 1.0, 0.0)
    assert pb[0] == pytest.approx(-0.2 - 0.1 - 0.1) and pb[2] == pytest.approx(0.95)


def test_deck_warning():
    assert gzr.deck_warning((1.2, 1.0, 1.5)) is None
    assert "overhang" not in gzr.deck_warning((1.2, 0.8, 1.35))
    assert "overhang" in gzr.deck_warning((1.1, 1.1, 1.5))


def test_gazebo_core_reachable_placement_has_joints(team):
    gd, robot = team
    ev = event(size=(0.2206, 0.1894, 0.0891), weight=0.345, target=((0.002, 0.002, 0.0), 0.0))
    core = gzr.GazeboReplayCore(gd, robot, ev["pallet_size"])
    plan = core.step(ev)
    cmd = plan.commands[0]
    assert cmd["action"] == "PLACE_CURRENT" and cmd["box_id"] == "B1"
    assert cmd["target_min_corner"] == [0.002, 0.002, 0.0, 0.0]
    assert len(cmd["robot"]["q_place"]) == 6 and len(cmd["robot"]["q_approach"]) == 6
    acts = plan.actions[0]
    assert acts.trajectory and acts.trajectory[-1][0] == pytest.approx(cmd["robot"]["q_approach"])
    name, sdf, pose = acts.spawn[0]
    # pallet 1.2 x 0.8 centred at (1.35, -1.0), deck top 0.15
    assert pose == pytest.approx((1.35 - 0.6 + 0.002 + 0.1103, -1.0 - 0.4 + 0.002 + 0.0947, 0.15 + 0.04455, 0.0))
    assert name == "B1" and "<box><size>" in sdf
    assert plan.conveyor_spawn[0][0] == "conv_B1" and plan.picked == ["conv_B1"]


def test_gazebo_core_unreachable_spawns_without_motion(team):
    gd, _ = team
    core = gzr.GazeboReplayCore(gd, None, (1.2, 0.8, 1.35), conveyor=False)
    plan = core.step(event())
    assert plan.commands[0]["robot"] is None and plan.commands[0]["robot_reject"] == ["NO_ROBOT_CHECK"]
    assert not plan.actions[0].trajectory and plan.actions[0].spawn[0][0] == "B1"
    assert core.unreachable == ["B1"] and "spawned without motion" in plan.notes[0]


def test_gazebo_core_repack_and_close(team):
    gd, _ = team
    core = gzr.GazeboReplayCore(gd, None, (1.2, 0.8, 1.35), conveyor=False)
    first = event(box_id="A", pallet=[row("A", (0, 0, 0), (0.4, 0.3, 0.2))])
    core.step(first)
    second = event(box_id="B", target=((0.4, 0.0, 0.0), 0.0), moved=["A"],
                   pallet=[row("A", (0.0, 0.3, 0.0), (0.4, 0.3, 0.2)), row("B", (0.4, 0, 0), (0.4, 0.3, 0.2))])
    plan = core.step(second)
    assert [c["action"] for c in plan.commands] == ["PLACE_CURRENT", "PARTIAL_REPACK"]
    assert plan.commands[1]["repack"][0]["target_min_corner"] == [0.0, 0.3, 0.0, 0.0]
    assert plan.actions[1].remove == ["A"] and plan.actions[1].spawn[0][0] == "A"
    closing = event(box_id="C", target=None, closed=True, pallet=[])
    plan = core.step(closing)
    assert [c["action"] for c in plan.commands] == ["PALLET_CLOSE"]
    assert sorted(plan.actions[0].remove) == ["A", "B"] and core.driver.on_pallet == []
    assert "pose not in timeline" in plan.notes[0]


def test_gazebo_core_fixture_dry_run(team):
    gd, robot = team
    data = sample()
    core = gzr.GazeboReplayCore(gd, robot, data["events"][0]["pallet_size"])
    io = gzr.PrintIO(out=lambda s: None)
    lines = []
    gzr.play(core, copy.deepcopy(data), io, tl.Pacer(speed=0), out=lines.append, motion_wait=False)
    assert len(lines) == 13 and lines[-1].startswith("done")
    # after the close only the boxes of the new pallet are on it
    last = tl.ordered_events(data)[-1]
    assert sorted(core.driver.on_pallet) == sorted(r["box_id"] for r in last["pallet"])


# ---------------------------------------------------------------------- HTML viewer
def test_export_viewer_html(tmp_path):
    import export_viewer as ev

    data = sample()
    data["events"][0]["box"]["sku"] = "</script><b>"   # must not break out of the data script
    html = ev.render_html(data)
    assert ev.THREE_URL in html and "three.js/r128/" in html
    assert html.count("</script>") == 2 and "__DATA__" not in html
    payload = json.loads(html.split("const DATA = ", 1)[1].split(";\nconst EV", 1)[0].replace("<\\/", "</"))
    assert len(payload["events"]) == 12 and payload["events"][0]["box"]["sku"] == "</script><b>"
    out = tmp_path / "v.html"
    assert ev.main([str(FIXTURE), str(out)]) == 0 and out.stat().st_size > 10000
