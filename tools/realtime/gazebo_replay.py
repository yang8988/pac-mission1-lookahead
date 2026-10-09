#!/usr/bin/env python3
"""Replay a real-time timeline in the pac2026-ahead Gazebo workcell.

    python3 tools/realtime/gazebo_replay.py reports/realtime_test0.json --speed 1

Gazebo must already run the HDR50-22 pedestal workcell of the team repo
(``tools/runtime/launch/hdr50_pedestal_workcell.launch.py``: world
``ahead_workcell_v2``, robot base at world (1.35, 0.15, 0.5), pallet centre
(1.35, -1.0), deck top z = 0.15).

The plant side reuses the team's ``pac_runtime/gazebo_driver.py`` read-only:
``GazeboReplayCore`` turns each timeline event into the command dicts that
``GazeboDriverCore.on_command`` expects (``action``, ``state_version``,
``box_id``, ``target_min_corner`` [x, y, z, yaw] in the pallet frame,
``robot`` {q_place, q_approach} from stage 6 ``RobotFeasibility`` with
``config/taehyeon/robot_check_gazebo.yaml``) and returns the driver's
``Actions`` (joint trajectory, boxes to spawn / remove). A placement that
stage 6 rejects is logged and its box is spawned without robot motion.

Per event: PLACE_CURRENT / RETRIEVE_BUFFER -> trajectory + spawn at the
target; ``moved`` boxes -> PARTIAL_REPACK (re-spawned at the snapshot pose);
``pallet_closed`` -> PALLET_CLOSE (all pallet boxes removed). The conveyor
window (current box at the pick point + the visible boxes upstream) is
shown as static boxes on ``conveyor_main``.

``main`` needs ROS 2 (rclpy, trajectory_msgs) and ``ros2 run ros_gz_sim
create``; ``--dry-run`` prints the plan without ROS. Everything else is pure
and unit-tested.
"""

from dataclasses import dataclass, field
import argparse
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(HERE))

from rt_timeline import (  # noqa: E402
    FROM_CONVEYOR,
    Pacer,
    action_kind,
    conveyor_window,
    load_timeline,
    lost_placement,
    ordered_events,
    placement_of,
    snapshot_placements,
    status_line,
    summary_text,
)

JOINTS = ["j1", "j2", "j3", "j4", "j5", "j6"]
DEFAULT_PICK_WORLD = (0.0, 1.2, 0.9)   # conveyor_main end under the pick point (robot_check_gazebo.yaml)
GAZEBO_DECK_XY = (1.20, 1.00)        # pallet_main in ahead_workcell_v2_hdp160.sdf
CONVEYOR_GAP_M = 0.08
CONVEYOR_PREFIX = "conv_"


# ---------------------------------------------------------------------- team packages
def team_source(name):
    """Source dir of a team package: team_paths, else a sibling of pac_common."""
    sys.path.insert(0, str(REPO / "scripts" / "taehyeon"))
    import team_paths

    path = team_paths.locate(name)
    if path is None:
        common = team_paths.locate("pac_common")
        if common is not None and (common.parent / name).exists():
            path = common.parent / name
    return path


def default_robot_config():
    """config/taehyeon/robot_check_gazebo.yaml of this repo or of the team
    checkout that holds pac_common (``<root>/ros2_ws/src/pac_common``)."""
    roots = [REPO]
    common = team_source("pac_common")
    if common is not None:
        roots.append(Path(common).parents[2])
    for root in roots:
        path = root / "config" / "taehyeon" / "robot_check_gazebo.yaml"
        if path.exists():
            return path
    return None


def load_team():
    """Import pac_common / pac_robot_check and the team gazebo driver module.

    ``gazebo_driver.py`` is loaded from its file so the ``pac_runtime``
    package ``__init__`` (which needs the whole runtime stack) is not run."""
    sys.path.insert(0, str(REPO / "scripts" / "taehyeon"))
    import team_paths

    team_paths.bootstrap(require_common=True)
    rc = team_source("pac_robot_check")
    if rc is not None and str(rc) not in sys.path:
        sys.path.append(str(rc))
    import pac_robot_check  # noqa: F401  (fails clearly when missing)

    runtime = team_source("pac_runtime")
    driver_file = runtime / "pac_runtime" / "gazebo_driver.py" if runtime is not None else None
    if driver_file is not None and driver_file.exists():
        spec = importlib.util.spec_from_file_location("pac_team_gazebo_driver", driver_file)
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        return module
    from pac_runtime import gazebo_driver  # installed ROS workspace

    return gazebo_driver


def make_robot(config_path):
    from pac_robot_check import RobotFeasibility, load_robot_check_config

    return RobotFeasibility(load_robot_check_config(config_path))


# ---------------------------------------------------------------------- pure core
def pick_point_world(robot, cell):
    """Conveyor pick point (box bottom centre) in the Gazebo world, from the
    stage-6 cell config (robot base relative to the pallet centre)."""
    if robot is None:
        return DEFAULT_PICK_WORLD
    c = robot.cfg.cell
    bx, by, bz, _ = c.base_from_pallet_center
    px, py, pz = c.pick_point_base_m
    base = (cell.pallet_center_xy[0] + bx, cell.pallet_center_xy[1] + by, cell.deck_top_z + bz)
    return (base[0] + px, base[1] + py, base[2] + pz)


def conveyor_layout(rows, pick, gap=CONVEYOR_GAP_M):
    """Static conveyor models: first row centred on the pick point, the next
    ones upstream (-x, the conveyor runs towards +x). name -> (size, pose)."""
    out = {}
    x_front = None
    for row in rows:
        sx, sy, sz = (float(v) for v in row["size"])
        if x_front is None:
            x = pick[0]
        else:
            x = x_front - gap - sx / 2
        x_front = x - sx / 2
        out[CONVEYOR_PREFIX + row["box_id"]] = ((sx, sy, sz), (round(x, 4), pick[1], round(pick[2] + sz / 2, 4), 0.0),
                                                float(row.get("weight", 1.0)))
    return out


def deck_warning(pallet_size, deck=GAZEBO_DECK_XY, tol=1e-3):
    """Warning text when the timeline pallet is not the Gazebo deck, else ``None``."""
    x, y = float(pallet_size[0]), float(pallet_size[1])
    if abs(x - deck[0]) <= tol and abs(y - deck[1]) <= tol:
        return None
    over = x > deck[0] + tol or y > deck[1] + tol
    return (f"timeline pallet {x:.3f} x {y:.3f} m differs from the Gazebo deck {deck[0]:.2f} x {deck[1]:.2f} m; "
            "boxes are centred on it" + (" and overhang it" if over else ""))


def static_sdf(gd, name, size, mass):
    return gd.box_sdf(name, size, mass).replace("<link", "<static>true</static><link", 1)


@dataclass
class StepPlan:
    commands: list = field(default_factory=list)          # dicts for GazeboDriverCore.on_command
    actions: list = field(default_factory=list)           # driver Actions, in order
    conveyor_remove: list = field(default_factory=list)   # model names
    conveyor_spawn: list = field(default_factory=list)    # (name, sdf, (x, y, z, yaw))
    picked: list = field(default_factory=list)            # conveyor models removed once the robot moved
    notes: list = field(default_factory=list)


class GazeboReplayCore:
    """Timeline events -> driver commands -> Gazebo actions (no ROS)."""

    def __init__(self, gd, robot=None, pallet_size=(1.2, 1.0, 1.35), speed_scale=0.3, settle_s=1.0,
                 conveyor=True):
        self.gd = gd
        self.robot = robot
        self.pallet_size = tuple(float(v) for v in pallet_size)
        self.cell = gd.GazeboCell(pallet_size_xy=self.pallet_size[:2])
        self.driver = gd.GazeboDriverCore([], robot, self.cell, speed_scale=speed_scale, settle_s=settle_s)
        self.conveyor = conveyor
        self.pick = pick_point_world(robot, self.cell)
        self.conveyor_models = {}
        self.previous = None
        self.version = 0
        self.unreachable = []

    # -- stage 6 ----------------------------------------------------------
    def _register(self, box_id, sku, size, weight):
        self.driver.by_id[box_id] = {"box_id": box_id, "sku": sku, "size_m": [float(v) for v in size],
                                     "weight_kg": float(weight)}

    def _placed_before(self):
        # snapshot after the previous event (already empty after a pallet close)
        return [] if self.previous is None else snapshot_placements(self.previous)

    def robot_details(self, placement, size, yaw):
        """Stage-6 verdict for the placement on the pallet as it was before
        the event: (details dict or None, reject codes)."""
        if self.robot is None:
            return None, ["NO_ROBOT_CHECK"]
        from pac_common import (InventoryState, PalletState, PlacedBox, PlacementCandidate, Pose3D, Size3D,
                                SystemState)

        def placed(p):
            s, y = p.size_and_yaw()
            return PlacedBox(p.box_id, p.sku or "SKU", Size3D(*s), max(0.0, p.weight),
                             Pose3D("pallet", *p.min_corner, yaw=y))

        pose = Pose3D("pallet", *placement.min_corner, yaw=yaw)
        box = PlacedBox(placement.box_id, placement.sku or "SKU", Size3D(*size), max(0.0, placement.weight), pose)
        others = tuple(placed(p) for p in self._placed_before() if p.box_id != placement.box_id)
        state = SystemState(self.version, 0.0, PalletState("pallet", Size3D(*self.pallet_size), others),
                            InventoryState({}, {}))
        cand = PlacementCandidate(f"rt-{self.version}", placement.box_id, pose, self.version)
        verdict = self.robot.validate_robot_motion(box, cand, state)
        if not verdict.success:
            return None, [c.value for c in verdict.codes]
        return dict(verdict.details), []

    # -- commands ---------------------------------------------------------
    def commands(self, event):
        """Driver command dicts for one event (plus notes)."""
        self.version += 1
        cmds, notes = [], []
        kind = action_kind(event["action"])
        placement = placement_of(event)
        if placement is not None:
            size, yaw = placement.size_and_yaw()
            self._register(placement.box_id, placement.sku, size, placement.weight)
            details, codes = self.robot_details(placement, size, yaw)
            cmd = {"action": kind, "state_version": self.version, "box_id": placement.box_id,
                   "target_min_corner": [*placement.min_corner, yaw], "robot": details}
            if kind == "RETRIEVE_BUFFER":
                cmd["slot"] = int(str(event["action"]).split("(")[1].rstrip(")"))
            if details is None:
                cmd["robot_reject"] = codes
                notes.append(f"stage 6 rejects {placement.box_id} ({', '.join(codes)}): spawned without motion")
                self.unreachable.append(placement.box_id)
            cmds.append(cmd)
        elif lost_placement(event):
            notes.append(f"{event['box']['box_id']}: pose not in timeline (pallet closed in this step), not spawned")
        elif kind == "BUFFER_CURRENT":
            cmds.append({"action": "BUFFER_CURRENT", "state_version": self.version,
                         "box_id": (event.get("box") or {}).get("box_id")})
        if event.get("moved") and not event.get("pallet_closed"):
            rows = {p.box_id: p for p in snapshot_placements(event)}
            moves = []
            for bid in event["moved"]:
                p = rows.get(bid)
                if p is None:
                    continue
                s, y = p.size_and_yaw()
                if bid not in self.driver.by_id:
                    self._register(bid, p.sku, s, p.weight)
                moves.append({"box_id": bid, "target_min_corner": [*p.min_corner, y], "robot": {}})
            if moves:
                cmds.append({"action": "PARTIAL_REPACK", "state_version": self.version, "repack": moves})
        if event.get("pallet_closed"):
            cmds.append({"action": "PALLET_CLOSE", "state_version": self.version})
        return cmds, notes

    def _execute(self, cmd):
        if cmd["action"] in ("PLACE_CURRENT", "RETRIEVE_BUFFER") and not cmd.get("robot"):
            a = self.gd.Actions()
            box = self.driver.by_id[cmd["box_id"]]
            corner = list(cmd["target_min_corner"])
            a.spawn.append((box["box_id"], self.gd.box_sdf(box["box_id"], box["size_m"], box["weight_kg"]),
                            self.cell.box_center(corner, box["size_m"])))
            self.driver.on_pallet.append(box["box_id"])
            self.driver.poses[box["box_id"]] = corner
            return a
        return self.driver.on_command(cmd)

    def _conveyor(self, event, plan):
        if not self.conveyor:
            return
        wanted = conveyor_layout(conveyor_window(event, self.previous), self.pick)
        for name, model in list(self.conveyor_models.items()):
            if wanted.get(name) != model:
                plan.conveyor_remove.append(name)
                del self.conveyor_models[name]
        for name, model in wanted.items():
            if name not in self.conveyor_models:
                size, pose, mass = model
                plan.conveyor_spawn.append((name, static_sdf(self.gd, name, size, mass), pose))
                self.conveyor_models[name] = model

    def step(self, event):
        plan = StepPlan()
        self._conveyor(event, plan)
        plan.commands, plan.notes = self.commands(event)
        for cmd in plan.commands:
            plan.actions.append(self._execute(cmd))
        if action_kind(event["action"]) in FROM_CONVEYOR and event.get("box"):
            # the box leaves the conveyor (to the pallet or the buffer)
            name = CONVEYOR_PREFIX + event["box"]["box_id"]
            if name in self.conveyor_models:
                plan.picked.append(name)
                del self.conveyor_models[name]
        expected = {r["box_id"] for r in event.get("pallet", [])}
        missing = expected - set(self.driver.on_pallet)
        if missing:
            plan.notes.append(f"pallet snapshot has {len(missing)} box(es) not spawned")
        self.previous = event
        return plan


def plan_duration(plan):
    return sum(a.duration_s for a in plan.actions)


# ---------------------------------------------------------------------- Gazebo I/O (main only)
def gz_spawn(world, name, sdf, pose):  # pragma: no cover - needs Gazebo
    x, y, z, yaw = pose
    subprocess.Popen(["ros2", "run", "ros_gz_sim", "create", "-world", world, "-name", name, "-string", sdf,
                      "-x", str(x), "-y", str(y), "-z", str(z + 0.002), "-Y", str(yaw)],
                     stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def gz_remove(world, name):  # pragma: no cover - needs Gazebo
    for tool in ("ign", "gz"):
        try:
            subprocess.Popen([tool, "service", "-s", f"/world/{world}/remove", "--reqtype", "ignition.msgs.Entity",
                              "--reptype", "ignition.msgs.Boolean", "--timeout", "2000",
                              "--req", f'name: "{name}" type: MODEL'],
                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            return
        except FileNotFoundError:
            continue


class RosIO:  # pragma: no cover - needs ROS 2
    def __init__(self, world, topic):
        import rclpy
        from trajectory_msgs.msg import JointTrajectory

        rclpy.init()
        self.rclpy = rclpy
        self.node = rclpy.create_node("pac_realtime_gazebo_replay")
        self.pub = self.node.create_publisher(JointTrajectory, topic, 10)
        self.world = world

    def trajectory(self, points):
        from builtin_interfaces.msg import Duration
        from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

        jt = JointTrajectory(joint_names=JOINTS)
        for q, t in points:
            sec = int(t)
            jt.points.append(JointTrajectoryPoint(positions=[float(v) for v in q],
                                                  time_from_start=Duration(sec=sec, nanosec=int((t - sec) * 1e9))))
        self.pub.publish(jt)

    def wait(self, seconds):
        end = time.monotonic() + seconds
        while time.monotonic() < end:
            self.rclpy.spin_once(self.node, timeout_sec=min(0.1, max(0.0, end - time.monotonic())))

    def spawn(self, name, sdf, pose):
        gz_spawn(self.world, name, sdf, pose)

    def remove(self, name):
        gz_remove(self.world, name)

    def close(self):
        self.node.destroy_node()
        self.rclpy.shutdown()


class PrintIO:
    """``--dry-run``: prints what would be sent to ROS / Gazebo."""

    def __init__(self, out=print):
        self.out = out

    def trajectory(self, points):
        self.out(f"    trajectory {len(points)} pts, {points[-1][1]:.1f} s, q_end "
                 f"{[round(v, 3) for v in points[-1][0]]}")

    def wait(self, seconds):
        pass

    def spawn(self, name, sdf, pose):
        self.out(f"    spawn {name} at {tuple(round(v, 3) for v in pose)}")

    def remove(self, name):
        self.out(f"    remove {name}")

    def close(self):
        pass


def play(core, timeline, io, pacer, out=print, start=0, limit=None, motion_wait=True):
    events = ordered_events(timeline)[start:]
    if limit is not None:
        events = events[:limit]
    max_lag = 0.0
    for i, event in enumerate(events):
        lag = pacer.wait_until(event["t_start"])
        max_lag = max(max_lag, lag)
        plan = core.step(event)
        notes = list(plan.notes) + ([f"behind schedule {lag:.1f} s"] if lag > 0.5 else [])
        out(status_line(event, i, len(events), "; ".join(notes)))
        for name in plan.conveyor_remove:
            io.remove(name)
        for name, sdf, pose in plan.conveyor_spawn:
            io.spawn(name, sdf, pose)
        for acts in plan.actions:
            for name in acts.remove:
                io.remove(name)
            if acts.trajectory:
                io.trajectory(acts.trajectory)
                if motion_wait:
                    io.wait(acts.duration_s)
            for name in plan.picked:
                io.remove(name)
            plan.picked = []
            for name, sdf, pose in acts.spawn:
                io.spawn(name, sdf, pose)
        for name in plan.picked:
            io.remove(name)
    out(f"done: {len(events)} events, stage-6 rejects {len(core.unreachable)} {core.unreachable or ''}, "
        f"max lag {max_lag:.1f} s")
    return max_lag


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("timeline", type=Path)
    ap.add_argument("--speed", type=float, default=1.0, help="timeline seconds per wall second (0 = no pacing)")
    ap.add_argument("--robot-config", type=Path, default=None, help="default: config/taehyeon/robot_check_gazebo.yaml")
    ap.add_argument("--world", default="ahead_workcell_v2")
    ap.add_argument("--trajectory-topic", default="/joint_trajectory_controller/joint_trajectory")
    ap.add_argument("--motion-speed", type=float, default=0.3, help="fraction of joint velocity limits")
    ap.add_argument("--no-conveyor", action="store_true", help="do not show the conveyor window")
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--dry-run", action="store_true", help="print the plan, no ROS / Gazebo")
    return ap.parse_args(argv)


def main(argv=None):  # pragma: no cover - ROS path; --dry-run is exercised by hand
    args = parse_args(argv)
    timeline = load_timeline(args.timeline)
    gd = load_team()
    config = args.robot_config or default_robot_config()
    if config is None:
        print("ERROR: robot_check_gazebo.yaml not found; pass --robot-config")
        return 2
    robot = make_robot(config)
    events = ordered_events(timeline)
    pallet = events[0]["pallet_size"] if events else (1.2, 1.0, 1.35)
    core = GazeboReplayCore(gd, robot, pallet, speed_scale=args.motion_speed, conveyor=not args.no_conveyor)
    print(summary_text(timeline))
    for size in sorted({tuple(e["pallet_size"]) for e in events}):
        warning = deck_warning(size)
        if warning:
            print(f"WARNING: {warning}")
    print(f"robot config {config}; pallet {pallet} centred at world {core.cell.pallet_center_xy}; "
          f"pick point {core.pick}")
    io = PrintIO() if args.dry_run else RosIO(args.world, args.trajectory_topic)
    try:
        if not args.dry_run:
            io.wait(1.0)  # let the publisher connect
        play(core, timeline, io, Pacer(speed=0 if args.dry_run else args.speed), start=args.start,
             limit=args.limit, motion_wait=not args.dry_run)
    finally:
        io.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
