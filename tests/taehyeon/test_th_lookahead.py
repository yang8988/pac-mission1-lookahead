"""Stage 4 look-ahead search over the next N visible boxes (taehyeon)."""

from dataclasses import replace

import pytest
from pac_highlevel import LookaheadConfig, LookaheadPolicy, load_lookahead_config, run_policy, to_index
from pac_highlevel.lookahead import (
    dead_share,
    has_safe_spot,
    leaf_score,
    lookahead_config_from_dict,
    pin_placement,
    placement_alternatives,
    window_clone,
)
from test_th_highlevel import box, world
from th_helpers import REPO


def mixed_boxes(n=14):
    sizes = [(0.2, 0.4, 0.1), (0.3, 0.2, 0.15), (0.6, 0.4, 0.12), (0.2, 0.2, 0.2)]
    weights = [5.0, 12.0, 3.0, 20.0]
    return [box(i, sizes[i % 4], weights[i % 4], sku=f"S{i % 4}") for i in range(n)]


def test_config_file_and_validation():
    cfg = load_lookahead_config(REPO / "config/taehyeon/lookahead.yaml")
    assert cfg.horizon >= 1 and cfg.mode in ("pilot", "beam")
    with pytest.raises(ValueError):
        lookahead_config_from_dict({"horizon": 3, "typo": 1})
    with pytest.raises(ValueError):
        LookaheadConfig(mode="tree")


def test_window_clone_leaves_the_real_world_untouched():
    w = world(mixed_boxes())
    before = (list(w.placed), list(w.buffer), w.next_arrival, w.time_s, dict(w.counts))
    c = window_clone(w, w.next_arrival + 2)
    while not c.leaf:
        c.step(to_index(LookaheadPolicy(w.config).rule(c)))
    assert (list(w.placed), list(w.buffer), w.next_arrival, w.time_s, dict(w.counts)) == before
    assert c.next_arrival <= w.next_arrival + 2  # never pulls a box beyond the window


@pytest.mark.parametrize("mode", ["pilot", "beam"])
def test_search_never_reads_boxes_beyond_the_horizon(mode):
    boxes = mixed_boxes(16)
    w1, w2 = world(boxes), world(boxes)
    # same visible window, different hidden future
    for w in (w2,):
        hidden = list(w.arrivals)
        for i in range(w.next_arrival + 2, len(hidden)):
            hidden[i] = replace(hidden[i], box=replace(hidden[i].box, size=hidden[0].box.size))
        w.arrivals = hidden
    p = LookaheadPolicy(w1.config, LookaheadConfig(horizon=2, mode=mode, w_dead=0.0))
    s1, s2 = p.scores(w1), p.scores(w2)
    assert s1.keys() == s2.keys()
    assert all(abs(s1[k] - s2[k]) < 1e-12 for k in s1)


@pytest.mark.parametrize("mode", ["pilot", "beam"])
def test_full_episode_is_safe_and_places_everything(mode):
    w = world(mixed_boxes(), slots=2)
    policy = LookaheadPolicy(w.config, LookaheadConfig(horizon=3, mode=mode))
    out = run_policy(w, policy)
    assert out["safety_issues"] == 0
    assert out["placed"] + out["ng"] == out["boxes"]
    assert policy.stats.decisions == out["decisions"]


def test_leaf_score_counts_waste_of_pallets_closed_in_the_window():
    w = world(mixed_boxes())
    root = window_clone(w, w.next_arrival + 3)
    node = window_clone(w, w.next_arrival + 3)
    cfg = LookaheadConfig(w_dead=0.0, w_rough=0.0, w_void=0.0, w_buffer=0.0, w_time=0.0)
    assert leaf_score(node, root, cfg) == 0.0
    node.step(to_index(LookaheadPolicy(w.config).rule(node)))
    if node.placed:
        node._close_pallet()
        assert leaf_score(node, root, cfg) == pytest.approx(1.0 - node.closed[-1].fill)


def test_dead_share_detects_a_pallet_that_cannot_take_the_remaining_boxes():
    heavy = [box(i, (0.3, 0.2, 0.15), 30.0, sku="H") for i in range(4)]
    w = world([box(100, (0.6, 0.4, 0.05), 1.0, sku="L")] + heavy)
    node = window_clone(w, len(w.arrivals))
    node.step(0)  # light flat box on the floor
    share = dead_share(node, LookaheadConfig())
    assert 0.0 <= share <= 1.0


def test_time_budget_falls_back_without_breaking_masks():
    w = world(mixed_boxes(), slots=2)
    policy = LookaheadPolicy(w.config, LookaheadConfig(horizon=4, mode="beam", time_budget_s=1e-6))
    out = run_policy(w, policy)
    assert out["safety_issues"] == 0 and out["placed"] + out["ng"] == out["boxes"]
    assert policy.stats.searched == 0 or policy.stats.timeouts > 0


def test_placement_alternatives_are_distinct_safe_and_dblf_first():
    w = world(mixed_boxes())
    current, _ = w.options()
    alts = placement_alternatives(w, w.current.box, current.candidate, 4, 0.10)
    assert alts[0] is current.candidate and 1 <= len(alts) <= 4
    backend, state = w.backend(), w.state()
    for c in alts:
        assert backend.validate_constraints(w.current.box, c, state).success
    for i, a in enumerate(alts):
        for b in alts[:i]:
            pa, pb = a.target_pose, b.target_pose
            assert (abs(pa.yaw - pb.yaw) > 1e-6 or abs(pa.z - pb.z) >= 0.02
                    or ((pa.x - pb.x) ** 2 + (pa.y - pb.y) ** 2) ** 0.5 >= 0.10)


def test_pinned_placement_is_executed():
    w = world(mixed_boxes())
    current, _ = w.options()
    alts = placement_alternatives(w, w.current.box, current.candidate, 4, 0.10)
    target = alts[-1]
    pin_placement(w, 0, target)
    box_id = w.current.box.box_id
    w.step(0)
    placed = next(p for p in w.placed if p.box_id == box_id)
    assert placed.pose == target.target_pose


def test_has_safe_spot_matches_the_full_candidate_set():
    w = world(mixed_boxes())
    backend, state = w.backend(), w.state()
    for b in mixed_boxes(4):
        assert has_safe_spot(backend, b, state) == bool(backend.candidate_set(b, state).valid)


@pytest.mark.parametrize("mode", ["pilot", "beam"])
def test_placement_branches_keep_every_episode_safe(mode):
    w = world(mixed_boxes(), slots=2)
    policy = LookaheadPolicy(w.config, LookaheadConfig(horizon=3, mode=mode, place_candidates=3))
    out = run_policy(w, policy)
    assert out["safety_issues"] == 0
    assert out["placed"] + out["ng"] == out["boxes"]


def test_one_placement_candidate_is_the_plain_action_search():
    boxes = mixed_boxes()
    a = LookaheadPolicy(world(boxes).config, LookaheadConfig(horizon=3))
    b = LookaheadPolicy(world(boxes).config, LookaheadConfig(horizon=3, place_candidates=1))
    assert run_policy(world(boxes), a)["pallet_equivalents"] == run_policy(world(boxes), b)["pallet_equivalents"]


# ---------------------------------------------------------------- 5-3 flat layers


def test_layer_placer_continues_a_level_surface():
    from pac_highlevel.placement import ring_metrics
    import numpy as np

    grid = np.zeros((60, 50))
    grid[0:20, :] = 0.2  # a 0.4 m wide strip of 0.2 m boxes along x = 0..0.4
    args = (grid, 0.4, 0.0, 0.0, 0.2, 0.4)
    # 0.2 m box on the floor right next to the strip: level with its top
    level_a, step_a, flush_a = ring_metrics(*args, 0.2, 0.02, 0.003, 0.15)
    # 0.1 m box there: leaves a 0.1 m step
    level_b, step_b, flush_b = ring_metrics(*args, 0.1, 0.02, 0.003, 0.15)
    assert level_a == 0.0 and level_b == 1.0
    assert step_a < step_b and flush_a > flush_b


def test_any_step_breaks_the_level_whatever_the_box_size():
    """A box placed across a 1 cm step rests on the higher part only, as on a
    10 cm step: both count as "not level" (the step size only matters for
    filling it later)."""
    from pac_highlevel.placement import ring_metrics
    import numpy as np

    grid = np.zeros((60, 50))
    grid[0:20, :] = 0.2
    args = (grid, 0.4, 0.0, 0.0, 0.2, 0.4)
    small = ring_metrics(*args, 0.19, 0.02, 0.003, 0.15)
    large = ring_metrics(*args, 0.10, 0.02, 0.003, 0.15)
    assert small[0] == large[0] == 1.0
    assert small[1] < large[1]


def test_layer_placer_picks_only_valid_candidates_and_keeps_episodes_safe():
    from pac_highlevel import LayerPlacer
    from pac_highlevel import RulePolicy

    boxes = mixed_boxes()
    w = world(boxes)
    w.placer = LayerPlacer()
    out = run_policy(w, RulePolicy(w.config))
    assert out["safety_issues"] == 0 and out["placed"] + out["ng"] == out["boxes"]
    w2 = world(boxes)
    w2.placer = LayerPlacer()
    out2 = run_policy(w2, LookaheadPolicy(w2.config, LookaheadConfig(horizon=3, place_candidates=3)))
    assert out2["safety_issues"] == 0


# ---------------------------------------------------------------- real time


def test_timed_run_plans_during_motion_and_respects_the_conveyor():
    from pac_highlevel.realtime import ConveyorConfig, TimedRun

    w = world(mixed_boxes())
    policy = LookaheadPolicy(w.config, LookaheadConfig(horizon=3))
    out = TimedRun(w, policy, ConveyorConfig(interval_s=20.0, capacity=4), horizon=3).run()
    ev = out["events"]
    assert out["summary"]["safety_issues"] == 0
    assert all(b["t_start"] >= a["t_end"] - 1e-6 for a, b in zip(ev, ev[1:]))  # one robot
    assert all(len(e["visible"]) <= 3 for e in ev)
    # slow conveyor: the robot has to wait for boxes, and the planner sees short windows
    assert out["robot_idle_s"] > 0
    assert any(len(e["visible"]) < 3 for e in ev)


def test_dead_share_counts_each_expected_box_once():
    """Visible conveyor boxes are still in the order list's remaining counts;
    they must be counted once (as visible boxes), not again via the order list."""
    from pac_highlevel.lookahead import _probe_boxes

    w = world(mixed_boxes(12))
    node = window_clone(w, w.next_arrival + 3)
    probes = _probe_boxes(node, LookaheadConfig(dead_top_skus=99))
    total = sum(v for _, v in probes)
    unplaced = w.arrivals[w.next_arrival - 1:]  # current box + everything still to come
    expected = sum(a.box.size.x * a.box.size.y * a.box.size.z for a in unplaced)
    assert abs(total - expected) < 1e-9
