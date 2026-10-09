"""Stage-4 entry point on real snapshots (``HighLevelDecider.decide``)."""

from dataclasses import replace

import pytest

from th_helpers import PLANNING, make_box, make_context, make_state, placed

from pac_candidates import CandidateBackend, CandidateConfig
from pac_common import BoxStatus, Size3D

from pac_highlevel import (
    ActionType,
    HighLevelConfig,
    HighLevelDecider,
    feature_names,
    load_policy,
    new_agent,
)
from pac_highlevel.runtime import as_dict

PALLET = Size3D(0.62, 0.42, 0.25)
FLOOR = [(0.004, 0.2), (0.214, 0.2), (0.424, 0.19)]  # three 0.2 x 0.4 floor slots (x, width)


def cfg(slots=2, fill=0.30, repack=False):
    c = HighLevelConfig()
    return replace(
        c,
        buffer=replace(c.buffer, slots=slots),
        close=replace(c.close, mode="fill", fill_before_buffer=fill),  # fill-threshold rule under test
        repack=replace(c.repack, enabled=repack),
    )


def ctx():
    return make_context(skus=("K", "H", "X"))


def floor(weights):
    return [
        placed(f"P{i}", x, 0.004, 0.0, size=(w, 0.4, 0.1), weight=wt)
        for i, ((x, w), wt) in enumerate(zip(FLOOR, weights))
    ]


def snapshot(boxes, current=None, buffered=(), version=7):
    tracked = {}
    if current is not None:
        tracked[current.box_id] = current
    for b in buffered:
        tracked[b.box_id] = replace(b, status=BoxStatus.BUFFERED)
    return make_state(boxes, version=version, pallet=PALLET, tracked=tracked)


def decider(config=None, policy=None):
    config = config or cfg()
    return HighLevelDecider(ctx(), CandidateConfig(), config, policy=policy or load_policy("rule", config=config))


def test_place_current_on_empty_pallet_and_inputs_untouched():
    state = snapshot([], current=make_box("C1", (0.2, 0.4, 0.1), weight=5))
    before = repr(state)
    d = decider().decide(state)
    assert d.action.type == ActionType.PLACE_CURRENT and d.requires_low_level
    assert d.box.box_id == "C1" and d.state_version == 7
    assert d.candidate.base_state_version == 7  # ids follow the real snapshot
    assert CandidateBackend(ctx(), CandidateConfig()).validate_constraints(d.box, d.candidate, state).success
    assert repr(state) == before
    assert as_dict(d)["action"] == "PLACE_CURRENT"


def test_retrieve_buffered_box_hands_the_buffered_state_to_stage5():
    # floor full of 5 kg boxes; current 20 kg box cannot go on top (share),
    # buffered 4 kg box can -> rule retrieves it; slot comes from the caller
    state = snapshot(floor([5, 5, 5]), current=make_box("C1", (0.2, 0.4, 0.1), weight=20, sku="H"),
                     buffered=[make_box("B1", (0.2, 0.4, 0.1), weight=4)])
    d = decider(cfg(fill=0.0)).decide(state, buffer_slots={1: "B1"})
    assert d.action.type == ActionType.RETRIEVE_BUFFER and d.action.slot == 1 and d.slot == 1
    assert d.box.box_id == "B1" and d.box.status == BoxStatus.BUFFERED
    assert d.requires_low_level and d.candidate is not None


def test_buffer_current_when_it_does_not_fit_and_pallet_is_not_full_enough():
    # two light floor boxes (fill 0.246 < 0.30); the 0.196 m gap is too
    # narrow and the 20 kg box may not rest on the 5 kg ones -> buffer it
    state = snapshot(floor([5, 5]), current=make_box("C1", (0.2, 0.4, 0.1), weight=20, sku="H"))
    d = decider(cfg(fill=0.30)).decide(state)
    assert not d.mask[0]
    assert d.action.type == ActionType.BUFFER_CURRENT and d.slot == 0 and not d.requires_low_level


def test_close_before_buffer_rule_on_a_well_filled_pallet():
    state = snapshot(floor([5, 5, 5]), current=make_box("C1", (0.2, 0.4, 0.1), weight=20, sku="H"))
    d = decider(cfg(fill=0.30)).decide(state)  # fill = 3 * 0.008 / 0.0651 = 0.37
    assert d.action.type == ActionType.PALLET_CLOSE and d.reason == "FILL_BEFORE_BUFFER"
    assert d.decided_by == "rule:close"


def test_close_when_nothing_is_feasible_and_buffer_full():
    heavy = [make_box(f"B{i}", (0.2, 0.4, 0.1), weight=20, sku="H") for i in range(2)]
    state = snapshot(floor([5, 5, 5]), current=make_box("C1", (0.2, 0.4, 0.1), weight=20, sku="H"),
                     buffered=heavy)
    d = decider(cfg(fill=0.0)).decide(state)
    assert d.action.type == ActionType.PALLET_CLOSE and d.reason == "NO_FEASIBLE_ACTION"
    assert not any(d.mask)


def test_repack_returns_moves_to_execute():
    a = placed("A", 0.004, 0.004, 0.0, size=(0.2, 0.4, 0.1), weight=5)
    b = placed("B", 0.414, 0.004, 0.0, size=(0.2, 0.4, 0.1), weight=5)
    state = snapshot([a, b], current=make_box("C1", (0.4, 0.4, 0.1), weight=5))
    d = decider(cfg(slots=0, fill=0.0, repack=True)).decide(state)
    assert d.action.type == ActionType.PARTIAL_REPACK and len(d.repack_moves) == 1
    moved, pose = d.repack_moves[0]
    assert moved in ("A", "B") and pose.frame_id == "pallet"


def test_ng_and_no_box():
    state = snapshot([], current=make_box("C1", (0.8, 0.5, 0.1), weight=5, sku="X"))
    d = decider().decide(state)
    assert d.action.type == ActionType.REJECT_NG and d.box.box_id == "C1"
    d = decider().decide(snapshot(floor([5])))
    assert d.action is None and d.reason == "NO_BOX"


def test_input_validation():
    state = snapshot([], current=make_box("C1", (0.2, 0.4, 0.1)), buffered=[make_box("B1", (0.2, 0.4, 0.1))])
    with pytest.raises(ValueError, match="BUFFERED"):
        decider().decide(state, buffer_slots={0: "B9"})
    with pytest.raises(ValueError, match="slot outside"):
        decider().decide(state, buffer_slots={5: "B1"})
    two = make_state([], version=1, pallet=PALLET, tracked={
        "C1": make_box("C1", (0.2, 0.4, 0.1)), "C2": make_box("C2", (0.2, 0.4, 0.1))})
    with pytest.raises(ValueError, match="current_box_id"):
        decider().decide(two)
    assert decider().decide(two, current_box_id="C2").box.box_id == "C2"


def test_learned_policy_and_contract(tmp_path):
    config = cfg()
    path = tmp_path / "p.json"
    new_agent(config).save(path)
    policy = load_policy("numpy", path, config=config)
    state = snapshot([], current=make_box("C1", (0.2, 0.4, 0.1)))
    d = decider(config, policy).decide(state)
    assert d.decided_by == "policy:maskable_ppo_numpy"
    assert len(d.probabilities) == len(d.mask) and sum(d.probabilities) == pytest.approx(1.0)
    assert all(p == 0.0 for p, m in zip(d.probabilities, d.mask) if not m)
    assert d.mask[d.action.slot + 2 if d.action.slot is not None else
                  (0 if d.action.type == ActionType.PLACE_CURRENT else 1)]
    other = replace(config, buffer=replace(config.buffer, slots=3))
    with pytest.raises(ValueError):
        load_policy("numpy", path, config=other)
    assert len(feature_names(2)) == len(new_agent(config).feature_names)


@pytest.mark.skipif(PLANNING is None, reason="pac_planning not available")
def test_handoff_to_donghan_planner():
    from pac_planning import PlacementPlanner, PlannerConfig

    state = snapshot(floor([5, 5, 5]), current=make_box("C1", (0.2, 0.4, 0.1), weight=20, sku="H"),
                     buffered=[make_box("B1", (0.2, 0.4, 0.1), weight=4)])
    d = decider(cfg(fill=0.0)).decide(state)
    assert d.requires_low_level
    backend = CandidateBackend(ctx(), CandidateConfig())
    planner = PlacementPlanner(context=ctx(), config=PlannerConfig(),
                               generate_candidates=backend.generate_candidates,
                               validate_constraints=backend.validate_constraints)
    result = planner.plan(d.box, state, seed=1, use_time_budget=False, mode="greedy")
    assert result.ranked and result.ranked[0].box_id == d.box.box_id
