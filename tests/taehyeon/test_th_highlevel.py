"""Stage 4 high-level action selection (taehyeon): world, masks, rules, PPO."""

import copy
import math
from dataclasses import replace

import numpy as np
import pytest
from pac_candidates import CandidateConfig
from pac_common import PlacedBox, Pose3D, Size3D, SkuSpec
from pac_highlevel import (
    ActionType,
    Arrival,
    GreedyPolicy,
    HighLevelAction,
    HighLevelConfig,
    MaskablePPO,
    PalletizingWorld,
    RulePolicy,
    action_count,
    agent_chooser,
    feature_names,
    from_index,
    imitate_teacher,
    load_highlevel_config,
    new_agent,
    observe,
    policy_contract,
    run_policy,
    to_index,
    train,
)
from pac_highlevel.config import PPOConfig, config_from_dict
from pac_highlevel.ppo import masked_softmax
from pac_highlevel.repack import accessible_ids, plan_repack
from th_helpers import HALF_PI, REPO, make_box

SMALL = Size3D(0.62, 0.42, 0.25)


def box(i, size=(0.2, 0.4, 0.1), weight=5.0, sku="K"):
    return replace(make_box(f"B{i:03d}", size, weight, sku=sku), stamp_sec=float(i))


def catalog(*specs):
    return {s: SkuSpec(s, Size3D(*size), w, (0.0, HALF_PI), 1000.0) for s, size, w in specs}


def world(boxes, pallet=SMALL, slots=2, **kw):
    cfg = HighLevelConfig()
    cfg = replace(cfg, buffer=replace(cfg.buffer, slots=slots), **kw)
    sizes = {b.sku_id: ((b.size.x, b.size.y, b.size.z), b.weight_kg) for b in boxes}
    cat = catalog(*[(s, size, w) for s, (size, w) in sizes.items()])
    return PalletizingWorld([Arrival(b) for b in boxes], pallet, cat, CandidateConfig(), cfg)


# ---------------------------------------------------------------- actions


def test_action_index_round_trip():
    slots = 3
    assert action_count(slots) == 5
    for i in range(action_count(slots)):
        assert to_index(from_index(i, slots)) == i
    with pytest.raises(ValueError):
        to_index(HighLevelAction(ActionType.PALLET_CLOSE))
    with pytest.raises(ValueError):
        HighLevelAction(ActionType.RETRIEVE_BUFFER)
    with pytest.raises(ValueError):
        from_index(5, slots)


def test_config_file_and_validation():
    cfg = load_highlevel_config(REPO / "config/taehyeon/highlevel.yaml")
    assert cfg == HighLevelConfig()  # yaml mirrors the code defaults
    with pytest.raises(ValueError):
        config_from_dict({"highlevel": {"buffer": {"slotz": 2}}})
    with pytest.raises(ValueError):
        config_from_dict({"highlevel": {"buffer": {"slots": 2, "travel_time_s": [1.0]}}})
    times = HighLevelConfig().buffer.travel_times()
    assert times == (4.0, 4.0, 5.0, 5.0)


# ---------------------------------------------------------------- world


def test_masks_follow_feasibility_and_buffer_state():
    w = world([box(i) for i in range(4)], slots=1)
    mask = w.action_mask()
    assert mask.tolist() == [True, True, False]
    w.step(HighLevelAction(ActionType.BUFFER_CURRENT))
    assert w.buffer[0] is not None and w.current.box.box_id == "B001"
    mask = w.action_mask()
    assert mask.tolist() == [True, False, True]  # slot full -> no BUFFER_CURRENT
    with pytest.raises(ValueError):
        w.step(HighLevelAction(ActionType.BUFFER_CURRENT))
    w.step(HighLevelAction(ActionType.RETRIEVE_BUFFER, 0))
    assert w.buffer[0] is None and [p.box_id for p in w.placed] == ["B000"]
    assert w.current.box.box_id == "B001"  # retrieving keeps the current box waiting


def test_close_rule_starts_a_new_pallet_and_every_box_is_valid():
    # three 0.2 x 0.4 floor boxes fill the 0.62 m pallet; heavier ones cannot
    # stack on lighter ones -> PALLET_CLOSE by rule, no buffer
    boxes = [box(i, weight=5.0) for i in range(3)] + [box(3 + i, weight=20.0, sku="H") for i in range(3)]
    no_repack = replace(HighLevelConfig().repack, enabled=False)
    out = run_policy(world(boxes, slots=0, repack=no_repack), GreedyPolicy())
    assert out["pallets_used"] == 2 and out["pallets_closed"] == 1
    assert out["counts"]["PALLET_CLOSE"] == 1
    assert out["placed"] == 6 and out["ng"] == 0 and out["safety_issues"] == 0
    assert out["pallets"][0]["boxes"] == 3
    # with PARTIAL_REPACK a light box is stacked on another light one, which
    # frees floor for the heavy boxes: the first pallet takes more boxes
    out = run_policy(world(boxes, slots=0), GreedyPolicy())
    assert out["counts"]["PARTIAL_REPACK"] >= 1
    assert out["pallets"][0]["boxes"] > 3 and out["safety_issues"] == 0


def test_buffer_avoids_a_close():
    # light, heavy, light, light, heavy, heavy: buffering the first heavy one
    # lets the light boxes finish the floor... the rule policy must never
    # use more pallets than the no-buffer baseline on this stream
    weights = [5, 20, 5, 5, 20, 20]
    boxes = [box(i, weight=w, sku="H" if w > 10 else "K") for i, w in enumerate(weights)]
    no_buf = run_policy(world(boxes, slots=0), GreedyPolicy())
    rule = run_policy(world(boxes, slots=2), RulePolicy(HighLevelConfig()))
    assert rule["pallet_equivalents"] <= no_buf["pallet_equivalents"] + 1e-9
    assert rule["safety_issues"] == 0 and rule["placed"] == 6


def test_oversize_box_goes_to_ng_not_forced():
    boxes = [box(0), box(1, size=(0.8, 0.5, 0.1), sku="X"), box(2)]
    out = run_policy(world(boxes), RulePolicy(HighLevelConfig()))
    assert out["ng"] == 1 and out["placed"] == 2
    assert out["counts"]["REJECT_NG"] == 1


def test_end_of_stream_flushes_the_buffer():
    boxes = [box(i) for i in range(3)]
    w = world(boxes, slots=2)
    w.step(HighLevelAction(ActionType.BUFFER_CURRENT))
    w.step(HighLevelAction(ActionType.BUFFER_CURRENT))
    w.step(HighLevelAction(ActionType.PLACE_CURRENT))
    assert w.current is None and not w.done
    mask = w.action_mask()
    assert not mask[0] and not mask[1] and mask[2] and mask[3]
    while not w.done:
        w.step(int(np.flatnonzero(w.action_mask())[0]))
    assert w.summary()["placed"] == 3


def test_observation_layout_is_fixed():
    w = world([box(i) for i in range(3)], slots=2)
    x = observe(w)
    assert x.shape == (len(feature_names(2)),)
    assert np.all(np.isfinite(x))
    w.step(HighLevelAction(ActionType.BUFFER_CURRENT))
    names = feature_names(2)
    x = observe(w)
    assert x[names.index("s0.occupied")] == 1.0 and x[names.index("s0.opt.feasible")] == 1.0
    assert x[names.index("s1.occupied")] == 0.0


# ---------------------------------------------------------------- repack


def test_repack_moves_an_accessible_box_to_free_the_floor():
    w = world([box(9, size=(0.4, 0.4, 0.1), sku="C")], slots=0, repack=replace(HighLevelConfig().repack))
    # A at x=0, B at x=0.4 -> 0.2 m gap in the middle; C needs 0.4 m of floor
    a = PlacedBox("A", "K", Size3D(0.2, 0.4, 0.1), 5.0, Pose3D("pallet", 0.004, 0.004, 0.0))
    b = PlacedBox("B", "K", Size3D(0.2, 0.4, 0.1), 5.0, Pose3D("pallet", 0.414, 0.004, 0.0))
    for p in (a, b):
        w._boxes[p.box_id] = replace(make_box(p.box_id, (0.2, 0.4, 0.1)), sku_id="K")
        w.placed_truth[p.box_id] = w._boxes[p.box_id]
    w.placed = [a, b]
    w._invalidate()
    assert not w.action_mask().any()
    assert sorted(accessible_ids(w, w.placed)) == ["A", "B"]
    plan = plan_repack(w)
    assert plan is not None
    moves, _ = plan
    assert len(moves) == 1
    w._apply_repack(plan)
    assert w.action_mask()[0]
    w.step(HighLevelAction(ActionType.PLACE_CURRENT))
    s = w.summary()
    assert s["pallets_used"] == 1 and s["placed"] == 3 and s["safety_issues"] == 0


# ---------------------------------------------------------------- PPO


def test_masked_softmax_never_samples_masked_actions():
    agent = MaskablePPO(4, 5, PPOConfig(hidden=(8,), seed=3))
    mask = np.array([False, True, False, True, False])
    for _ in range(200):
        a, _, _ = agent.act(np.random.default_rng(_).standard_normal(4), mask)
        assert mask[a]
    p = masked_softmax(np.zeros((1, 5)), mask[None])
    assert p[0, 0] == 0.0 and math.isclose(p.sum(), 1.0)


def test_policy_gradient_matches_finite_differences():
    cfg = PPOConfig(hidden=(6, 6), seed=1)
    agent = MaskablePPO(5, 4, cfg)
    rng = np.random.default_rng(1)
    k = 12
    x = rng.standard_normal((k, 5))
    m = rng.random((k, 4)) > 0.3
    m[:, 0] = True
    a = np.array([rng.choice(np.flatnonzero(r)) for r in m])
    A = rng.standard_normal(k)
    old = np.log(rng.uniform(0.1, 0.9, k))
    An = (A - A.mean()) / (A.std() + 1e-8)

    def loss(pi):
        logits, _ = pi.forward(x)
        p = masked_softmax(logits, m)
        lpa = np.where(m, np.log(np.where(m, p, 1)), 0)
        r = np.exp(lpa[np.arange(k), a] - old)
        ent = -(p * lpa).sum(1)
        surr = np.minimum(r * An, np.clip(r, 1 - cfg.clip_range, 1 + cfg.clip_range) * An)
        return -surr.mean() - cfg.entropy_coef * ent.mean()

    # analytic gradient: run one update with lr=0 and capture the gradients
    captured = {}

    class Spy:
        def step(self, params, grads):
            captured.setdefault("g", grads)

    agent.pi_opt = Spy()
    agent.vf_opt = Spy()
    cfg_one = replace(cfg, n_epochs=1, batch_size=k, max_grad_norm=1e9)
    agent.cfg = cfg_one
    agent.rng = type("R", (), {"permutation": staticmethod(lambda n: np.arange(n))})()
    base = copy.deepcopy(agent.pi)
    agent.update({"obs": x, "masks": m, "actions": a, "logp": old, "adv": A, "ret": np.zeros(k)})
    grads = captured["g"]
    eps = 1e-6
    for li in range(len(base.W)):
        num = np.zeros_like(base.W[li])
        for idx in [(0, 0), (1, 1), (2, 3)]:
            if idx[0] >= num.shape[0] or idx[1] >= num.shape[1]:
                continue
            plus = copy.deepcopy(base)
            plus.W[li][idx] += eps
            minus = copy.deepcopy(base)
            minus.W[li][idx] -= eps
            fd = (loss(plus) - loss(minus)) / (2 * eps)
            assert grads[li][idx] == pytest.approx(fd, rel=1e-4, abs=1e-8)


def test_policy_save_load_and_contract(tmp_path):
    cfg = HighLevelConfig()
    agent = new_agent(cfg)
    path = tmp_path / "p.json"
    agent.save(path)
    names = feature_names(cfg.buffer.slots)
    loaded = MaskablePPO.load(path, feature_names=names, contract=policy_contract(cfg))
    x = np.random.default_rng(0).standard_normal(len(names))
    mask = np.ones(action_count(cfg.buffer.slots), dtype=bool)
    assert np.allclose(agent.probs(x, mask), loaded.probs(x, mask))
    other = replace(cfg, features=replace(cfg.features, value_provider="donghan"))
    with pytest.raises(ValueError, match="value_provider"):
        MaskablePPO.load(path, feature_names=names, contract=policy_contract(other))
    with pytest.raises(ValueError, match="feature layout"):
        MaskablePPO.load(path, feature_names=feature_names(3))


def test_training_loop_runs_and_policy_respects_masks():
    weights = [5, 20, 5, 5, 20, 20]
    boxes = [box(i, weight=w, sku="H" if w > 10 else "K") for i, w in enumerate(weights)]
    cfg = HighLevelConfig()
    cfg = replace(cfg, buffer=replace(cfg.buffer, slots=2),
                  ppo=replace(cfg.ppo, n_steps=48, batch_size=16, n_epochs=2, hidden=(16, 16)))
    sizes = catalog(("K", (0.2, 0.4, 0.1), 5.0), ("H", (0.2, 0.4, 0.1), 20.0))

    def make_world(i):
        return PalletizingWorld([Arrival(b) for b in boxes], SMALL, sizes, CandidateConfig(), cfg)

    agent = new_agent(cfg)
    finished = train(agent, make_world, 96, workers=1, log=lambda row: None)
    assert len(agent.history) == 2 and finished
    out = run_policy(make_world(0), agent_chooser(agent))
    assert out["placed"] == 6 and out["safety_issues"] == 0


def test_gym_env_interface():
    gym_env = pytest.importorskip("pac_highlevel.gym_env")
    if gym_env.HighLevelGymEnv is None:
        pytest.skip("gymnasium not installed")
    boxes = [box(i) for i in range(4)]

    def make_world(i):
        return world(boxes, slots=2)

    env = gym_env.HighLevelGymEnv(make_world, 2)
    obs, _ = env.reset(seed=0)
    assert env.observation_space.shape == obs.shape
    done = False
    while not done:
        mask = env.action_masks()
        obs, r, done, trunc, info = env.step(int(np.flatnonzero(mask)[0]))
    assert info["summary"]["placed"] == 4


def test_imitation_warm_start_reproduces_the_rule_teacher():
    weights = [5, 20, 5, 5, 20, 20, 5, 20]
    boxes = [box(i, weight=w, sku="H" if w > 10 else "K") for i, w in enumerate(weights)]
    cfg = HighLevelConfig()
    cfg = replace(cfg, buffer=replace(cfg.buffer, slots=2),
                  ppo=replace(cfg.ppo, batch_size=16, hidden=(32, 32), learning_rate=3e-3))
    cat = catalog(("K", (0.2, 0.4, 0.1), 5.0), ("H", (0.2, 0.4, 0.1), 20.0))

    def make_world(i):
        return PalletizingWorld([Arrival(b) for b in boxes], SMALL, cat, CandidateConfig(), cfg)

    agent = new_agent(cfg)
    hist = imitate_teacher(agent, make_world, RulePolicy(cfg), 4, epochs=60, workers=2)
    assert hist[-1]["bc_accuracy"] > 0.95
    rule = run_policy(make_world(0), RulePolicy(cfg))
    clone = run_policy(make_world(0), agent_chooser(agent))
    assert clone["counts"] == rule["counts"]
    assert clone["pallet_equivalents"] == pytest.approx(rule["pallet_equivalents"])


def test_sb3_backend_trains_imitates_and_respects_masks(tmp_path):
    pytest.importorskip("torch")
    pytest.importorskip("sb3_contrib")
    from pac_highlevel import sb3
    from pac_highlevel.trainer import collect_teacher

    weights = [5, 20, 5, 5, 20, 20]
    boxes = [box(i, weight=w, sku="H" if w > 10 else "K") for i, w in enumerate(weights)]
    cfg = HighLevelConfig()
    cfg = replace(cfg, buffer=replace(cfg.buffer, slots=2),
                  ppo=replace(cfg.ppo, n_steps=64, batch_size=16, n_epochs=2, hidden=(16, 16)))
    cat = catalog(("K", (0.2, 0.4, 0.1), 5.0), ("H", (0.2, 0.4, 0.1), 20.0))

    def make_world(i):
        return PalletizingWorld([Arrival(b) for b in boxes], SMALL, cat, CandidateConfig(), cfg)

    env = sb3.make_vec_env(make_world, 2, n_envs=2, subprocess=False)
    model = sb3.new_model(env, cfg.ppo)
    data = collect_teacher(make_world, RulePolicy(cfg), 3, gamma=cfg.ppo.gamma)
    hist = sb3.imitate(model, data, epochs=40, batch_size=16, lr=3e-3)
    assert hist[-1]["bc_accuracy"] > 0.9
    model.learn(total_timesteps=64)
    path = tmp_path / "p.zip"
    sb3.save(model, path, 2, policy_contract(cfg))
    loaded = sb3.load(path, slots=2, contract=policy_contract(cfg))
    # frozen normalisation stored with the policy == VecNormalize at save time
    assert loaded.pac_obs_norm is not None
    x = observe(make_world(0))
    expected = model.get_vec_normalize_env().normalize_obs(x)
    assert np.allclose(sb3.normalize_obs(x, loaded.pac_obs_norm), expected, atol=1e-5)
    out = run_policy(make_world(0), sb3.chooser(loaded))
    assert out["placed"] == 6 and out["safety_issues"] == 0
    other = replace(cfg, features=replace(cfg.features, value_provider="donghan"))
    with pytest.raises(ValueError, match="value_provider"):
        sb3.load(path, slots=2, contract=policy_contract(other))


def test_close_before_buffer_rule():
    # 3 light floor boxes fill the 0.62 m pallet (fill 0.4); a heavy box
    # cannot go on top. Rule on: close instead of buffering it.
    boxes = [box(i, weight=5.0) for i in range(3)] + [box(3, weight=20.0, sku="H")]
    no_repack = replace(HighLevelConfig().repack, enabled=False)
    fill_mode = replace(HighLevelConfig().close, mode="fill")
    off = world(boxes, slots=2, repack=no_repack, close=replace(fill_mode, fill_before_buffer=0.0))
    for _ in range(3):
        off.step(HighLevelAction(ActionType.PLACE_CURRENT))
    assert off.action_mask().tolist() == [False, True, False, False]  # must buffer
    on = world(boxes, slots=2, repack=no_repack, close=replace(fill_mode, fill_before_buffer=0.3))
    for _ in range(3):
        on.step(HighLevelAction(ActionType.PLACE_CURRENT))
    assert on.counts["PALLET_CLOSE"] == 1 and on.placed == []
    assert on.action_mask()[0]  # heavy box goes on the new pallet directly


def test_dead_pallet_close_rule():
    # same cell: the only box still expected (the heavy one) has no safe spot,
    # so the whole expected volume is "dead" -> close (no fill threshold involved)
    boxes = [box(i, weight=5.0) for i in range(3)] + [box(3, weight=20.0, sku="H")]
    no_repack = replace(HighLevelConfig().repack, enabled=False)
    dead = world(boxes, slots=2, repack=no_repack, close=replace(HighLevelConfig().close, mode="dead"))
    for _ in range(3):
        dead.step(HighLevelAction(ActionType.PLACE_CURRENT))
    assert dead.counts["PALLET_CLOSE"] == 1 and dead.action_mask()[0]
    # a stricter share than what is dead keeps the pallet open (box must be buffered)
    keep = world(boxes, slots=2, repack=no_repack,
                 close=replace(HighLevelConfig().close, mode="dead", dead_share=1.01))
    for _ in range(3):
        keep.step(HighLevelAction(ActionType.PLACE_CURRENT))
    assert keep.counts["PALLET_CLOSE"] == 0


def test_order_list_known_flag_controls_remaining_counts():
    boxes = [box(i) for i in range(4)]
    known = world(boxes)
    assert known.state().inventory.remaining_by_sku == {"K": 3}
    hidden = world(boxes, features=replace(HighLevelConfig().features, order_list_known=False))
    assert dict(hidden.state().inventory.remaining_by_sku) == {}
    names = feature_names(2)
    assert observe(hidden)[names.index("inv.unseen_count")] == 0.0
    assert observe(known)[names.index("inv.unseen_count")] > 0.0


def test_ng_before_first_decision_is_charged_once():
    boxes = [box(0, size=(0.8, 0.5, 0.1), sku="X"), box(1)]
    w = world(boxes)
    assert w.ng == ["B000"]
    r = w.step(HighLevelAction(ActionType.PLACE_CURRENT))
    vol = 0.2 * 0.4 * 0.1 / (SMALL.x * SMALL.y * SMALL.z)
    assert r < vol  # includes the -ng_penalty of the rejected first box
    assert r == pytest.approx(vol - HighLevelConfig().reward.ng_penalty
                              - HighLevelConfig().reward.time_weight * HighLevelConfig().timing.place_time_s)


def test_detected_damage_box_never_carries_anything():
    # B000 is damaged (detected): it may be placed, but nothing on top of it
    boxes = [box(i, size=(0.2, 0.4, 0.1)) for i in range(6)]
    cfg = HighLevelConfig()
    cfg = replace(cfg, buffer=replace(cfg.buffer, slots=0))
    cat = catalog(("K", (0.2, 0.4, 0.1), 5.0))
    arrivals = [Arrival(b, damage_detected=(b.box_id == "B000")) for b in boxes]
    w = PalletizingWorld(arrivals, SMALL, cat, CandidateConfig(), cfg)
    policy = GreedyPolicy()
    saw_b000 = False
    while not w.done:
        w.step(policy(w))
        if any(p.box_id == "B000" for p in w.placed):
            saw_b000 = True
            model = w.backend().model_for(w.state())
            assert all(c.supporter_id != "B000" for cs in model.contacts.values() for c in cs)
    assert saw_b000 and w.capacity_overrides == {"B000": 0.0}
    assert w.summary()["placed"] == 6 and w.summary()["safety_issues"] == 0


def test_repack_footprint_rule():
    from pac_highlevel.repack import _footprint

    a = Pose3D("pallet", 0.1, 0.1, 0.0, yaw=0.0)
    turned = Pose3D("pallet", 0.1, 0.1, 0.0, yaw=HALF_PI)
    flipped = Pose3D("pallet", 0.1, 0.1, 0.0, yaw=math.pi)
    size = Size3D(0.4, 0.2, 0.1)
    assert _footprint(a, size) != _footprint(turned, size)  # rotate in place: a real move
    assert _footprint(a, size) == _footprint(flipped, size)  # same footprint: not a move


def test_donghan_value_provider_requires_a_model():
    pytest.importorskip("pac_planning")
    from pac_highlevel.value import make_value_provider

    with pytest.raises(ValueError, match="model_path"):
        make_value_provider("donghan")


def test_rule_baseline_reward_and_teacher_labels():
    gym_env = pytest.importorskip("pac_highlevel.gym_env")
    if gym_env.HighLevelGymEnv is None:
        pytest.skip("gymnasium not installed")
    from pac_highlevel.trainer import collect_teacher

    weights = [5, 20, 5, 5, 20, 20]
    boxes = [box(i, weight=w, sku="H" if w > 10 else "K") for i, w in enumerate(weights)]
    cfg = replace(HighLevelConfig(), buffer=replace(HighLevelConfig().buffer, slots=2))
    cat = catalog(("K", (0.2, 0.4, 0.1), 5.0), ("H", (0.2, 0.4, 0.1), 20.0))

    def make_world(i):
        return PalletizingWorld([Arrival(b) for b in boxes], SMALL, cat, CandidateConfig(), cfg)

    rule = RulePolicy(cfg)
    rule_return = run_policy(make_world(0), rule)["return"]
    env = gym_env.HighLevelGymEnv(make_world, 2, baseline=lambda w: run_policy(w, rule)["return"], teacher=rule)
    env.reset(seed=0)
    total, done, labels = 0.0, False, []
    while not done:  # follow the teacher: relative return must be 0
        expected = to_index(rule(env.world))
        _, r, done, _, info = env.step(expected)
        labels.append(info["teacher_action"] == expected)
        total += r
    assert all(labels)
    assert info["summary"]["baseline_return"] == pytest.approx(rule_return)
    assert info["summary"]["return"] == pytest.approx(rule_return)
    assert total == pytest.approx(0.0, abs=1e-9)
    data = collect_teacher(make_world, rule, 1, gamma=1.0, relative=True)
    assert data["returns"][0] == pytest.approx(0.0, abs=1e-9)


def test_sb3_teacher_bc_callback_pulls_policy_to_the_rule():
    pytest.importorskip("torch")
    pytest.importorskip("sb3_contrib")
    from pac_highlevel import sb3

    weights = [5, 20, 5, 5, 20, 20]
    boxes = [box(i, weight=w, sku="H" if w > 10 else "K") for i, w in enumerate(weights)]
    cfg = HighLevelConfig()
    cfg = replace(cfg, buffer=replace(cfg.buffer, slots=2),
                  ppo=replace(cfg.ppo, n_steps=64, batch_size=16, n_epochs=1, hidden=(16, 16), learning_rate=3e-3))
    cat = catalog(("K", (0.2, 0.4, 0.1), 5.0), ("H", (0.2, 0.4, 0.1), 20.0))

    def make_world(i):
        return PalletizingWorld([Arrival(b) for b in boxes], SMALL, cat, CandidateConfig(), cfg)

    rule = RulePolicy(cfg)
    env = sb3.make_vec_env(make_world, 2, n_envs=2, subprocess=False, teacher=rule,
                           baseline=lambda w: run_policy(w, rule)["return"])
    model = sb3.new_model(env, cfg.ppo)
    cb = sb3.teacher_bc_callback(5.0, 5.0, epochs=20, batch_size=16)
    model.learn(total_timesteps=64 * 6, callback=cb)
    assert model.logger.name_to_value.get("bc/agreement", 0.0) > 0.9 or cb.pending is not None
    cb._clone(*cb.pending)
    norm = model.get_vec_normalize_env()
    model.pac_obs_norm = {"mean": norm.obs_rms.mean.tolist(), "var": norm.obs_rms.var.tolist(),
                          "clip": float(norm.clip_obs), "eps": float(norm.epsilon)}
    out = run_policy(make_world(0), sb3.chooser(model))
    assert out["counts"] == run_policy(make_world(0), rule)["counts"]
