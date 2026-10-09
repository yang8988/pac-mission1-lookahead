"""Per-option future value ("선택지별 AI Future Value 값, 학습·실전 동일").

The provider name is stored in the trained policy and checked on load, so a
policy trained with one provider is never deployed with another.

* ``proxy``  : heightmap flatness after the placement (1 - std/H). Fast, no
               team dependency; used for training in this repository.
* ``donghan``: donghan's 5-4 value head (``PlacementPlanner.plan`` in
               ``ranking`` mode, no rollout): predicted mean future added
               volume / pallet capacity for the chosen candidate.
"""


def proxy_value(world, box, candidate, state, option):
    return option.flatness_after


def _load_ranker(model_path):
    from pac_planning.model import DualHeadRanker

    if model_path is None:
        raise ValueError("donghan value provider needs model_path (dual_head_ranker.json)")
    return DualHeadRanker.load(model_path)


class DonghanValue:
    """Future value from donghan's 5-4 value head (``plan`` in ranking mode).

    The trained model is loaded once; without a model the planner would
    return zero future values, so a model path is required.
    """

    name = "donghan"

    def __init__(self, model_path=None, planner_config=None):
        from pac_planning import PlacementPlanner, PlannerConfig

        self._planner_cls = PlacementPlanner
        self._config = planner_config or PlannerConfig()
        self._model = _load_ranker(model_path)

    def __call__(self, world, box, candidate, state, option):
        backend = world.backend()
        planner = self._planner_cls(
            context=backend.context,
            config=self._config,
            generate_candidates=backend.generate_candidates,
            validate_constraints=backend.validate_constraints,
            model=self._model,
        )
        box = state.inventory.tracked_boxes.get(box.box_id, box)
        result = planner.plan(box, state, [candidate], mode="ranking", use_time_budget=False)
        if not result.evaluations:
            return 0.0
        return float(result.evaluations[0].future.mean)


class DonghanPlacer:
    """Low-level placement by donghan's 5-3~5-6 planner (instead of DBLF).

    Called with the hard-mask-valid candidates of 5-1/5-2; returns the
    planner's rank-1 candidate. Slow (one ``plan`` per option and decision),
    used for evaluation, not for PPO training. The model (optional: without
    it the planner uses its heuristic) is loaded once.
    """

    wants_context = True

    def __init__(self, model_path=None, planner_config=None, seed=7):
        from pac_planning import PlacementPlanner, PlannerConfig

        self._planner_cls = PlacementPlanner
        self._config = planner_config or PlannerConfig()
        self._model = _load_ranker(model_path) if model_path is not None else None
        self._seed = seed
        self.calls = 0

    def __call__(self, valid, box, state, backend):
        self.calls += 1
        planner = self._planner_cls(
            context=backend.context,
            config=self._config,
            generate_candidates=backend.generate_candidates,
            validate_constraints=backend.validate_constraints,
            model=self._model,
        )
        # the planner requires the State Manager's copy (e.g. status BUFFERED)
        box = state.inventory.tracked_boxes.get(box.box_id, box)
        result = planner.plan(box, state, valid, seed=self._seed)
        return result.ranked[0] if result.ranked else None


def make_value_provider(name, **kwargs):
    if name == "proxy":
        return proxy_value
    if name == "donghan":
        return DonghanValue(**kwargs)
    raise ValueError(f"Unknown value provider {name}")
