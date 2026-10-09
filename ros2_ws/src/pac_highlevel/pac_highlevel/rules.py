"""Rule policies for the learned action slots (1st step: everything by rule).

``RulePolicy`` (default baseline)
    1. a buffered box older than ``max_buffer_age`` that fits -> retrieve it
    2. current box fits well (support >= good_support) -> PLACE_CURRENT,
       unless a buffered box fits ``retrieve_margin_m`` lower -> retrieve it
    3. current box fits only poorly or not at all:
       a buffered box fits -> retrieve the lowest one
       free slot -> BUFFER_CURRENT
       current fits -> PLACE_CURRENT
    (no feasible learned action -> world applies PARTIAL_REPACK / PALLET_CLOSE)

``GreedyPolicy``
    PLACE_CURRENT whenever feasible; otherwise retrieve / buffer like the
    rule. Run with ``buffer.slots = 0`` it is the no-buffer baseline (the
    world closes the pallet as soon as the current box does not fit).
"""

from .actions import ActionType, HighLevelAction


def _best_buffered(buffered, mask):
    best = None
    for i, opt in enumerate(buffered):
        if opt is None or not mask[2 + i]:
            continue
        key = (round(opt.top_after, 6), -opt.support_ratio, i)
        if best is None or key < best[0]:
            best = (key, i, opt)
    return best


class RulePolicy:
    name = "rule"

    def __init__(self, config):
        self.cfg = config.rules

    def __call__(self, world):
        mask = world.action_mask()
        current, buffered = world.options()
        for i, entry in enumerate(world.buffer):
            if entry is not None and mask[2 + i] and (
                world.decisions - entry.stored_at >= self.cfg.age_limit(world.slots)
            ):
                return HighLevelAction(ActionType.RETRIEVE_BUFFER, i)
        best = _best_buffered(buffered, mask)
        if mask[0] and current.support_ratio >= self.cfg.good_support:
            margin = self.cfg.margin_for(world.current.box)
            if best is not None and best[2].top_after < current.top_after - margin:
                return HighLevelAction(ActionType.RETRIEVE_BUFFER, best[1])
            return HighLevelAction(ActionType.PLACE_CURRENT)
        if best is not None:
            return HighLevelAction(ActionType.RETRIEVE_BUFFER, best[1])
        if mask[1]:
            return HighLevelAction(ActionType.BUFFER_CURRENT)
        if mask[0]:
            return HighLevelAction(ActionType.PLACE_CURRENT)
        raise RuntimeError("world asked for a decision with an empty mask")


class GreedyPolicy:
    name = "greedy"

    def __call__(self, world):
        mask = world.action_mask()
        if mask[0]:
            return HighLevelAction(ActionType.PLACE_CURRENT)
        best = _best_buffered(world.options()[1], mask)
        if best is not None:
            return HighLevelAction(ActionType.RETRIEVE_BUFFER, best[1])
        if mask[1]:
            return HighLevelAction(ActionType.BUFFER_CURRENT)
        raise RuntimeError("world asked for a decision with an empty mask")

