"""Stage 4 high-level actions (flowchart box 4).

Five actions exist. The first three are the learned ones (MaskablePPO in the
extension step); PALLET_CLOSE and PARTIAL_REPACK always stay rules (TXT 6/8
conditions). Every action that cannot be executed is masked.

Discrete index layout used by the PPO policy (buffer of S slots)::

    0           PLACE_CURRENT
    1           BUFFER_CURRENT
    2 .. 2+S-1  RETRIEVE_BUFFER(i)
"""

from dataclasses import dataclass
from enum import Enum


class ActionType(str, Enum):
    PLACE_CURRENT = "PLACE_CURRENT"
    BUFFER_CURRENT = "BUFFER_CURRENT"
    RETRIEVE_BUFFER = "RETRIEVE_BUFFER"
    PALLET_CLOSE = "PALLET_CLOSE"
    PARTIAL_REPACK = "PARTIAL_REPACK"
    # not a choice: a box that cannot be placed even on an empty pallet is
    # sent to the Inspection / NG area (stage 2 flow), recorded for metrics
    REJECT_NG = "REJECT_NG"


LEARNED = (ActionType.PLACE_CURRENT, ActionType.BUFFER_CURRENT, ActionType.RETRIEVE_BUFFER)
RULE_ONLY = (ActionType.PALLET_CLOSE, ActionType.PARTIAL_REPACK)


@dataclass(frozen=True)
class HighLevelAction:
    type: ActionType
    slot: int | None = None

    def __post_init__(self):
        if (self.type == ActionType.RETRIEVE_BUFFER) != (self.slot is not None):
            raise ValueError("slot is required for RETRIEVE_BUFFER only")

    def label(self):
        if self.type == ActionType.RETRIEVE_BUFFER:
            return f"RETRIEVE_BUFFER({self.slot})"
        return self.type.value


def action_count(slots):
    return 2 + int(slots)


def to_index(action):
    if action.type == ActionType.PLACE_CURRENT:
        return 0
    if action.type == ActionType.BUFFER_CURRENT:
        return 1
    if action.type == ActionType.RETRIEVE_BUFFER:
        return 2 + action.slot
    raise ValueError(f"{action.type} is rule-only and has no policy index")


def from_index(index, slots):
    index = int(index)
    if index == 0:
        return HighLevelAction(ActionType.PLACE_CURRENT)
    if index == 1:
        return HighLevelAction(ActionType.BUFFER_CURRENT)
    if 2 <= index < 2 + slots:
        return HighLevelAction(ActionType.RETRIEVE_BUFFER, index - 2)
    raise ValueError(f"action index {index} out of range")
