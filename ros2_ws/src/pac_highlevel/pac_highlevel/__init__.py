"""AHEAD stage 4: high-level action selection.

PLACE_CURRENT / BUFFER_CURRENT / RETRIEVE_BUFFER(i) are chosen by a rule
policy (1st step), a look-ahead search over the next N visible boxes
(``lookahead``, no learning) or a MaskablePPO policy (extension); PALLET_CLOSE and
PARTIAL_REPACK stay rules; infeasible actions are masked using stages
5-1/5-2 (``pac_candidates``).
"""

from .actions import ActionType, HighLevelAction, action_count, from_index, to_index
from .config import HighLevelConfig, config_from_dict, load_highlevel_config
from .features import feature_names, observe
from .lookahead import LookaheadConfig, LookaheadPolicy, load_lookahead_config
from .placement import LayerConfig, LayerPlacer
from .ppo import MaskablePPO
from .rules import GreedyPolicy, RulePolicy
from .runtime import HighLevelDecider, HighLevelDecision, load_policy
from .trainer import agent_chooser, imitate_teacher, new_agent, policy_contract, run_policy, train
from .value import make_value_provider
from .world import Arrival, PalletizingWorld

__all__ = [
    "ActionType",
    "Arrival",
    "GreedyPolicy",
    "HighLevelAction",
    "HighLevelConfig",
    "HighLevelDecider",
    "HighLevelDecision",
    "LayerConfig",
    "LayerPlacer",
    "LookaheadConfig",
    "LookaheadPolicy",
    "MaskablePPO",
    "PalletizingWorld",
    "RulePolicy",
    "action_count",
    "agent_chooser",
    "config_from_dict",
    "feature_names",
    "from_index",
    "imitate_teacher",
    "load_highlevel_config",
    "load_lookahead_config",
    "load_policy",
    "make_value_provider",
    "new_agent",
    "observe",
    "policy_contract",
    "run_policy",
    "to_index",
    "train",
]
