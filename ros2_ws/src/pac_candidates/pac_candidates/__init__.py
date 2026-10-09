"""AHEAD low-level planner stages 5-1 (candidate generation) and 5-2 (hard mask).

Owner: taehyeon. Connects to stages 5-3~5-6 through the team v0.2 callbacks
``generate_candidates(box, state)`` and ``validate_constraints(box, candidate,
state)``; see docs/taehyeon/interface.md.
"""

from .backend import CandidateBackend
from .config import CandidateConfig, config_from_dict, load_candidate_config
from .hard_mask import EVIDENCE_SOURCE
from .reports import CandidateInfo, CandidateSet, GenerationReport

__version__ = "1.0.0"

__all__ = [
    "CandidateBackend",
    "CandidateConfig",
    "CandidateInfo",
    "CandidateSet",
    "EVIDENCE_SOURCE",
    "GenerationReport",
    "config_from_dict",
    "load_candidate_config",
]
