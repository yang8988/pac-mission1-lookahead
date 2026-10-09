"""Virtual data generator for AHEAD stages 5-1/5-2 (owner: taehyeon).

Replays jaesung's dataset-generator scenarios with a stage-2 measurement
model, runs candidate generation (5-1) and the hard mask (5-2) at every
arrival, and writes donghan-compatible planner scenes plus candidate labels.
"""

from .config import VirtualDataConfig, load_virtual_config, virtual_config_from_dict
from .pipeline import generate

__all__ = ["VirtualDataConfig", "generate", "load_virtual_config", "virtual_config_from_dict"]
