"""Checked configuration of the virtual data generator."""

from dataclasses import dataclass, field, fields
import math
from pathlib import Path

import yaml

POLICIES = ("dblf", "random", "mixed", "planner")


def _nonneg(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be numeric")
    if not math.isfinite(value) or value < 0:
        raise ValueError(f"{name} must be finite and >= 0")


@dataclass(frozen=True)
class PalletSection:
    height_limit_includes_pallet: bool = True
    default_max_load_kg: float = 1000.0
    # Pallet footprints (x, y) m to cover pallet-spec changes, assigned round
    # robin inside each scenario family. Empty = keep the generator's pallet.
    sizes_m: tuple = ((1.1, 1.1), (1.2, 1.0), (1.2, 0.8))
    # cargo height above the deck; 0 = from the scenario data
    max_stack_height_m: float = 0.0

    def __post_init__(self):
        _nonneg(self.max_stack_height_m, "max_stack_height_m")
        sizes = tuple(tuple(float(v) for v in xy) for xy in self.sizes_m)
        for xy in sizes:
            if len(xy) != 2 or min(xy) <= 0 or not all(math.isfinite(v) for v in xy):
                raise ValueError("pallet.sizes_m entries must be positive [x, y] pairs")
        object.__setattr__(self, "sizes_m", sizes)
        _nonneg(self.default_max_load_kg, "default_max_load_kg")
        if self.default_max_load_kg <= 0:
            raise ValueError("default_max_load_kg must be positive")


PACKAGING = ("corrugated", "rigid_crate", "no_stack")


@dataclass(frozen=True)
class CatalogSection:
    nominal_weight: str = "midpoint"
    # top-load strength model per packaging type (5-2 crush check):
    #   corrugated  -> McKee formula from the footprint, / safety factor
    #   rigid_crate -> rated top load (rated_top_load_n per SKU, else default_rated_top_load_n)
    #   no_stack    -> nothing may rest on it (bags, fragile goods)
    packaging_default: str = "corrugated"
    packaging: dict = None              # {sku_id: type}
    rated_top_load_n: dict = None       # {sku_id: N} for rigid_crate
    default_rated_top_load_n: float = 2000.0

    def __post_init__(self):
        if self.nominal_weight not in ("midpoint", "max"):
            raise ValueError("nominal_weight must be midpoint or max")
        object.__setattr__(self, "packaging", dict(self.packaging or {}))
        object.__setattr__(self, "rated_top_load_n", {k: float(v) for k, v in (self.rated_top_load_n or {}).items()})
        for kind in [self.packaging_default, *self.packaging.values()]:
            if kind not in PACKAGING:
                raise ValueError(f"packaging must be one of {PACKAGING}")

    def packaging_of(self, sku_id):
        return self.packaging.get(sku_id, self.packaging_default)


@dataclass(frozen=True)
class ObservationSection:
    dimension_noise_std_m: float = 0.001
    dimension_noise_clip_m: float = 0.003
    weight_noise_std_ratio: float = 0.01
    uncertain_probability: float = 0.05
    uncertain_dimension_noise_std_m: float = 0.004
    uncertain_confidence: float = 0.6

    def __post_init__(self):
        for f in fields(self):
            _nonneg(getattr(self, f.name), f.name)
        if self.uncertain_probability > 1 or self.uncertain_confidence > 1:
            raise ValueError("probabilities/confidence must be <= 1")


@dataclass(frozen=True)
class EpisodeSection:
    policy: str = "mixed"
    epsilon: float = 0.3
    top_n_random: int = 5
    scene_every: int = 1
    max_steps: int = 0

    def __post_init__(self):
        if self.policy not in POLICIES:
            raise ValueError(f"policy must be one of {POLICIES}")
        _nonneg(self.epsilon, "epsilon")
        if self.epsilon > 1:
            raise ValueError("epsilon must be <= 1")
        for name in ("top_n_random", "scene_every"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"{name} must be a positive integer")
        if type(self.max_steps) is not int or self.max_steps < 0:
            raise ValueError("max_steps must be a nonnegative integer")


@dataclass(frozen=True)
class StrengthSection:
    """True carton strength scenarios (see strength.py)."""

    enabled: bool = True
    # profile -> weight. round_robin cycles through the listed profiles by
    # scenario index (balanced coverage); random draws by weight.
    profiles: dict = field(
        default_factory=lambda: {
            "strong": 1, "nominal": 1, "weak": 1, "humid": 1, "mixed": 1, "extreme": 1
        }
    )
    assignment: str = "round_robin"
    damage_factor: float = 0.25
    detect_probability: float = 0.8

    def __post_init__(self):
        from .strength import STRENGTH_PROFILES

        if not self.profiles:
            raise ValueError("strength.profiles must not be empty")
        for name, weight in self.profiles.items():
            if name not in STRENGTH_PROFILES:
                raise ValueError(f"Unknown strength profile {name}")
            _nonneg(weight, f"strength.profiles.{name}")
        if sum(self.profiles.values()) <= 0:
            raise ValueError("strength.profiles needs at least one positive weight")
        if self.assignment not in ("round_robin", "random"):
            raise ValueError("strength.assignment must be round_robin or random")
        _nonneg(self.damage_factor, "damage_factor")
        _nonneg(self.detect_probability, "detect_probability")
        if self.damage_factor > 1 or self.detect_probability > 1:
            raise ValueError("damage_factor / detect_probability must be <= 1")


@dataclass(frozen=True)
class VirtualDataConfig:
    seed: int = 20261007
    pallet: PalletSection = field(default_factory=PalletSection)
    catalog: CatalogSection = field(default_factory=CatalogSection)
    observation: ObservationSection = field(default_factory=ObservationSection)
    episode: EpisodeSection = field(default_factory=EpisodeSection)
    strength: StrengthSection = field(default_factory=StrengthSection)

    def __post_init__(self):
        if type(self.seed) is not int:
            raise ValueError("seed must be an integer")


_SECTIONS = {
    "pallet": PalletSection,
    "catalog": CatalogSection,
    "observation": ObservationSection,
    "episode": EpisodeSection,
    "strength": StrengthSection,
}


def virtual_config_from_dict(data):
    data = dict(data or {})
    unknown = set(data) - {"seed", *_SECTIONS}
    if unknown:
        raise ValueError(f"Unknown virtual_data keys: {sorted(unknown)}")
    kwargs = {}
    if "seed" in data:
        kwargs["seed"] = data["seed"]
    for name, cls in _SECTIONS.items():
        if name in data:
            section = data[name] or {}
            extra = set(section) - {f.name for f in fields(cls)}
            if extra:
                raise ValueError(f"Unknown keys in virtual_data.{name}: {sorted(extra)}")
            kwargs[name] = cls(**section)
    return VirtualDataConfig(**kwargs)


def load_virtual_config(path):
    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("schema_version") != 1:
        raise ValueError("Unsupported virtual data config schema")
    return virtual_config_from_dict(payload.get("virtual_data"))
