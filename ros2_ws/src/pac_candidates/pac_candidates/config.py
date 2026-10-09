"""Checked YAML configuration for stages 5-1 (candidates) and 5-2 (hard mask).

All lengths are metres, masses kilograms, forces newtons, angles radians.
Every default is documented in ``docs/taehyeon/hard_mask.md`` together with
the reason it was chosen. Values are development defaults, not measured
constants of the competition cell; replace them when measurements exist.
"""

from dataclasses import dataclass, field, fields, replace
import math
from pathlib import Path

import yaml

ANCHORS = ("corner_ll", "corner_lr", "corner_ul", "corner_ur", "center")
DEDUP_MODES = ("geometric", "support_aware", "mask_aware", "off")
UNCERTAIN_POLICIES = ("robust", "reject")
LOAD_SHARE_MODELS = ("area", "lever")


def _number(value, name, minimum=None, maximum=None, strict_min=False):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be numeric")
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    if minimum is not None:
        if strict_min and value <= minimum:
            raise ValueError(f"{name} must be > {minimum}")
        if not strict_min and value < minimum:
            raise ValueError(f"{name} must be >= {minimum}")
    if maximum is not None and value > maximum:
        raise ValueError(f"{name} must be <= {maximum}")


@dataclass(frozen=True)
class GenerationConfig:
    """5-1 candidate generation."""

    # 1st-stage orientation set (0 / 90 deg), intersected with each box's
    # ``allowed_yaws_rad``. Yaws giving an identical AABB are kept once.
    yaw_set_rad: tuple[float, ...] = (0.0, math.pi / 2.0)
    use_ems: bool = True
    use_extreme_points: bool = True
    ems_anchors: tuple[str, ...] = ANCHORS
    # Candidates closer than this (same yaw, x and y) are duplicates.
    dedup_distance_m: float = 0.08
    # scale-free form (used when > 0): duplicates within this share of the
    # box's shorter footprint side, so small and large boxes are sampled
    # equally densely
    dedup_distance_ratio: float = 0.0
    # geometric: keep the first by priority (flow-chart order: dedup before
    # mask). support_aware: inside a duplicate cluster prefer a candidate
    # whose vectorised support-ratio/height estimate passes (cheap proxy of
    # the dominant mask reasons). mask_aware: same with the exact hard mask
    # (most faithful, slowest). The hard mask always re-checks everything.
    dedup_mode: str = "support_aware"
    # Output order. priority: deepest-bottom-left (z, y, x). likely_valid_first:
    # candidates passing the cheap pre-mask estimate (support ratio, height,
    # heavy-on-light) first, each group in priority order. Consumers looking
    # only at the first N candidates then do not miss feasible positions.
    order: str = "likely_valid_first"
    # 0 = unlimited. When >0 the first ``max_candidates`` (output order) remain.
    max_candidates: int = 0
    # heavy-on-light ``share`` mode: a heavy box may rest on light boxes only
    # where its weight is split between them (e.g. centred on the seam of two
    # supporters or over a junction). Corner/centre anchors rarely land
    # there, so levels on top of lighter boxes are also scanned (40 mm grid,
    # vectorised estimates) for a few load-balanced anchors.
    balance_anchors: bool = True
    balance_step_m: float = 0.04  # scan grid of the balance anchors

    def __post_init__(self):
        object.__setattr__(self, "yaw_set_rad", tuple(self.yaw_set_rad))
        object.__setattr__(self, "ems_anchors", tuple(self.ems_anchors))
        # empty yaw_set_rad = use each SKU's own allowed orientations
        for yaw in self.yaw_set_rad:
            _number(yaw, "yaw_set_rad")
        if not (self.use_ems or self.use_extreme_points):
            raise ValueError("Enable EMS and/or extreme points")
        for anchor in self.ems_anchors:
            if anchor not in ANCHORS:
                raise ValueError(f"Unknown EMS anchor: {anchor}")
        _number(self.dedup_distance_m, "dedup_distance_m", 0.0)
        _number(self.dedup_distance_ratio, "dedup_distance_ratio", 0.0)
        if self.dedup_mode not in DEDUP_MODES:
            raise ValueError(f"dedup_mode must be one of {DEDUP_MODES}")
        if self.order not in ("priority", "likely_valid_first"):
            raise ValueError("order must be priority or likely_valid_first")
        _number(self.balance_step_m, "balance_step_m", 0.0, strict_min=True)
        if type(self.max_candidates) is not int or self.max_candidates < 0:
            raise ValueError("max_candidates must be a nonnegative integer")


@dataclass(frozen=True)
class UncertaintyConfig:
    """Measurement uncertainty delta handed over from stage 2."""

    # Required free gap between neighbouring boxes (placement repeatability,
    # carton bulge). Neighbours keep >= clearance + both size tolerances.
    lateral_clearance_m: float = 0.004
    # +- dimension uncertainty of a measured box (each side).
    size_tolerance_m: float = 0.002
    # Top faces within this height difference form one contact level.
    height_tolerance_m: float = 0.003
    # CoG may lie anywhere in centre +- max(ratio * side, min) per axis.
    cog_uncertainty_ratio: float = 0.05
    cog_uncertainty_min_m: float = 0.005
    # For PlanningContext.uncertain_box_ids: robust = multiply tolerances,
    # reject = return SENSOR_UNCERTAIN (reference-backend behaviour).
    uncertain_policy: str = "robust"
    uncertain_multiplier: float = 2.0

    def __post_init__(self):
        for name in (
            "lateral_clearance_m",
            "size_tolerance_m",
            "height_tolerance_m",
            "cog_uncertainty_ratio",
            "cog_uncertainty_min_m",
        ):
            _number(getattr(self, name), name, 0.0)
        if self.height_tolerance_m > 0.05:
            raise ValueError("height_tolerance_m above 50 mm is unsafe")
        if self.cog_uncertainty_ratio >= 0.5:
            raise ValueError("cog_uncertainty_ratio must be < 0.5")
        if self.uncertain_policy not in UNCERTAIN_POLICIES:
            raise ValueError(
                f"uncertain_policy must be one of {UNCERTAIN_POLICIES}"
            )
        _number(self.uncertain_multiplier, "uncertain_multiplier", 1.0)


@dataclass(frozen=True)
class LoadModelConfig:
    """Fallback carton top-load capacity when the catalog has no value.

    McKee: BCT = 5.87 * ECT * sqrt(t * Z), Z = footprint perimeter.
    Allowable static stacking load = BCT / safety_factor.
    """

    ect_n_per_m: float = 5000.0
    board_thickness_m: float = 0.003
    safety_factor: float = 4.0
    # How a box's load is split among its supporters.
    share_model: str = "lever"

    def __post_init__(self):
        _number(self.ect_n_per_m, "ect_n_per_m", 0.0, strict_min=True)
        _number(
            self.board_thickness_m, "board_thickness_m", 0.0, strict_min=True
        )
        _number(self.safety_factor, "safety_factor", 1.0)
        if self.share_model not in LOAD_SHARE_MODELS:
            raise ValueError(f"share_model must be one of {LOAD_SHARE_MODELS}")


@dataclass(frozen=True)
class PalletCogConfig:
    """Progressive pallet-CoG region (prevents early-stack deadlock)."""

    enabled: bool = True
    # Final allowed |offset| from the pallet centre as a fraction of the half
    # side. 0.5 -> the CoG must stay inside the central 50 % x 50 % area.
    final_half_extent_ratio: float = 0.5
    # Below this stacked mass the region relaxes linearly to the full pallet.
    ramp_mass_kg: float = 150.0
    # A placement that moves an out-of-region CoG toward the centre is
    # allowed; otherwise a lopsided early layer could block all candidates.
    allow_improving: bool = True

    def __post_init__(self):
        _number(
            self.final_half_extent_ratio,
            "final_half_extent_ratio",
            0.0,
            1.0,
            strict_min=True,
        )
        _number(self.ramp_mass_kg, "ramp_mass_kg", 0.0)


@dataclass(frozen=True)
class HeavyOnLightConfig:
    """Mission requirement: heavy boxes must not be stacked on light boxes."""

    enabled: bool = True
    # per_box (mission wording): reject if new.weight > ratio * supporter.weight
    #   + tolerance_kg for any direct supporter carrying >= ``min_share``.
    # share (default, taehyeon 2026-10-08): compare the load actually
    #   transferred (new.weight * share) instead, so a heavy box bridging
    #   several light boxes may be allowed. Same pallets on the virtual data
    #   used ~20 % fewer pallets with no extra true crush (VALIDATION 10).
    mode: str = "share"
    max_weight_ratio: float = 1.0
    tolerance_kg: float = 0.5
    min_share: float = 0.10

    def __post_init__(self):
        if self.mode not in ("per_box", "share"):
            raise ValueError("heavy_on_light.mode must be per_box or share")
        _number(self.max_weight_ratio, "max_weight_ratio", 0.0, strict_min=True)
        _number(self.tolerance_kg, "tolerance_kg", 0.0)
        _number(self.min_share, "min_share", 0.0, 1.0)


@dataclass(frozen=True)
class ConstraintConfig:
    """5-2 hard mask thresholds (never skipped by a time budget)."""

    min_support_ratio: float = 0.70
    # Extra inward distance required between the worst-case CoG and the
    # load-bearing convex polygon boundary (after the CoG delta).
    min_lbcp_margin_m: float = 0.0
    # Allowed overhang beyond the pallet edge (0 = mission: no protrusion).
    pallet_overhang_m: float = 0.0
    heavy_on_light: HeavyOnLightConfig = field(
        default_factory=HeavyOnLightConfig
    )
    load_model: LoadModelConfig = field(default_factory=LoadModelConfig)
    pallet_cog: PalletCogConfig = field(default_factory=PalletCogConfig)
    # Used only when PlanningContext is absent (standalone use).
    default_pallet_max_weight_kg: float = 1000.0

    def __post_init__(self):
        _number(self.min_support_ratio, "min_support_ratio", 0.0, 1.0)
        _number(self.min_lbcp_margin_m, "min_lbcp_margin_m", 0.0)
        _number(self.pallet_overhang_m, "pallet_overhang_m", 0.0)
        _number(
            self.default_pallet_max_weight_kg,
            "default_pallet_max_weight_kg",
            0.0,
            strict_min=True,
        )


@dataclass(frozen=True)
class CandidateConfig:
    generation: GenerationConfig = field(default_factory=GenerationConfig)
    uncertainty: UncertaintyConfig = field(default_factory=UncertaintyConfig)
    constraints: ConstraintConfig = field(default_factory=ConstraintConfig)
    # Size of the per-state geometric model cache (rollouts reuse states).
    cache_size: int = 256
    # False (runtime): the hard mask stops at the first failing check group;
    # verdicts are identical, rejected details list fewer secondary reasons.
    # True: every reason is listed (used for virtual-data training labels).
    collect_all_reasons: bool = False

    def __post_init__(self):
        if type(self.cache_size) is not int or self.cache_size < 1:
            raise ValueError("cache_size must be a positive integer")
        if not isinstance(self.collect_all_reasons, bool):
            raise ValueError("collect_all_reasons must be a boolean")


_NESTED = {
    CandidateConfig: {
        "generation": GenerationConfig,
        "uncertainty": UncertaintyConfig,
        "constraints": ConstraintConfig,
    },
    ConstraintConfig: {
        "heavy_on_light": HeavyOnLightConfig,
        "load_model": LoadModelConfig,
        "pallet_cog": PalletCogConfig,
    },
}


def _build(cls, data, path):
    if not isinstance(data, dict):
        raise ValueError(f"{path} must be a mapping")
    known = {f.name for f in fields(cls)}
    unknown = set(data) - known
    if unknown:
        raise ValueError(f"Unknown keys in {path}: {sorted(unknown)}")
    kwargs = {}
    for key, value in data.items():
        nested = _NESTED.get(cls, {}).get(key)
        if nested is not None:
            kwargs[key] = _build(nested, value, f"{path}.{key}")
        elif isinstance(value, list):
            kwargs[key] = tuple(value)
        else:
            kwargs[key] = value
    return cls(**kwargs)


def config_from_dict(data):
    """Build a validated config from a mapping (unknown keys are errors)."""
    return _build(CandidateConfig, data or {}, "candidates")


def load_candidate_config(path):
    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("schema_version") != 1:
        raise ValueError("Unsupported candidate config schema")
    return config_from_dict(payload.get("candidates", {}))


def with_overrides(config, **sections):
    """Return a copy with whole sections replaced (testing convenience)."""
    return replace(config, **sections)
