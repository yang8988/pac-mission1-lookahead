"""Typed results of stages 5-1/5-2 beyond the bare team callbacks."""

from dataclasses import dataclass, field

from pac_common import FrozenDict


@dataclass(frozen=True)
class CandidateInfo:
    """How a candidate was produced (for features, logs and debugging)."""

    candidate_id: str
    source: str  # "EMS" | "EP"
    anchor: str  # corner_ll/lr/ul/ur, center, ep_*
    yaw_rad: float
    footprint_m: tuple  # rotated (dx, dy, dz)
    ems_lower_m: tuple | None  # actual-plane EMS (x0, y0, z) or None
    ems_upper_m: tuple | None  # actual-plane EMS (x1, y1, z_top) or None


@dataclass(frozen=True)
class GenerationReport:
    """Output of 5-1 for one (box, state)."""

    state_version: int
    box_id: str
    candidates: tuple  # PlacementCandidate, priority order
    infos: dict  # candidate_id -> CandidateInfo
    raw_count: int  # before 80 mm de-duplication
    ems_count: int
    extreme_point_count: int
    yaws_rad: tuple
    elapsed_sec: float

    def __post_init__(self):
        object.__setattr__(self, "candidates", tuple(self.candidates))
        object.__setattr__(self, "infos", FrozenDict(self.infos))


@dataclass(frozen=True)
class CandidateSet:
    """5-1 + 5-2 result handed to 5-3 (feature computation)."""

    generation: GenerationReport
    valid: tuple  # PlacementCandidate passing the hard mask, priority order
    rejected: dict  # candidate_id -> ValidationResult
    code_counts: dict  # RejectCode value -> count
    reason_counts: dict  # detailed reason (prefix) -> count
    mask_elapsed_sec: float
    snapshot_issues: dict = field(default_factory=dict)

    def __post_init__(self):
        object.__setattr__(self, "valid", tuple(self.valid))
        for name in ("rejected", "code_counts", "reason_counts", "snapshot_issues"):
            object.__setattr__(self, name, FrozenDict(getattr(self, name)))

    @property
    def generated_count(self):
        return len(self.generation.candidates)

    @property
    def masked_count(self):
        return len(self.rejected)

    @property
    def valid_count(self):
        return len(self.valid)

    def summary(self):
        return (
            f"{self.generated_count} generated -> {self.masked_count} masked "
            f"-> {self.valid_count} valid"
        )
