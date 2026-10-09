"""Stage 4 configuration (``config/taehyeon/highlevel.yaml``).

Units: seconds, metres, kilograms. Rewards are expressed in "pallet volume
fractions" (the same unit as donghan's 5-4 future value: added volume /
pallet capacity) so time and space can be traded explicitly.
"""

from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path


@dataclass(frozen=True)
class BufferConfig:
    # Buffer shelves on both sides of the pallet; one box per slot.
    slots: int = 4
    # Travel time pick-station -> slot -> pallet per slot (s). Slots nearer
    # the pallet are cheaper. Empty = generated from the layout below.
    travel_time_s: tuple = ()
    base_travel_time_s: float = 4.0
    per_slot_extra_s: float = 1.0
    # "박스당 1회만" (one buffer visit per box) holds by construction: only
    # the current conveyor box can be buffered and a retrieved box is placed.

    def travel_times(self):
        if self.travel_time_s:
            if len(self.travel_time_s) != self.slots:
                raise ValueError("travel_time_s needs one value per slot")
            return tuple(float(t) for t in self.travel_time_s)
        # alternate left/right shelves: 0,1 nearest, 2,3 next, ...
        return tuple(
            self.base_travel_time_s + self.per_slot_extra_s * (i // 2) for i in range(self.slots)
        )


@dataclass(frozen=True)
class TimingConfig:
    place_time_s: float = 8.0  # pick from conveyor and place (HDR50-22 cycle, assumed)
    pallet_change_time_s: float = 60.0  # T_change: CLOSE -> PALLET_CHANGE -> new pallet
    repack_move_time_s: float = 12.0  # pick a placed box and re-place it


@dataclass(frozen=True)
class RewardConfig:
    volume_weight: float = 1.0  # + placed volume / pallet volume
    # On PALLET_CLOSE: - (1 - fill) * weight. Summed over pallets this is
    # "pallets used - total volume", i.e. minimise the number of pallets.
    close_waste_weight: float = 1.0
    # cost of robot time, in pallet-volume fractions per second
    time_weight: float = 0.0005
    occupancy_weight: float = 0.002  # per occupied slot per decision
    ng_penalty: float = 0.05  # per box sent to NG (not a policy choice)


@dataclass(frozen=True)
class RuleConfig:
    # rule policy (1st step: all five actions by rule)
    good_support: float = 0.95  # current box placement considered "good"
    retrieve_margin_m: float = 0.02  # buffered box preferred if its top is this much lower
    max_buffer_age: int = 12  # decisions; older buffered boxes are retrieved first
    # scale-free forms (used when > 0, so the rule holds for any box/buffer size):
    retrieve_margin_ratio: float = 0.15  # x height of the current box
    max_buffer_age_per_slot: float = 3.0  # x buffer slots

    def margin_for(self, box):
        if self.retrieve_margin_ratio > 0:
            return self.retrieve_margin_ratio * box.size.z
        return self.retrieve_margin_m

    def age_limit(self, slots):
        if self.max_buffer_age_per_slot > 0:
            return max(1, int(round(self.max_buffer_age_per_slot * max(1, slots))))
        return self.max_buffer_age


@dataclass(frozen=True)
class CloseConfig:
    # PALLET_CLOSE rule (always a rule, never learned). Closes when no learned
    # action is feasible; additionally, with the current box not placeable and
    # no buffered box placeable, a pallet filled to at least this volume
    # fraction is closed right away instead of forcing the box into the
    # buffer (0 disables; taehyeon 2026-10-08).
    fill_before_buffer: float = 0.30  # val sweep: pallets ~even (+0.04), robot time -26 s
    # "fill": close when fill >= fill_before_buffer (fixed number, environment
    # dependent). "dead": close when at least ``dead_share`` of the expected
    # volume (current + buffer boxes + the order list's main SKUs) has no safe
    # spot on this pallet any more -- a judgement, not a tuned fill level.
    mode: str = "dead"
    dead_share: float = 1.0
    # boxes on the conveyor after the current one that the camera sees (exact
    # size/weight/order): counted in the dead share together with the order list
    visible_boxes: int = 5
    # weight of the not-yet-seen order list against the boxes actually seen
    # (current, buffer, visible): nearer boxes matter more
    order_weight: float = 1.0


@dataclass(frozen=True)
class RepackConfig:
    enabled: bool = True
    max_moves: int = 2  # relocated boxes per repack
    max_nodes: int = 24  # search budget (states expanded)
    min_gain: float = 0.0  # net gain in pallet-volume fractions after move cost


@dataclass(frozen=True)
class FeatureConfig:
    value_provider: str = "proxy"  # proxy | donghan (must match training and deployment)
    value_model_path: str = ""  # donghan: path of dual_head_ranker.json
    # the order list (SKU types, sizes, weights and quantities) is known in
    # advance, only the arrival order is not (mission brief; taehyeon
    # 2026-10-08). False hides remaining counts from the policy and stages 5.
    order_list_known: bool = True
    heightmap_cell_m: float = 0.02


@dataclass(frozen=True)
class PPOConfig:
    hidden: tuple = (64, 64)
    learning_rate: float = 3e-4
    gamma: float = 0.995
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    max_grad_norm: float = 0.5
    n_steps: int = 1024  # per update (all workers)
    batch_size: int = 128
    n_epochs: int = 6
    total_steps: int = 30000
    seed: int = 0


@dataclass(frozen=True)
class HighLevelConfig:
    buffer: BufferConfig = field(default_factory=BufferConfig)
    timing: TimingConfig = field(default_factory=TimingConfig)
    reward: RewardConfig = field(default_factory=RewardConfig)
    rules: RuleConfig = field(default_factory=RuleConfig)
    close: CloseConfig = field(default_factory=CloseConfig)
    repack: RepackConfig = field(default_factory=RepackConfig)
    features: FeatureConfig = field(default_factory=FeatureConfig)
    ppo: PPOConfig = field(default_factory=PPOConfig)

    def __post_init__(self):
        if self.buffer.slots < 0:
            raise ValueError("buffer.slots must be >= 0")
        self.buffer.travel_times()
        if not 0.0 <= self.close.fill_before_buffer <= 1.0:
            raise ValueError("close.fill_before_buffer must be in [0, 1]")
        if self.features.value_provider not in ("proxy", "donghan"):
            raise ValueError("features.value_provider must be proxy | donghan")


def _build(cls, data):
    if data is None:
        return cls()
    known = {f.name: f for f in fields(cls)}
    unknown = set(data) - set(known)
    if unknown:
        raise ValueError(f"Unknown {cls.__name__} keys: {sorted(unknown)}")
    kwargs = {}
    for name, value in data.items():
        default = known[name].default_factory() if callable(known[name].default_factory) else None
        if default is not None and is_dataclass(default):
            kwargs[name] = _build(type(default), value)
        elif isinstance(value, list):
            kwargs[name] = tuple(value)
        else:
            kwargs[name] = value
    return cls(**kwargs)


def config_from_dict(data):
    return _build(HighLevelConfig, (data or {}).get("highlevel", data or {}))


def load_highlevel_config(path=None):
    if path is None:
        return HighLevelConfig()
    import yaml

    return config_from_dict(yaml.safe_load(Path(path).read_text(encoding="utf-8")))
