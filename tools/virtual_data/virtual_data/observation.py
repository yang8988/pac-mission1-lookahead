"""Stage-2 measurement model: true box -> measured ``BoxState`` (+ delta).

jaesung's generator currently provides identity observations only. To test
the hard mask's uncertainty handling (delta) we perturb the measured size and
weight. A small fraction of boxes gets a larger error and lower confidence;
their IDs go to ``PlanningContext.uncertain_box_ids`` so 5-2 widens the
clearance and CoG margins for them.
"""

from dataclasses import dataclass, replace

from pac_common import BoxStatus, Size3D


@dataclass(frozen=True)
class Observation:
    box: object  # measured BoxState
    truth: object  # ground-truth BoxState
    uncertain: bool
    size_error_m: tuple


def _noisy(rng, value, std, clip):
    if std <= 0:
        return value, 0.0
    err = rng.gauss(0.0, std)
    if clip > 0:
        err = max(-clip, min(clip, err))
    measured = max(0.01, round(value + err, 4))
    return measured, measured - value


def observe(truth, rng, cfg, stamp_sec):
    uncertain = rng.random() < cfg.uncertain_probability
    std = cfg.uncertain_dimension_noise_std_m if uncertain else cfg.dimension_noise_std_m
    clip = 3.0 * std if uncertain else cfg.dimension_noise_clip_m
    sx, ex = _noisy(rng, truth.size.x, std, clip)
    sy, ey = _noisy(rng, truth.size.y, std, clip)
    sz, ez = _noisy(rng, truth.size.z, std, clip)
    weight = truth.weight_kg
    if cfg.weight_noise_std_ratio > 0:
        weight = max(0.01, round(weight * (1.0 + rng.gauss(0.0, cfg.weight_noise_std_ratio)), 4))
    measured = replace(
        truth,
        size=Size3D(sx, sy, sz),
        weight_kg=weight,
        status=BoxStatus.READY_FOR_PICK,
        confidence=cfg.uncertain_confidence if uncertain else 1.0,
        stamp_sec=float(stamp_sec),
        source="VIRTUAL_OBSERVATION",
    )
    return Observation(measured, truth, uncertain, (ex, ey, ez))
