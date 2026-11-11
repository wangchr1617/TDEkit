"""Summarize comparable occupancy endpoints across independent trials."""

from collections import Counter
from dataclasses import dataclass
from math import isfinite, sqrt
from statistics import mean, stdev
from typing import Callable, Iterable, Optional

from .occupancy import OccupancyFrame


@dataclass(frozen=True)
class OccupancyTrial:
    energy_ev: float
    direction: str
    repeat_id: str
    endpoint: Optional[OccupancyFrame]


@dataclass(frozen=True)
class CountSummary:
    mean: Optional[float]
    sample_sd: Optional[float]


@dataclass(frozen=True)
class OccupancySummary:
    energy_ev: float
    direction: str
    observation_time_fs: float
    n_valid: int
    n_clustered: int
    missing_repeats: tuple[str, ...]
    vacancies: CountSummary
    overoccupied_sites: CountSummary
    excess_atoms: CountSummary
    antisites: CountSummary
    cluster_histogram: Optional[dict[str, dict[int, int]]]


def _count_summary(frames, attribute):
    values = [getattr(frame, attribute) for frame in frames]
    return CountSummary(mean(values) if values else None,
                        stdev(values) if len(values) > 1 else None)


def _group_trials(trials, observation_time_fs, time_tolerance_fs):
    if (not isfinite(observation_time_fs) or observation_time_fs < 0
            or not isfinite(time_tolerance_fs) or time_tolerance_fs < 0):
        raise ValueError("Expected a nonnegative finite observation time and tolerance")
    groups = {}
    for trial in trials:
        if (not isfinite(trial.energy_ev) or trial.energy_ev < 0
                or not trial.direction.strip() or not trial.repeat_id.strip()):
            raise ValueError("Expected finite nonnegative energy and nonempty direction and repeat IDs")
        key = trial.energy_ev, trial.direction
        repeats = groups.setdefault(key, {})
        if trial.repeat_id in repeats:
            raise ValueError(f"Duplicate repeat {trial.repeat_id!r} for {key}")
        frame = trial.endpoint
        if frame is not None and (frame.time_fs is None or not isfinite(frame.time_fs)
                                  or abs(frame.time_fs - observation_time_fs) > time_tolerance_fs):
            raise ValueError(f"Repeat {trial.repeat_id!r} differs from the observation time")
        repeats[trial.repeat_id] = frame

    return groups


def summarize_trials(trials: Iterable[OccupancyTrial], *, observation_time_fs: float,
                     time_tolerance_fs: float = 1e-5) -> tuple[OccupancySummary, ...]:
    """Group comparable endpoints by exact energy and direction label.

    Each repeat contributes one endpoint. The caller defines the shared
    simulation protocol, reference, cutoff, and independent repeat IDs.
    """
    groups = _group_trials(trials, observation_time_fs, time_tolerance_fs)
    summaries = []
    for (energy, direction), repeats in groups.items():
        frames = [frame for frame in repeats.values() if frame is not None]
        histogram = {}
        clustered = 0
        for frame in frames:
            counts = frame.cluster_size_histogram()
            if counts is not None:
                clustered += 1
                for kind, sizes in counts.items():
                    histogram.setdefault(kind, Counter()).update(sizes)
        summaries.append(OccupancySummary(
            energy, direction, observation_time_fs, len(frames), clustered,
            tuple(key for key, frame in repeats.items() if frame is None),
            _count_summary(frames, "vacancy_count"),
            _count_summary(frames, "overoccupied_site_count"),
            _count_summary(frames, "excess_atom_count"),
            _count_summary(frames, "antisite_count"),
            {kind: dict(sizes) for kind, sizes in histogram.items()} if clustered else None,
        ))
    return tuple(summaries)


@dataclass(frozen=True)
class DamageProbability:
    energy_ev: float
    n_damaged: int
    n_valid: int
    missing_repeats: tuple[str, ...]
    probability: Optional[float]
    wilson_95: Optional[tuple[float, float]]


@dataclass(frozen=True)
class DirectionDamageScan:
    direction: str
    observation_time_fs: float
    threshold_probability: float
    points: tuple[DamageProbability, ...]
    upward_brackets_ev: tuple[tuple[float, float], ...]
    observed_decrease: bool


def _probability_point(energy, repeats, damaged):
    decisions = [damaged(frame) for frame in repeats.values() if frame is not None]
    if any(type(decision) is not bool for decision in decisions):
        raise ValueError("The damage criterion must return a boolean")
    n = len(decisions)
    k = sum(decisions)
    missing = tuple(key for key, frame in repeats.items() if frame is None)
    if not n:
        return DamageProbability(energy, k, n, missing, None, None)
    probability = k / n
    z = 1.959963984540054
    z2 = z * z
    denominator = 1 + z2 / n
    center = (probability + z2 / (2 * n)) / denominator
    half_width = z * sqrt((probability * (1 - probability) + z2 / (4 * n)) / n) / denominator
    interval = (max(0.0, center - half_width), min(1.0, center + half_width))
    return DamageProbability(energy, k, n, missing, probability, interval)


def scan_damage_probability(
    trials: Iterable[OccupancyTrial],
    *,
    observation_time_fs: float,
    damaged: Callable[[OccupancyFrame], bool],
    threshold_probability: float = 0.5,
    time_tolerance_fs: float = 1e-5,
) -> tuple[DirectionDamageScan, ...]:
    """Report empirical probabilities and adjacent sampled crossing intervals.

    The bracket uses point estimates. Wilson intervals describe individual
    probabilities and are not confidence intervals on the crossing energy.
    """
    if not 0 < threshold_probability <= 1:
        raise ValueError("Expected a threshold probability in (0, 1]")
    groups = _group_trials(trials, observation_time_fs, time_tolerance_fs)
    directions = {}
    for (energy, direction), repeats in groups.items():
        directions.setdefault(direction, []).append(_probability_point(energy, repeats, damaged))

    scans = []
    for direction, points in directions.items():
        points.sort(key=lambda point: point.energy_ev)
        brackets = []
        decrease = False
        for lower, upper in zip(points, points[1:]):
            lo, hi = lower.probability, upper.probability
            if lo is None or hi is None:
                continue
            if lo < threshold_probability <= hi:
                brackets.append((lower.energy_ev, upper.energy_ev))
            decrease |= hi < lo
        scans.append(DirectionDamageScan(direction, observation_time_fs, threshold_probability,
                                         tuple(points), tuple(brackets), decrease))
    return tuple(scans)
