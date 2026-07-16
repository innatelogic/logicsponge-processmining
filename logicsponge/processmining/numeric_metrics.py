"""Aggregation and evaluation functions for numeric sequence prediction."""

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass

from logicsponge.processmining.types import (
    NumericAggregator,
    NumericErrorFunction,
    NumericErrorName,
    NumericEstimatorName,
)

_MAX_TRIM_FRACTION = 0.5


def _validate_distribution(counts: Mapping[int, int]) -> int:
    total = sum(counts.values())
    if total <= 0 or any(count < 0 for count in counts.values()):
        msg = "A numeric distribution must contain non-negative counts with a positive total."
        raise ValueError(msg)
    return total


def weighted_mean(counts: Mapping[int, int]) -> float:
    """Return the frequency-weighted arithmetic mean."""
    total = _validate_distribution(counts)
    return sum(value * count for value, count in counts.items()) / total


def weighted_quantile(counts: Mapping[int, int], quantile: float = 0.5) -> float:
    """Return a quantile from a discrete frequency distribution."""
    if not 0.0 <= quantile <= 1.0:
        msg = "quantile must be between 0 and 1."
        raise ValueError(msg)
    total = _validate_distribution(counts)
    rank = quantile * (total - 1)
    lower_rank = math.floor(rank)
    upper_rank = math.ceil(rank)

    def value_at_rank(wanted_rank: int) -> int:
        cumulative = 0
        for value, count in sorted(counts.items()):
            cumulative += count
            if cumulative > wanted_rank:
                return value
        return max(counts)  # pragma: no cover - guarded by distribution validation

    lower_value = value_at_rank(lower_rank)
    upper_value = value_at_rank(upper_rank)
    interpolation = rank - lower_rank
    return lower_value + interpolation * (upper_value - lower_value)


def weighted_median(counts: Mapping[int, int]) -> float:
    """Return the median from a discrete frequency distribution."""
    return weighted_quantile(counts, 0.5)


def weighted_mode(counts: Mapping[int, int]) -> float:
    """Return the mean of all values tied for the highest frequency."""
    _validate_distribution(counts)
    highest_count = max(counts.values())
    modes = [value for value, count in counts.items() if count == highest_count]
    return sum(modes) / len(modes)


def weighted_trimmed_mean(counts: Mapping[int, int], trim_fraction: float = 0.1) -> float:
    """Return a weighted mean after trimming both tails by observation count."""
    if not 0.0 <= trim_fraction < _MAX_TRIM_FRACTION:
        msg = "trim_fraction must be at least 0 and less than 0.5."
        raise ValueError(msg)
    total = _validate_distribution(counts)
    trim_count = total * trim_fraction
    weighted_values = [[float(value), float(count)] for value, count in sorted(counts.items())]

    remaining_trim = trim_count
    for item in weighted_values:
        removed = min(item[1], remaining_trim)
        item[1] -= removed
        remaining_trim -= removed
        if remaining_trim == 0:
            break

    remaining_trim = trim_count
    for item in reversed(weighted_values):
        removed = min(item[1], remaining_trim)
        item[1] -= removed
        remaining_trim -= removed
        if remaining_trim == 0:
            break

    remaining = sum(weight for _, weight in weighted_values)
    return sum(value * weight for value, weight in weighted_values) / remaining


def aggregate_numeric_distribution(
    counts: Mapping[int, int],
    estimator: NumericEstimatorName | NumericAggregator = "mean",
    *,
    quantile: float = 0.5,
    trim_fraction: float = 0.1,
) -> float:
    """Aggregate a next-value frequency distribution into one prediction."""
    if callable(estimator):
        return float(estimator(counts))
    if estimator == "mean":
        return weighted_mean(counts)
    if estimator == "median":
        return weighted_median(counts)
    if estimator == "mode":
        return weighted_mode(counts)
    if estimator == "quantile":
        return weighted_quantile(counts, quantile)
    if estimator == "trimmed_mean":
        return weighted_trimmed_mean(counts, trim_fraction)
    msg = f"Unknown numeric estimator: {estimator}"
    raise ValueError(msg)


def symmetric_absolute_percentage_error(actual: float, predicted: float) -> float:
    """Return one symmetric absolute percentage error in the range 0 to 200."""
    denominator = abs(actual) + abs(predicted)
    return 0.0 if denominator == 0 else 200.0 * abs(predicted - actual) / denominator


def mean_absolute_error(actual: float, predicted: float) -> float:
    """Return one absolute error."""
    return abs(predicted - actual)


def absolute_percentage_error(actual: float, predicted: float) -> float:
    """Return one absolute percentage error, using a stable zero denominator."""
    denominator = max(abs(actual), 1e-12)
    return 100.0 * abs(predicted - actual) / denominator


def squared_error(actual: float, predicted: float) -> float:
    """Return one squared error."""
    return (predicted - actual) ** 2


def _mean(errors: Sequence[float]) -> float | None:
    return sum(errors) / len(errors) if errors else None


def _root_mean(errors: Sequence[float]) -> float | None:
    mean = _mean(errors)
    return math.sqrt(mean) if mean is not None else None


@dataclass(frozen=True, slots=True)
class ResolvedNumericErrorMetric:
    """Point-error and aggregation behavior for one evaluation metric."""

    name: str
    point_error: NumericErrorFunction
    aggregate: Callable[[Sequence[float]], float | None]


def resolve_numeric_error_metric(
    metric: NumericErrorName | NumericErrorFunction = "smape",
) -> ResolvedNumericErrorMetric:
    """Resolve a built-in metric name or custom point-error function."""
    if callable(metric):
        return ResolvedNumericErrorMetric(
            name=getattr(metric, "__name__", "custom"),
            point_error=metric,
            aggregate=_mean,
        )
    metrics = {
        "smape": ResolvedNumericErrorMetric("smape", symmetric_absolute_percentage_error, _mean),
        "mae": ResolvedNumericErrorMetric("mae", mean_absolute_error, _mean),
        "mape": ResolvedNumericErrorMetric("mape", absolute_percentage_error, _mean),
        "mse": ResolvedNumericErrorMetric("mse", squared_error, _mean),
        "rmse": ResolvedNumericErrorMetric("rmse", squared_error, _root_mean),
    }
    if metric not in metrics:
        msg = f"Unknown numeric error metric: {metric}"
        raise ValueError(msg)
    return metrics[metric]
