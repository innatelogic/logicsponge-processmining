"""Integer-valued N-gram prediction and regression-style evaluation."""

from __future__ import annotations

from collections import Counter, defaultdict, deque
from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING, Any

from logicsponge.processmining.numeric_metrics import (
    aggregate_numeric_distribution,
    resolve_numeric_error_metric,
)
from logicsponge.processmining.types import (
    CaseId,
    NumericAggregator,
    NumericErrorFunction,
    NumericErrorName,
    NumericEstimatorName,
    NumericEvaluationResult,
    NumericEvent,
    NumericMetrics,
)

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping, Sequence

_MAX_TRIM_FRACTION = 0.5


@dataclass(frozen=True, slots=True)
class NumericNGramConfig:
    """Configuration for integer-valued N-gram estimation."""

    window_length: int = 3
    estimator: NumericEstimatorName | NumericAggregator = "mean"
    min_observations: int = 1
    backoff: bool = True
    quantile: float = 0.5
    trim_fraction: float = 0.1
    case_id_key: str = "case_id"
    value_key: str = "value"
    timestamp_key: str | None = None
    coerce_values: bool = False

    def __post_init__(self) -> None:
        """Validate configuration values."""
        if self.window_length < 0:
            msg = "window_length cannot be negative."
            raise ValueError(msg)
        if self.min_observations < 1:
            msg = "min_observations must be positive."
            raise ValueError(msg)
        if not 0.0 <= self.quantile <= 1.0:
            msg = "quantile must be between 0 and 1."
            raise ValueError(msg)
        if not 0.0 <= self.trim_fraction < _MAX_TRIM_FRACTION:
            msg = "trim_fraction must be at least 0 and less than 0.5."
            raise ValueError(msg)


def _integer_value(raw_value: Any, *, coerce: bool) -> int:  # noqa: ANN401
    if isinstance(raw_value, bool):
        msg = "Boolean values are not valid numeric observations."
        raise TypeError(msg)
    if isinstance(raw_value, int):
        return raw_value
    if not coerce:
        msg = f"Expected an integer observation, received {type(raw_value).__name__}."
        raise TypeError(msg)
    if isinstance(raw_value, float) and not raw_value.is_integer():
        msg = f"Cannot coerce non-integral value {raw_value!r} to an integer."
        raise ValueError(msg)
    try:
        return int(raw_value)
    except (TypeError, ValueError) as error:
        msg = f"Cannot coerce {raw_value!r} to an integer."
        raise ValueError(msg) from error


def numeric_event_from_mapping(
    row: Mapping[str, Any],
    *,
    case_id_key: str = "case_id",
    value_key: str = "value",
    timestamp_key: str | None = "timestamp",
    coerce_values: bool = False,
) -> NumericEvent:
    """Convert a dataset row to the canonical numeric event representation."""
    if case_id_key not in row:
        msg = f"Missing case identifier column: {case_id_key}"
        raise KeyError(msg)
    if value_key not in row:
        msg = f"Missing numeric value column: {value_key}"
        raise KeyError(msg)

    case_id = row[case_id_key]
    if not isinstance(case_id, (str, tuple)):
        case_id = str(case_id)
    timestamp = row.get(timestamp_key) if timestamp_key is not None else None
    if timestamp is not None and not isinstance(timestamp, datetime):
        msg = "Numeric event timestamps must be datetime objects or None."
        raise TypeError(msg)
    return NumericEvent(
        case_id=case_id,
        value=_integer_value(row[value_key], coerce=coerce_values),
        timestamp=timestamp,
    )


class NumericNGram:
    """Learn next-integer distributions for suffixes of per-case histories."""

    def __init__(self, config: NumericNGramConfig | None = None) -> None:
        """Initialize an empty numeric N-gram."""
        self.config = config or NumericNGramConfig()
        self._target_counts: defaultdict[tuple[int, ...], Counter[int]] = defaultdict(Counter)
        self._case_histories: dict[CaseId, deque[int]] = {}
        self.modified_cases: set[CaseId] = set()

    @property
    def estimator_name(self) -> str:
        """Return a stable display name for the configured estimator."""
        estimator = self.config.estimator
        return getattr(estimator, "__name__", "custom") if callable(estimator) else estimator

    @property
    def target_counts(self) -> dict[tuple[int, ...], Counter[int]]:
        """Return a defensive copy of learned next-value counts."""
        return {context: counts.copy() for context, counts in self._target_counts.items()}

    def reset(self) -> None:
        """Clear learned distributions and streaming case histories."""
        self._target_counts.clear()
        self._case_histories.clear()
        self.modified_cases.clear()

    def _normalize_event(self, event: Mapping[str, Any]) -> NumericEvent:
        return numeric_event_from_mapping(
            event,
            case_id_key=self.config.case_id_key,
            value_key=self.config.value_key,
            timestamp_key=self.config.timestamp_key,
            coerce_values=self.config.coerce_values,
        )

    def _contexts(self, history: Sequence[int]) -> list[tuple[int, ...]]:
        history_tuple = tuple(history)
        maximum = min(self.config.window_length, len(history_tuple))
        return [() if length == 0 else history_tuple[-length:] for length in range(maximum + 1)]

    def _matching_counts(self, history: Sequence[int]) -> tuple[tuple[int, ...], Counter[int] | None]:
        contexts = self._contexts(history)
        candidates = reversed(contexts) if self.config.backoff else contexts[-1:]
        for context in candidates:
            counts = self._target_counts.get(context)
            if counts is not None and sum(counts.values()) >= self.config.min_observations:
                return context, counts
        return contexts[-1], None

    def get_state_from_case(self, case_id: CaseId) -> tuple[int, ...]:
        """Return the current bounded history for a case."""
        return tuple(self._case_histories.get(case_id, ()))

    def distribution_for_history(self, history: Sequence[int]) -> dict[int, float]:
        """Return the normalized next-value distribution for a history."""
        _, counts = self._matching_counts(history)
        if counts is None:
            return {}
        total = sum(counts.values())
        return {value: count / total for value, count in sorted(counts.items())}

    def metrics_for_history(self, history: Sequence[int]) -> NumericMetrics:
        """Return numeric prediction metrics for an arbitrary history."""
        context, counts = self._matching_counts(history)
        if counts is None:
            return NumericMetrics(state_id=context, prediction=None, distribution={}, estimator=self.estimator_name)
        total = sum(counts.values())
        prediction = aggregate_numeric_distribution(
            counts,
            self.config.estimator,
            quantile=self.config.quantile,
            trim_fraction=self.config.trim_fraction,
        )
        distribution = {value: count / total for value, count in sorted(counts.items())}
        return NumericMetrics(
            state_id=context,
            prediction=prediction,
            distribution=distribution,
            estimator=self.estimator_name,
        )

    def case_metrics(self, case_id: CaseId) -> NumericMetrics:
        """Return prediction metrics for a streaming case."""
        return self.metrics_for_history(self.get_state_from_case(case_id))

    def predict_next(self, case_id: CaseId) -> float | None:
        """Predict the next numeric value for a streaming case."""
        return self.case_metrics(case_id).prediction

    def update(self, event: Mapping[str, Any]) -> None:
        """Learn an observed integer as the target following its current context."""
        normalized = self._normalize_event(event)
        case_id = normalized["case_id"]
        value = normalized["value"]
        history = self._case_histories.setdefault(case_id, deque(maxlen=self.config.window_length))
        updated_contexts = set(self._contexts(history))
        for context in updated_contexts:
            self._target_counts[context][value] += 1
        history.append(value)

        self.modified_cases = {
            candidate_case
            for candidate_case, candidate_history in self._case_histories.items()
            if self._matching_counts(candidate_history)[0] in updated_contexts
        }

    def fit(self, sequences: Iterable[Iterable[Mapping[str, Any]]], *, reset: bool = False) -> None:
        """Train on grouped event sequences."""
        if reset:
            self.reset()
        for sequence in sequences:
            for event in sequence:
                self.update(event)

    def get_modified_cases(self) -> set[CaseId]:
        """Return cases whose current predictions may have changed."""
        return set(self.modified_cases)


class NumericNGramMiner:
    """Train, predict, and evaluate an integer-valued N-gram."""

    def __init__(
        self,
        algorithm: NumericNGram,
        *,
        error_metric: NumericErrorName | NumericErrorFunction = "smape",
    ) -> None:
        """Initialize a numeric miner and its default evaluation metric."""
        self.algorithm = algorithm
        self.error_metric = error_metric

    def update(self, event: Mapping[str, Any]) -> None:
        """Update the numeric N-gram with one event."""
        self.algorithm.update(event)

    def fit(self, sequences: Iterable[Iterable[Mapping[str, Any]]], *, reset: bool = False) -> None:
        """Train the numeric N-gram on grouped sequences."""
        self.algorithm.fit(sequences, reset=reset)

    def case_metrics(self, case_id: CaseId) -> NumericMetrics:
        """Return prediction metrics for one streaming case."""
        return self.algorithm.case_metrics(case_id)

    def predict_next(self, case_id: CaseId) -> float | None:
        """Predict the next value for one streaming case."""
        return self.algorithm.predict_next(case_id)

    def evaluate(
        self,
        sequences: Iterable[Iterable[Mapping[str, Any]]],
        *,
        error_metric: NumericErrorName | NumericErrorFunction | None = None,
        update: bool = False,
        warm_start_cases: bool = False,
    ) -> NumericEvaluationResult:
        """Evaluate without equality assumptions, optionally updating online."""
        resolved_metric = resolve_numeric_error_metric(error_metric or self.error_metric)
        histories: dict[CaseId, deque[int]] = {}
        predictions: list[float | None] = []
        actuals: list[int] = []
        errors: list[float | None] = []

        for sequence in sequences:
            for event in sequence:
                normalized = self.algorithm._normalize_event(event)  # noqa: SLF001
                case_id = normalized["case_id"]
                actual = normalized["value"]
                if update:
                    metrics = self.algorithm.case_metrics(case_id)
                else:
                    if case_id not in histories:
                        initial = self.algorithm.get_state_from_case(case_id) if warm_start_cases else ()
                        histories[case_id] = deque(initial, maxlen=self.algorithm.config.window_length)
                    metrics = self.algorithm.metrics_for_history(histories[case_id])

                prediction = metrics.prediction
                predictions.append(prediction)
                actuals.append(actual)
                errors.append(None if prediction is None else resolved_metric.point_error(actual, prediction))

                if update:
                    self.algorithm.update(event)
                else:
                    histories[case_id].append(actual)

        observed_errors = [error for error in errors if error is not None]
        return NumericEvaluationResult(
            metric=resolved_metric.name,
            score=resolved_metric.aggregate(observed_errors),
            total_observations=len(actuals),
            evaluated_observations=len(observed_errors),
            missing_predictions=len(actuals) - len(observed_errors),
            predictions=predictions,
            actuals=actuals,
            errors=errors,
        )
