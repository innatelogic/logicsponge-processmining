"""Tests for integer-valued N-gram prediction."""

from datetime import UTC, datetime

import pytest

from logicsponge.processmining.numeric import (
    NumericNGram,
    NumericNGramConfig,
    NumericNGramMiner,
    numeric_event_from_mapping,
)
from logicsponge.processmining.numeric_metrics import (
    aggregate_numeric_distribution,
    symmetric_absolute_percentage_error,
)


def numeric_event(case_id: str, value: int) -> dict:
    return {"case_id": case_id, "value": value, "timestamp": None}


def training_sequences() -> list[list[dict]]:
    return [
        [numeric_event("a", value) for value in (1, 2, 4)],
        [numeric_event("b", value) for value in (1, 2, 8)],
    ]


def largest_observed_value(distribution: dict[int, int]) -> float:
    return float(max(distribution))


@pytest.mark.parametrize(
    ("estimator", "expected"),
    [
        ("mean", 7.5),
        ("median", 10.0),
        ("mode", 10.0),
    ],
)
def test_builtin_distribution_estimators(estimator: str, expected: float) -> None:
    assert aggregate_numeric_distribution({0: 1, 10: 3}, estimator) == expected


def test_parameterized_and_custom_distribution_estimators() -> None:
    counts = {0: 1, 10: 3}

    assert aggregate_numeric_distribution(counts, "quantile", quantile=0.25) == 7.5
    assert aggregate_numeric_distribution(counts, "trimmed_mean", trim_fraction=0.25) == 10.0
    assert aggregate_numeric_distribution(counts, largest_observed_value) == 10.0


def test_numeric_ngram_learns_distribution_and_backs_off() -> None:
    algorithm = NumericNGram(NumericNGramConfig(window_length=2))
    algorithm.fit(training_sequences())

    exact = algorithm.metrics_for_history((1, 2))
    backed_off = algorithm.metrics_for_history((99, 2))

    assert exact.state_id == (1, 2)
    assert exact.distribution == {4: 0.5, 8: 0.5}
    assert exact.prediction == 6.0
    assert backed_off.state_id == (2,)
    assert backed_off.prediction == 6.0


def test_numeric_ngram_can_disable_backoff() -> None:
    algorithm = NumericNGram(NumericNGramConfig(window_length=2, backoff=False))
    algorithm.fit(training_sequences())

    metrics = algorithm.metrics_for_history((99, 2))

    assert metrics.state_id == (99, 2)
    assert metrics.prediction is None
    assert metrics.distribution == {}


def test_evaluation_defaults_to_smape_without_mutating_model() -> None:
    algorithm = NumericNGram(NumericNGramConfig(window_length=2))
    miner = NumericNGramMiner(algorithm)
    miner.fit(training_sequences())
    counts_before = algorithm.target_counts
    state_before = algorithm.get_state_from_case("a")

    result = miner.evaluate([[numeric_event("test", value) for value in (1, 2, 4)]])

    assert result.metric == "smape"
    assert result.predictions == [3.0, 2.0, 6.0]
    assert result.actuals == [1, 2, 4]
    assert result.score == pytest.approx((100.0 + 0.0 + 40.0) / 3)
    assert result.evaluated_observations == 3
    assert result.missing_predictions == 0
    assert algorithm.target_counts == counts_before
    assert algorithm.get_state_from_case("a") == state_before


def test_evaluation_metric_is_selectable_or_custom() -> None:
    algorithm = NumericNGram(NumericNGramConfig(window_length=2))
    miner = NumericNGramMiner(algorithm, error_metric="mae")
    miner.fit(training_sequences())
    sequence = [[numeric_event("test", value) for value in (1, 2, 4)]]

    mae_result = miner.evaluate(sequence)
    custom_result = miner.evaluate(sequence, error_metric=lambda actual, predicted: abs(actual - predicted) ** 3)

    assert mae_result.metric == "mae"
    assert mae_result.score == pytest.approx(4 / 3)
    assert custom_result.metric == "<lambda>"
    assert custom_result.score == pytest.approx(16 / 3)


def test_smape_handles_zero_values() -> None:
    assert symmetric_absolute_percentage_error(0, 0) == 0.0
    assert symmetric_absolute_percentage_error(0, 5) == 200.0


def test_dataset_columns_and_integer_coercion_are_configurable() -> None:
    algorithm = NumericNGram(
        NumericNGramConfig(
            window_length=1,
            case_id_key="session",
            value_key="visits",
            coerce_values=True,
        )
    )
    algorithm.fit([[{"session": "one", "visits": "12"}, {"session": "one", "visits": "15"}]])

    assert algorithm.get_state_from_case("one") == (15,)
    assert algorithm.metrics_for_history((12,)).prediction == 15.0


def test_numeric_event_conversion_validates_input() -> None:
    timestamp = datetime.now(tz=UTC)
    converted = numeric_event_from_mapping(
        {"session": 10, "visits": "4", "observed_at": timestamp},
        case_id_key="session",
        value_key="visits",
        timestamp_key="observed_at",
        coerce_values=True,
    )

    assert converted == {"case_id": "10", "value": 4, "timestamp": timestamp}
    with pytest.raises(TypeError):
        numeric_event_from_mapping({"case_id": "a", "value": 1.5})
