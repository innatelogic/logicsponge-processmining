"""Tests for the oracle ensemble accuracy baseline."""

from datetime import UTC, datetime

import pytest

from logicsponge.processmining.batch_helpers import build_strategies
from logicsponge.processmining.miners import CheatingMiner
from logicsponge.processmining.types import Event
from tests.test_ordered_model_selection import FixedPredictionMiner


def event(activity: str) -> Event:
    """Build a labeled event for oracle evaluation."""
    return Event(case_id="case", activity=activity, timestamp=datetime.now(tz=UTC))


def test_cheating_miner_uses_correct_constituent_prediction() -> None:
    strategy = CheatingMiner(models=[FixedPredictionMiner("a"), FixedPredictionMiner("b")])

    _, predictions = strategy.evaluate([[event("b"), event("missing")]])

    assert predictions == ["b", "b"]
    assert strategy.oracle_hits == 1
    assert strategy.oracle_misses == 1
    assert strategy.get_oracle_accuracy() == 0.5
    assert strategy.stats["correct_predictions"] == 1
    assert strategy.stats["wrong_predictions"] == 1
    assert strategy.active_model_trace == [1, -1]


def test_cheating_miner_uses_soft_voting_without_a_label() -> None:
    strategy = CheatingMiner(models=[FixedPredictionMiner("a"), FixedPredictionMiner("b")])

    probabilities = strategy.voting_probs(
        [
            {"a": 0.51, "b": 0.49},
            {"a": 0.51, "b": 0.49},
            {"a": 0.0, "b": 1.0},
        ]
    )

    assert probabilities["b"] > probabilities["a"]


def test_cheating_miner_rejects_perplexity() -> None:
    strategy = CheatingMiner(models=[FixedPredictionMiner("a")])

    with pytest.raises(NotImplementedError, match="undefined"):
        strategy.evaluate([[event("a")]], compute_perplexity=True)


def test_batch_strategies_include_independent_cheating_baseline() -> None:
    strategies = build_strategies(
        config={},
        test_set_transformed=[],
        ngram_names=[],
        voting_ngrams=[],
        adaptive_ngram=[],
        select_best_args=[],
    )

    cheating = strategies["cheating voting"][0]
    hard_voting = strategies["hard voting"][0]

    assert isinstance(cheating, CheatingMiner)
    assert cheating.models is not hard_voting.models
    assert {id(model) for model in cheating.models}.isdisjoint({id(model) for model in hard_voting.models})
