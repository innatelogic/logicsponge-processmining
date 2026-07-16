"""Tests for forward-only and bidirectional ordered model selection."""

from datetime import UTC, datetime

from logicsponge.processmining.miners import BidirectionalPromotion, Promotion, StreamingMiner
from logicsponge.processmining.types import ActivityName, CaseId, ComposedState, Event, Metrics, OrderedModelState


class FixedPredictionMiner(StreamingMiner):
    """Small stateful miner with a fixed prediction for selection tests."""

    def __init__(self, prediction: ActivityName) -> None:
        super().__init__()
        self.prediction = prediction
        self.initial_state = 0
        self.case_states: dict[CaseId, int] = {}
        self.update_count = 0

    def get_state_from_case(self, case_id: CaseId) -> int:
        return self.case_states.get(case_id, 0)

    def get_state_info(self, state_id: ComposedState | None) -> None:  # noqa: ARG002
        return None

    def get_modified_cases(self) -> set[CaseId]:
        return set(self.case_states)

    def propagate_config(self) -> None:
        return None

    def update(self, event: Event) -> None:
        self.update_count += 1
        self.case_states[event["case_id"]] = self.case_states.get(event["case_id"], 0) + 1

    def next_state(self, current_state: ComposedState | None, activity: ActivityName) -> int:  # noqa: ARG002
        return int(current_state or 0) + 1

    def state_metrics(self, state: ComposedState | None) -> Metrics:
        return Metrics(state_id=state, probs={self.prediction: 1.0}, predicted_delays={})

    def case_metrics(self, case_id: CaseId) -> Metrics:
        return self.state_metrics(self.get_state_from_case(case_id))

    def state_act_likelihood(self, state: ComposedState | None, next_activity: ActivityName) -> float:  # noqa: ARG002
        return float(next_activity == self.prediction)


def event(activity: str) -> Event:
    """Build a test event."""
    return Event(case_id="case", activity=activity, timestamp=datetime.now(tz=UTC))


def test_promotion_keeps_legacy_pair_state_and_two_active_models() -> None:
    models = [FixedPredictionMiner(activity) for activity in ("a", "b", "c")]
    strategy = Promotion(models=models, min_votes=20)

    assert isinstance(strategy.initial_state, tuple)
    assert len(strategy.initial_state) == 2

    strategy.update(event("b"))

    assert [model.update_count for model in models] == [1, 1, 0]
    assert strategy.current_index == 0


def test_bidirectional_strategy_promotes_and_demotes() -> None:
    models = [FixedPredictionMiner(activity) for activity in ("a", "b", "c")]
    strategy = BidirectionalPromotion(models=models, threshold=0.0, min_votes=2)

    assert strategy.initial_state == OrderedModelState(previous=None, current=0, next=0)

    strategy.update(event("b"))
    strategy.update(event("b"))

    assert strategy.current_index == 1
    assert isinstance(strategy.initial_state, OrderedModelState)
    assert strategy.initial_state.previous == 0
    assert [model.update_count for model in models] == [2, 2, 0]

    strategy.update(event("a"))
    strategy.update(event("a"))

    assert strategy.current_index == 0
    assert strategy.demotion_votes == 0
    assert [model.update_count for model in models] == [4, 4, 2]


def test_bidirectional_strategy_respects_switch_cooldown() -> None:
    models = [FixedPredictionMiner(activity) for activity in ("a", "b")]
    strategy = BidirectionalPromotion(
        models=models,
        threshold=0.0,
        min_votes=1,
        min_demotion_votes=1,
        cooldown_predictions=2,
    )

    strategy.update(event("b"))
    assert strategy.current_index == 1

    strategy.update(event("a"))
    assert strategy.current_index == 1
    strategy.update(event("a"))
    assert strategy.current_index == 0
