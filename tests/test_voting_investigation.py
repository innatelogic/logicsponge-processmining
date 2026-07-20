"""Tests for the standalone voting investigation engine and dashboard."""

from pathlib import Path

from logicsponge.processmining.voting_dashboard import create_dashboard, load_results
from logicsponge.processmining.voting_investigation import (
    DecisionRule,
    GlobalAccuracyRule,
    ModelSpec,
    VotingInvestigator,
    analyze_rule_impacts,
    build_soft_failure_analysis,
    build_summary,
    default_hypotheses,
    discover_condition_hypotheses,
    evaluate_hypotheses,
    evaluate_weighted_conditions,
    save_results,
)
from tests.test_ordered_model_selection import FixedPredictionMiner, event


def fixed_specs() -> list[ModelSpec]:
    return [
        ModelSpec("model_a", 1, lambda: FixedPredictionMiner("a")),
        ModelSpec("model_b1", 2, lambda: FixedPredictionMiner("b")),
        ModelSpec("model_b2", 3, lambda: FixedPredictionMiner("b")),
    ]


def test_diagnostics_explain_recoverable_soft_vote_failure() -> None:
    investigator = VotingInvestigator(specs=fixed_specs())

    rows = investigator.diagnose([[event("a"), event("b")]], split="test")

    assert rows[0]["soft_prediction"] == "b"
    assert rows[0]["oracle_prediction"] == "a"
    assert rows[0]["correct_models"] == ["model_a"]
    assert rows[0]["oracle_gap"] is True
    assert rows[0]["agreement_count"] == 2
    assert rows[1]["soft_correct"] is True
    assert "hard_prediction" not in rows[0]


def test_hypotheses_fit_on_calibration_and_annotate_test_rows() -> None:
    investigator = VotingInvestigator(specs=fixed_specs())
    calibration = investigator.diagnose([[event("a"), event("a")]], split="calibration")
    test_rows = investigator.diagnose([[event("a")]], split="test")

    summaries = evaluate_hypotheses(calibration, test_rows, rules=[GlobalAccuracyRule()])

    assert summaries[0]["accuracy"] == 1.0
    assert test_rows[0]["rule_models"]["best calibration accuracy"] == "model_a"
    assert test_rows[0]["rule_predictions"]["best calibration accuracy"] == "a"


def test_rule_impact_counts_soft_recoveries_and_harms() -> None:
    investigator = VotingInvestigator(specs=fixed_specs())
    calibration = investigator.diagnose([[event("a"), event("a")]], split="calibration")
    test_rows = investigator.diagnose([[event("a"), event("b")]], split="test")
    evaluate_hypotheses(calibration, test_rows, rules=[GlobalAccuracyRule()])

    impact = analyze_rule_impacts(test_rows)[0]

    assert impact["recoveries"] == 1
    assert impact["harms"] == 1
    assert impact["net_correct"] == 0
    assert impact["soft_error_recall"] == 1.0
    assert impact["resulting_accuracy"] == 0.5


def test_conditions_are_learned_separately_and_can_be_weighted() -> None:
    investigator = VotingInvestigator(specs=fixed_specs())
    calibration = investigator.diagnose([[event("a"), event("a")]], split="calibration")
    test_rows = investigator.diagnose([[event("a")]], split="test")

    conditions, source = discover_condition_hypotheses(calibration, test_rows, minimum_support=1)
    chosen = conditions[0]
    impact = evaluate_weighted_conditions(test_rows, conditions, selected_ids={chosen["id"]})

    assert source == "calibration"
    assert chosen["recommended_model"] == "model_a"
    assert chosen["id"] in test_rows[0]["condition_matches"]
    assert impact["recoveries"] == 1
    assert impact["harms"] == 0
    assert impact["net_accuracy_delta"] == 1.0


def test_advanced_hypotheses_and_delayed_feedback_hook() -> None:
    investigator = VotingInvestigator(specs=fixed_specs())
    calibration = investigator.diagnose([[event("a"), event("a")]], split="calibration")
    test_rows = investigator.diagnose([[event("a"), event("b")]], split="test")

    names = {rule.name for rule in default_hypotheses()}
    assert "hierarchical reliability (support 3)" in names
    assert "stacked rule portfolio (support 5)" in names
    assert "delayed-feedback adaptive (decay 0.94)" in names

    class LifecycleRule(DecisionRule):
        name = "lifecycle probe"
        family = "test"

        def __init__(self) -> None:
            self.selections = 0
            self.observations = 0

        def select(self, row: dict[str, object]) -> str:  # noqa: ARG002
            assert self.selections == self.observations
            self.selections += 1
            return "model_a"

        def observe(self, row: dict[str, object], selected_model: str) -> None:  # noqa: ARG002
            self.observations += 1

    probe = LifecycleRule()
    evaluate_hypotheses(calibration, test_rows, rules=[probe])
    assert probe.selections == probe.observations == len(test_rows)


def test_saved_results_can_initialize_dashboard(tmp_path: Path) -> None:
    specs = fixed_specs()
    investigator = VotingInvestigator(specs=specs)
    calibration = investigator.diagnose([[event("a")]], split="calibration")
    test_rows = investigator.diagnose([[event("a"), event("b")]], split="test")
    hypotheses = evaluate_hypotheses(calibration, test_rows, rules=[GlobalAccuracyRule()])
    summary = build_summary(
        dataset_name="fixture",
        specs=specs,
        train_rows=0,
        calibration_rows=calibration,
        test_rows=test_rows,
        hypotheses=hypotheses,
    )
    summary["soft_failure_analysis"] = build_soft_failure_analysis(calibration, test_rows)
    assert [strategy["name"] for strategy in summary["strategies"]] == ["soft voting", "cheating voting"]
    assert summary["rule_scenarios"]
    assert "hard_failures" not in summary
    save_results(tmp_path, summary, test_rows)

    loaded_summary, loaded_rows = load_results(tmp_path)
    app = create_dashboard(tmp_path)

    assert loaded_summary["dataset"] == "fixture"
    assert len(loaded_rows) == 2
    assert app.server.test_client().get("/").status_code == 200
    layout = app.server.test_client().get("/_dash-layout").get_json()
    assert "Soft-vote analysis" in str(layout)
    assert "Rule-set scenarios" in str(layout)
    assert "Hard voting" not in str(layout)
    assert "overview-panel overview-main-panel" in str(layout)
