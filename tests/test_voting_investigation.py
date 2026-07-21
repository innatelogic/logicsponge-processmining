"""Tests for the standalone voting investigation engine and dashboard."""

from copy import deepcopy
from pathlib import Path

import pytest

from logicsponge.processmining.voting_dashboard import (
    _best_available_sequence_candidate,
    _headline_results,
    _sequence_figure,
    _sequence_options,
    create_dashboard,
    load_results,
)
from logicsponge.processmining.voting_investigation import (
    CalibratedGeneralizationRecoveryRule,
    CalibratedLoneDissenterRule,
    CalibratedLoneDissenterSecondRankRule,
    CalibratedComplexityContrastExceptionRule,
    CalibratedSoftRankRule,
    DecisionRule,
    EvidenceWeightedDistributionRule,
    GlobalAccuracyRule,
    ModelSpec,
    NGramCorrectStreakBoostRule,
    LargestNGramDisagreementMultiplierRule,
    PreviousErrorCorrectSetRule,
    StateEvidenceTopologyRule,
    StructuralBranchingEnsembleRule,
    TransientBagFavoritismRule,
    TransientGeneralizationBoostRule,
    VotingInvestigator,
    analyze_rule_impacts,
    archived_hypotheses,
    build_soft_failure_analysis,
    build_summary,
    dataset_result_dir,
    default_hypotheses,
    default_model_specs,
    discover_condition_hypotheses,
    evaluate_archived_rule_integrations,
    evaluate_hypotheses,
    evaluate_rule_integrations,
    evaluate_weighted_conditions,
    save_results,
    select_best_deployable,
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
    assert rows[0]["soft_ranked_predictions"][0]["activity"] == "b"
    assert rows[0]["soft_ranked_predictions"][1]["activity"] == "a"
    assert rows[0]["previous_soft_correct"] is None
    assert rows[1]["previous_soft_correct"] is False
    assert "normalized_entropy" in rows[0]["models"][0]
    assert "soft_divergence" in rows[0]["models"][0]
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


def test_compact_interpretable_rules_are_active_and_other_rules_are_archived() -> None:
    investigator = VotingInvestigator(specs=fixed_specs())
    calibration = investigator.diagnose([[event("a"), event("a")]], split="calibration")
    test_rows = investigator.diagnose([[event("a"), event("b")]], split="test")

    names = {rule.name for rule in default_hypotheses()}
    assert names == {
        "evidence-weighted distribution mixture (support 12)",
        "calibrated soft rank 2 override (support 2)",
        "calibrated lone-dissenter override (support 2)",
        "calibrated lone-dissenter rank 2 override (support 2)",
        "complexity-contrast exception override",
        "transient Bag favoritism after generalist-correct error (3 steps)",
    }
    archived_names = {rule.name for rule in archived_hypotheses()}
    assert "best calibration accuracy" in archived_names
    assert "per-state accuracy (support 5)" in archived_names
    assert "agreement ≥ 3, else accuracy" in archived_names
    assert "hierarchical reliability (support 3)" in archived_names
    assert "stacked rule portfolio (support 5)" in archived_names
    assert "delayed-feedback adaptive (decay 0.94)" in archived_names
    assert "calibrated probability pool (support 5)" in archived_names
    assert "state-evidence topology (support 5)" in archived_names
    assert "structural branching ensemble (support 8)" in archived_names
    assert "distribution-shape reliability (support 8)" in archived_names
    assert "previous-error correct-set persistence (support 3)" in archived_names
    assert "transient generalization boost after generalist-correct error (3 steps)" in archived_names
    assert "transient generalization boost after any soft error (3 steps)" in archived_names
    assert "transient recovery calibrated by age gate" in archived_names
    assert names.isdisjoint(archived_names)
    assert not any(name.startswith("suffix length") for name in names)
    assert not any("residual correction" in name for name in names)
    assert not any(name.startswith("previous-outcome") for name in names)
    assert not any(name.startswith("hashed rank router") for name in names)

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


def test_previous_error_rule_filters_to_models_correct_on_previous_event() -> None:
    investigator = VotingInvestigator(specs=fixed_specs())
    calibration = investigator.diagnose([[event("a"), event("a")]], split="calibration")
    test_rows = investigator.diagnose([[event("a"), event("b")]], split="test")
    rule = PreviousErrorCorrectSetRule(minimum_support=1)

    evaluate_hypotheses(calibration, test_rows, rules=[rule])

    assert test_rows[0]["soft_correct"] is False
    assert test_rows[0]["correct_models"] == ["model_a"]
    assert test_rows[1]["previous_correct_models"] == ["model_a"]
    assert test_rows[1]["rule_models"][rule.name] == "model_a"


def test_transient_generalization_boost_enforces_lowest_complexity_for_full_horizon() -> None:
    investigator = VotingInvestigator(specs=fixed_specs())
    rows = investigator.diagnose(
        [[event("a"), event("b"), event("b"), event("b"), event("b")]],
        split="test",
    )
    rule = TransientGeneralizationBoostRule(horizon=3, trigger_mode="all soft errors")

    sources_and_predictions = [rule.choose(row) for row in rows]

    assert rows[0]["soft_correct"] is False
    assert sources_and_predictions[1] == ("model_a", "a")
    assert sources_and_predictions[2] == ("model_a", "a")
    assert sources_and_predictions[3] == ("model_a", "a")
    assert sources_and_predictions[4] == ("soft voting", "b")
    assert [rows[index]["rule_diagnostics"][rule.name]["recovery_age"] for index in (1, 2, 3)] == [0, 1, 2]
    assert all(rows[index]["rule_diagnostics"][rule.name]["prediction_matches_target"] for index in (1, 2, 3))


def test_default_generalist_is_uniquely_bag() -> None:
    specs = default_model_specs((2, 3, 4, 5, 6))
    minimum_complexity = min(spec.complexity for spec in specs)
    generalists = [spec for spec in specs if spec.complexity == minimum_complexity]

    assert minimum_complexity == 0
    assert [(spec.name, spec.complexity) for spec in generalists] == [("bag", 0)]
    assert [(spec.model_type, spec.window_size) for spec in specs[2:]] == [
        ("ngram", 2),
        ("ngram", 3),
        ("ngram", 4),
        ("ngram", 5),
        ("ngram", 6),
    ]
    with pytest.raises(ValueError, match="at least 2"):
        default_model_specs((0, 2))
    with pytest.raises(ValueError, match="at least 2"):
        default_model_specs((1, 2))


def test_transient_rule_selects_bag_by_complexity_not_model_order() -> None:
    specs = [
        ModelSpec("ngram_2", 2, lambda: FixedPredictionMiner("b")),
        ModelSpec("bag", 0, lambda: FixedPredictionMiner("a")),
        ModelSpec("fpt", 1, lambda: FixedPredictionMiner("b")),
    ]
    rows = VotingInvestigator(specs=specs).diagnose([[event("c"), event("a")]], split="test")
    rule = TransientGeneralizationBoostRule(trigger_mode="all soft errors")

    predictions = [rule.choose(row) for row in rows]

    assert rows[0]["soft_correct"] is False
    assert predictions[1] == ("bag", "a")
    assert rows[1]["rule_diagnostics"][rule.name]["target_model"] == "bag"
    assert rows[1]["rule_diagnostics"][rule.name]["target_complexity"] == 0


def test_any_error_boost_activates_when_generalist_was_also_wrong() -> None:
    investigator = VotingInvestigator(specs=fixed_specs())
    rows = investigator.diagnose([[event("c"), event("b")]], split="test")
    conditional = TransientGeneralizationBoostRule(trigger_mode="generalist was correct")
    unconditional = TransientGeneralizationBoostRule(trigger_mode="all soft errors")

    conditional_predictions = [conditional.choose(row) for row in rows]
    unconditional_predictions = [unconditional.choose(row) for row in rows]

    assert rows[0]["soft_correct"] is False
    assert "model_a" not in rows[0]["correct_models"]
    assert conditional_predictions[1] == ("soft voting", "b")
    assert unconditional_predictions[1] == ("model_a", "a")


def test_transient_generalization_boost_persists_fitted_parameters() -> None:
    investigator = VotingInvestigator(specs=fixed_specs())
    calibration = investigator.diagnose([[event("a"), event("b"), event("b")]], split="calibration")
    test_rows = investigator.diagnose([[event("a"), event("b")]], split="test")
    rule = TransientGeneralizationBoostRule()

    summary = evaluate_hypotheses(calibration, test_rows, rules=[rule])[0]

    assert summary["parameters"]["target"] == "minimum model complexity"
    assert summary["parameters"]["horizon"] == 3
    assert summary["parameters"]["calibrated"] is False
    assert summary["parameters"]["enforcement"] == "direct constituent prediction"
    assert summary["parameters"]["target_models"] == ["model_a"]
    assert "initial_multiplier" not in summary["parameters"]
    assert "log_decay_rate" not in summary["parameters"]


def test_transient_bag_favoritism_is_positive_decaying_and_conditionally_triggered() -> None:
    investigator = VotingInvestigator(specs=fixed_specs())
    rows = investigator.diagnose(
        [[event("a"), event("b"), event("b"), event("b"), event("b")]],
        split="test",
    )
    rule = TransientBagFavoritismRule()
    summary = evaluate_hypotheses([], rows, rules=[rule])[0]

    diagnostics = [row["rule_diagnostics"][rule.name] for row in rows]
    assert rows[0]["soft_correct"] is False
    assert rows[0]["correct_models"] == ["model_a"]
    assert [diagnostics[index]["multiplier"] for index in (1, 2, 3)] == [3.0, 2.0, 1.5]
    assert diagnostics[4]["multiplier"] == 1.0
    assert all(
        diagnostics[index]["model_multipliers"] == {"model_a": value}
        for index, value in zip((1, 2, 3), (3.0, 2.0, 1.5), strict=True)
    )
    assert diagnostics[4]["model_multipliers"] == {}
    assert summary["parameters"]["calibrated"] is False
    assert summary["parameters"]["composition"] == "positive per-model multiplier"

    generalist_wrong_rows = investigator.diagnose([[event("c"), event("b")]], split="test")
    wrong_rule = TransientBagFavoritismRule(trigger_mode="generalist was wrong")
    evaluate_hypotheses([], generalist_wrong_rows, rules=[TransientBagFavoritismRule(), wrong_rule])
    assert generalist_wrong_rows[1]["rule_diagnostics"][rule.name]["active"] is False
    assert generalist_wrong_rows[1]["rule_diagnostics"][wrong_rule.name]["active"] is True
    assert generalist_wrong_rows[1]["rule_diagnostics"][wrong_rule.name]["model_multipliers"] == {"model_a": 2.5}


def test_three_calibrated_recovery_techniques_gate_or_blend_bag() -> None:
    investigator = VotingInvestigator(specs=fixed_specs())
    favorable = investigator.diagnose(
        [[event("c"), event("a")] for _ in range(6)],
        split="calibration",
    )
    unfavorable = investigator.diagnose(
        [[event("c"), event("b")] for _ in range(6)],
        split="calibration",
    )
    test_rows = investigator.diagnose([[event("c"), event("a")]], split="test")

    favorable_rules = [
        CalibratedGeneralizationRecoveryRule(mode, minimum_support=1)
        for mode in ("age gate", "structural gate", "distribution blend")
    ]
    summaries = evaluate_hypotheses(favorable, test_rows, rules=favorable_rules)

    assert len(summaries) == 3
    assert all(summary["parameters"]["calibrated"] for summary in summaries)
    assert all(summary["parameters"]["calibration_available"] for summary in summaries)
    assert test_rows[1]["rule_predictions"]["transient recovery calibrated by age gate"] == "a"
    assert test_rows[1]["rule_predictions"]["transient recovery calibrated by structural gate"] == "a"
    assert test_rows[1]["rule_diagnostics"]["transient recovery calibrated by distribution blend"]["multiplier"] > 1.0

    rejected_rows = investigator.diagnose([[event("c"), event("b")]], split="test")
    evaluate_hypotheses(
        unfavorable,
        rejected_rows,
        rules=[CalibratedGeneralizationRecoveryRule("age gate", minimum_support=1)],
    )
    age_name = "transient recovery calibrated by age gate"
    assert rejected_rows[1]["rule_diagnostics"][age_name]["recovery_age"] == 0
    assert rejected_rows[1]["rule_diagnostics"][age_name]["active"] is False
    assert rejected_rows[1]["rule_predictions"][age_name] == rejected_rows[1]["soft_prediction"]


def test_ngram_streak_boost_uses_exponential_formula_and_delayed_correctness() -> None:
    specs = [
        ModelSpec("bag", 0, lambda: FixedPredictionMiner("b"), model_type="bag"),
        ModelSpec("fpt", 1, lambda: FixedPredictionMiner("b"), model_type="fpt"),
        ModelSpec("other", 1, lambda: FixedPredictionMiner("b")),
        ModelSpec("short", 2, lambda: FixedPredictionMiner("a"), model_type="ngram", window_size=2),
        ModelSpec("long", 5, lambda: FixedPredictionMiner("a"), model_type="ngram", window_size=5),
    ]
    rows = VotingInvestigator(specs=specs).diagnose(
        [[*[event("a") for _ in range(7)], event("b"), event("b")]],
        split="test",
    )
    rule = NGramCorrectStreakBoostRule()
    summary = evaluate_hypotheses([], rows, rules=[rule])[0]

    assert rule.boost_ratio(0, 2) == 0.0
    assert rule.boost_ratio(1, 2) == 1.0
    assert rule.boost_ratio(1, 5) == pytest.approx(0.0320586033)
    assert rule.boost_ratio(2, 5) == pytest.approx(0.1192029220)
    assert rule.boost_ratio(3, 5) == pytest.approx(0.3560857401)
    assert rule.boost_ratio(4, 5) == 1.0
    assert rule.boost_ratio(5, 5) == 1.0
    assert rule.boost_ratio(6, 5) == 0.0
    assert rule.boost_ratio(9, 5) == 0.0
    assert rule.streak_multiplier(0, 5) == 0.1
    assert rule.streak_multiplier(2, 5) == 1.0
    assert rule.streak_multiplier(4, 5) == 2.0
    assert rule.streak_multiplier(5, 5) == 2.0
    assert rule.streak_multiplier(6, 5) == 1.0
    assert rows[1]["rule_diagnostics"][rule.name]["eligible"]["short"] is True
    assert rows[1]["rule_diagnostics"][rule.name]["boost_ratios"]["short"] == 1.0
    assert rows[3]["rule_diagnostics"][rule.name]["boost_ratios"] == {
        "short": 0.0,
        "long": pytest.approx(0.3560857401),
    }
    assert rows[3]["rule_diagnostics"][rule.name]["multipliers"] == {
        "short": 1.0,
        "long": pytest.approx(2.2360679775),
    }
    assert rows[3]["rule_diagnostics"][rule.name]["model_multipliers"] == {
        "long": pytest.approx(2.2360679775),
    }
    assert rows[4]["rule_predictions"][rule.name] == "a"
    assert rows[7]["rule_predictions"][rule.name] == "b"
    assert rows[8]["rule_diagnostics"][rule.name]["active"] is True
    assert rows[8]["rule_diagnostics"][rule.name]["model_multipliers"] == {"short": 0.0, "long": 0.0}
    assert summary["parameters"]["calibrated"] is False


def test_ngram_streak_boost_requires_a_previous_soft_error() -> None:
    specs = [
        ModelSpec("bag", 0, lambda: FixedPredictionMiner("a"), model_type="bag"),
        ModelSpec("fpt", 1, lambda: FixedPredictionMiner("a"), model_type="fpt"),
        ModelSpec("ngram_2", 2, lambda: FixedPredictionMiner("a"), model_type="ngram", window_size=2),
    ]
    rows = VotingInvestigator(specs=specs).diagnose([[event("a"), event("a")]], split="test")
    rule = NGramCorrectStreakBoostRule()

    evaluate_hypotheses([], rows, rules=[rule])

    diagnostic = rows[1]["rule_diagnostics"][rule.name]
    assert diagnostic["boost_ratios"] == {"ngram_2": 1.0}
    assert diagnostic["follows_soft_error"] is False
    assert diagnostic["multipliers"] == {"ngram_2": 1.0}
    assert diagnostic["model_multipliers"] == {}
    assert diagnostic["active"] is False


def test_largest_ngram_disagreement_multiplier_only_breaks_streak_ties() -> None:
    specs = [
        ModelSpec("bag", 0, lambda: FixedPredictionMiner("b"), model_type="bag"),
        ModelSpec("short", 2, lambda: FixedPredictionMiner("a"), model_type="ngram", window_size=2),
        ModelSpec("long", 5, lambda: FixedPredictionMiner("b"), model_type="ngram", window_size=5),
    ]
    row = VotingInvestigator(specs=specs).diagnose([[event("a")]], split="test")[0]
    rule = LargestNGramDisagreementMultiplierRule()
    rule.fit([])
    row["rule_diagnostics"] = {NGramCorrectStreakBoostRule.name: {"multipliers": {"short": 1.0, "long": 1.0}}}

    source, prediction = rule.choose(row)

    diagnostic = row["rule_diagnostics"][rule.name]
    assert source == "largest ngram weighted pool"
    assert prediction == "b"
    assert diagnostic["active"] is True
    assert diagnostic["model_multipliers"] == {"long": 1.1}

    row["rule_diagnostics"] = {NGramCorrectStreakBoostRule.name: {"multipliers": {"short": 0.1, "long": 2.0}}}
    source, prediction = rule.choose(row)

    assert source == "soft voting"
    assert prediction == row["soft_prediction"]
    assert row["rule_diagnostics"][rule.name]["active"] is False


def test_bag_and_ngram_boosts_stack_without_priority_or_negative_weights() -> None:
    rows = VotingInvestigator(specs=fixed_specs()).diagnose([[event("a")]], split="test")
    bag_name = TransientBagFavoritismRule().name
    wrong_bag_name = TransientBagFavoritismRule(trigger_mode="generalist was wrong").name
    ngram_name = NGramCorrectStreakBoostRule.name
    row = rows[0]
    row["rule_predictions"] = {bag_name: "a", wrong_bag_name: "a", ngram_name: "b"}
    row["rule_diagnostics"] = {
        bag_name: {"active": True, "model_multipliers": {"model_a": 10.0}},
        wrong_bag_name: {"active": True, "model_multipliers": {"model_a": 5.5}},
        ngram_name: {"active": True, "model_multipliers": {"model_b1": 0.25}},
    }

    integrations = evaluate_rule_integrations([], rows, [bag_name, wrong_bag_name, ngram_name])

    assert [item["name"] for item in integrations] == ["independent Bag + N-gram boost stack"]
    stack = integrations[0]
    assert stack["deployment_priority"] is True
    assert row["integration_predictions"][stack["name"]] == "a"
    diagnostics = row["integration_diagnostics"][stack["name"]]
    assert diagnostics["model_multipliers"] == {"model_a": 10.0, "model_b1": 0.25, "model_b2": 1.0}
    assert diagnostics["simultaneous"] is True
    assert diagnostics["target_overlap"] == []
    assert diagnostics["all_multipliers_non_negative"] is True
    audit = stack["parameters"]["enforcement_audit"]
    assert audit["simultaneous_boost_events"] == 1
    assert audit["cross_rule_target_overlap_events"] == 0
    assert audit["bag_rule_overlap_events"] == 1
    assert audit["negative_multiplier_events"] == 1


def test_ngram_streak_rule_migrates_legacy_rows_without_model_metadata() -> None:
    specs = [
        ModelSpec("bag", 0, lambda: FixedPredictionMiner("b"), model_type="bag"),
        ModelSpec("ngram_2", 2, lambda: FixedPredictionMiner("a"), model_type="ngram", window_size=2),
        ModelSpec("ngram_5", 5, lambda: FixedPredictionMiner("a"), model_type="ngram", window_size=5),
    ]
    rows = VotingInvestigator(specs=specs).diagnose([[event("a") for _ in range(8)]], split="test")
    for row in rows:
        for model in row["models"]:
            model.pop("model_type")
            model.pop("window_size")

    evaluate_hypotheses([], rows, rules=[NGramCorrectStreakBoostRule()])

    assert rows[3]["rule_diagnostics"][NGramCorrectStreakBoostRule.name]["boost_ratios"] == {
        "ngram_2": 0.0,
        "ngram_5": pytest.approx(0.3560857401),
    }


def test_structural_rules_fit_without_activity_specific_conditions() -> None:
    investigator = VotingInvestigator(specs=fixed_specs())
    calibration = investigator.diagnose(
        [[event("a"), event("a")], [event("a"), event("b")]],
        split="calibration",
    )
    test_rows = investigator.diagnose([[event("a"), event("b")]], split="test")
    rules = [
        StateEvidenceTopologyRule(minimum_support=1),
        EvidenceWeightedDistributionRule(minimum_support=1),
        StructuralBranchingEnsembleRule(minimum_support=1, confidence_z=0.0),
    ]

    summaries = evaluate_hypotheses(calibration, test_rows, rules=rules)

    assert len(summaries) == 3
    assert all(summary["description"] for summary in summaries)
    assert all(summary["interpretation"] for summary in summaries)
    assert all(summary["total"] == len(test_rows) for summary in summaries)
    assert all(rule.name in test_rows[0]["rule_predictions"] for rule in rules)


def test_structural_rule_decisions_are_invariant_to_activity_renaming() -> None:
    investigator = VotingInvestigator(specs=fixed_specs())
    calibration = investigator.diagnose(
        [[event("a"), event("a")], [event("a"), event("b")]],
        split="calibration",
    )
    test_rows = investigator.diagnose([[event("a"), event("b")]], split="test")
    rename = {"a": "x", "b": "y", "": ""}

    def renamed(rows: list[dict[str, object]]) -> list[dict[str, object]]:
        copied = deepcopy(rows)
        for row in copied:
            for field in ("actual", "soft_prediction", "consensus_prediction", "oracle_prediction"):
                row[field] = rename.get(str(row[field]), str(row[field]))
            row["soft_distribution"] = {
                rename.get(activity, activity): probability
                for activity, probability in row["soft_distribution"].items()  # type: ignore[union-attr]
            }
            for item in row["soft_ranked_predictions"]:  # type: ignore[union-attr]
                item["activity"] = rename.get(item["activity"], item["activity"])
            for model in row["models"]:  # type: ignore[union-attr]
                model["prediction"] = rename.get(model["prediction"], model["prediction"])
                model["distribution"] = {
                    rename.get(activity, activity): probability
                    for activity, probability in model["distribution"].items()
                }
                for item in model["ranked_predictions"]:
                    item["activity"] = rename.get(item["activity"], item["activity"])
        return copied

    renamed_calibration = renamed(calibration)  # type: ignore[arg-type]
    renamed_test = renamed(test_rows)  # type: ignore[arg-type]
    factories = (
        lambda: StateEvidenceTopologyRule(minimum_support=1),
        lambda: EvidenceWeightedDistributionRule(minimum_support=1),
        lambda: StructuralBranchingEnsembleRule(minimum_support=1, confidence_z=0.0),
    )
    for factory in factories:
        original_rule = factory()
        renamed_rule = factory()
        original_rule.fit(calibration)
        renamed_rule.fit(renamed_calibration)  # type: ignore[arg-type]
        original_source, original_prediction = original_rule.choose(test_rows[0])
        renamed_source, renamed_prediction = renamed_rule.choose(renamed_test[0])  # type: ignore[arg-type]
        assert renamed_source == original_source
        assert renamed_prediction == rename[original_prediction]


def test_default_rules_are_invariant_to_constituent_model_renaming() -> None:
    investigator = VotingInvestigator(specs=fixed_specs())
    calibration = investigator.diagnose(
        [[event("a"), event("a")], [event("a"), event("b")]],
        split="calibration",
    )
    test_rows = investigator.diagnose([[event("a"), event("b")]], split="test")
    model_rename = {
        "model_a": "renamed_alpha",
        "model_b1": "renamed_beta",
        "model_b2": "renamed_gamma",
    }

    def renamed_models(rows: list[dict[str, object]]) -> list[dict[str, object]]:
        copied = deepcopy(rows)
        for row in copied:
            for model in row["models"]:  # type: ignore[union-attr]
                model["name"] = model_rename[model["name"]]
            for field in ("correct_models", "previous_correct_models"):
                row[field] = [model_rename[name] for name in row[field]]  # type: ignore[index]
            if row.get("oracle_model"):
                row["oracle_model"] = model_rename[str(row["oracle_model"])]
        return copied

    renamed_calibration = renamed_models(calibration)  # type: ignore[arg-type]
    renamed_test = renamed_models(test_rows)  # type: ignore[arg-type]
    original_rules = default_hypotheses()
    renamed_rules = default_hypotheses()

    for original_rule, renamed_rule in zip(original_rules, renamed_rules, strict=True):
        original_rule.fit(calibration)
        renamed_rule.fit(renamed_calibration)  # type: ignore[arg-type]
        assert original_rule.selection_policy
        for original_row, renamed_row in zip(test_rows, renamed_test, strict=True):
            original_source, original_prediction = original_rule.choose(original_row)
            renamed_source, renamed_prediction = renamed_rule.choose(renamed_row)  # type: ignore[arg-type]
            assert renamed_source == model_rename.get(original_source, original_source)
            assert renamed_prediction == original_prediction


def test_learned_conditions_are_invariant_to_constituent_model_renaming() -> None:
    investigator = VotingInvestigator(specs=fixed_specs())
    calibration = investigator.diagnose(
        [[event("a"), event("a")], [event("a"), event("b")]],
        split="calibration",
    )
    test_rows = investigator.diagnose([[event("a"), event("b")]], split="test")
    model_rename = {
        "model_a": "renamed_alpha",
        "model_b1": "renamed_beta",
        "model_b2": "renamed_gamma",
    }

    def renamed_models(rows: list[dict[str, object]]) -> list[dict[str, object]]:
        copied = deepcopy(rows)
        for row in copied:
            for model in row["models"]:  # type: ignore[union-attr]
                model["name"] = model_rename[model["name"]]
            for field in ("correct_models", "previous_correct_models"):
                row[field] = [model_rename[name] for name in row[field]]  # type: ignore[index]
            if row.get("oracle_model"):
                row["oracle_model"] = model_rename[str(row["oracle_model"])]
        return copied

    def canonical_label(value: str) -> str:
        for original, renamed in model_rename.items():
            value = value.replace(renamed, original)
        return value

    original_conditions, _ = discover_condition_hypotheses(calibration, test_rows, minimum_support=1)
    renamed_conditions, _ = discover_condition_hypotheses(
        renamed_models(calibration),  # type: ignore[arg-type]
        renamed_models(test_rows),  # type: ignore[arg-type]
        minimum_support=1,
    )
    renamed_by_condition = {
        (canonical_label(condition["feature"]), canonical_label(condition["value"])): condition
        for condition in renamed_conditions
    }

    assert original_conditions
    for original in original_conditions:
        renamed = renamed_by_condition[(original["feature"], original["value"])]
        assert renamed["recommended_model"] == model_rename[original["recommended_model"]]
        assert renamed["selection_policy"] == original["selection_policy"]


def test_calibrated_second_choice_can_override_soft_voting() -> None:
    investigator = VotingInvestigator(specs=fixed_specs())
    calibration = investigator.diagnose([[event("a"), event("a")]], split="calibration")
    test_rows = investigator.diagnose([[event("a")]], split="test")

    summaries = evaluate_hypotheses(
        calibration,
        test_rows,
        rules=[CalibratedSoftRankRule(rank=2, minimum_support=1)],
    )

    rule_name = "calibrated soft rank 2 override (support 1)"
    assert test_rows[0]["soft_prediction"] == "b"
    assert test_rows[0]["rule_predictions"][rule_name] == "a"
    assert test_rows[0]["rule_models"][rule_name] == "soft rank 2"
    assert summaries[0]["accuracy"] == 1.0


def test_calibrated_lone_dissenter_overrides_only_after_supported_advantage() -> None:
    investigator = VotingInvestigator(specs=fixed_specs())
    calibration = investigator.diagnose([[event("a")] for _ in range(3)], split="calibration")
    test_rows = investigator.diagnose([[event("a")]], split="test")
    rule = CalibratedLoneDissenterRule(minimum_support=3, confidence_z=0.0)

    summaries = evaluate_hypotheses(calibration, test_rows, rules=[rule])

    assert test_rows[0]["soft_prediction"] == "b"
    assert test_rows[0]["rule_models"][rule.name] == "model_a"
    assert test_rows[0]["rule_predictions"][rule.name] == "a"
    diagnostic = test_rows[0]["rule_diagnostics"][rule.name]
    assert diagnostic["active"] is True
    assert diagnostic["candidate_model"] == "model_a"
    assert diagnostic["decision"]["support"] == 3
    assert diagnostic["decision"]["lower_bound"] > 0
    assert summaries[0]["accuracy"] == 1.0


def test_calibrated_lone_dissenter_keeps_soft_vote_without_positive_history() -> None:
    investigator = VotingInvestigator(specs=fixed_specs())
    calibration = investigator.diagnose([[event("b")] for _ in range(3)], split="calibration")
    test_rows = investigator.diagnose([[event("a")]], split="test")
    rule = CalibratedLoneDissenterRule(minimum_support=3, confidence_z=0.0)

    evaluate_hypotheses(calibration, test_rows, rules=[rule])

    assert test_rows[0]["rule_models"][rule.name] == "soft voting"
    assert test_rows[0]["rule_predictions"][rule.name] == test_rows[0]["soft_prediction"]
    assert test_rows[0]["rule_diagnostics"][rule.name]["active"] is False


def test_lone_dissenter_second_choice_uses_calibrated_low_representation_signal() -> None:
    row = {
        "actual": "c",
        "soft_prediction": "b",
        "soft_correct": False,
        "consensus_prediction": "b",
        "agreement_count": 2,
        "soft_normalized_entropy": 0.3,
        "soft_margin": 0.2,
        "model_confidence_spread": 0.1,
        "previous_soft_correct": True,
        "previous_correct_models": ["model_b1", "model_b2"],
        "relative_position": 0.5,
        "models": [
            {
                "index": 0,
                "name": "model_a",
                "prediction": "a",
                "ranked_predictions": [
                    {"activity": "a", "probability": 0.6},
                    {"activity": "c", "probability": 0.4},
                ],
                "state": "rare-state",
                "state_visits": 1,
                "confidence": 0.6,
                "margin": 0.2,
            },
            {
                "index": 1,
                "name": "model_b1",
                "prediction": "b",
                "ranked_predictions": [{"activity": "b", "probability": 1.0}],
                "state": "common-state",
                "state_visits": 10,
                "confidence": 1.0,
                "margin": 1.0,
            },
            {
                "index": 2,
                "name": "model_b2",
                "prediction": "b",
                "ranked_predictions": [{"activity": "b", "probability": 1.0}],
                "state": "common-state",
                "state_visits": 10,
                "confidence": 1.0,
                "margin": 1.0,
            },
        ],
    }
    rule = CalibratedLoneDissenterSecondRankRule(minimum_support=4, confidence_z=0.0)
    rule.fit([deepcopy(row) for _ in range(4)])

    source, prediction = rule.choose(row)

    assert source == "model_a rank 2"
    assert prediction == "c"
    diagnostic = row["rule_diagnostics"][rule.name]
    assert diagnostic["active"] is True
    assert diagnostic["risk_signals"] == ["low_representation"]
    assert diagnostic["decision"] == {"systematic": True}


def test_complexity_contrast_selects_calibrated_high_model_exception() -> None:
    low_model = {
        "index": 0,
        "name": "generalist",
        "complexity": 0,
        "prediction": "a",
        "distribution": {"a": 0.45, "b": 0.4, "c": 0.15},
        "ranked_predictions": [
            {"activity": "a", "probability": 0.45},
            {"activity": "b", "probability": 0.4},
            {"activity": "c", "probability": 0.15},
        ],
    }
    high_models = [
        {
            "index": index,
            "name": f"specialist_{index}",
            "complexity": index + 1,
            "prediction": "a",
            "distribution": {"a": 0.48, "b": 0.45, "c": 0.07},
            "ranked_predictions": [
                {"activity": "a", "probability": 0.48},
                {"activity": "b", "probability": 0.45},
                {"activity": "c", "probability": 0.07},
            ],
        }
        for index in range(1, 5)
    ]
    row = {
        "actual": "b",
        "soft_prediction": "a",
        "soft_correct": False,
        "agreement_count": 5,
        "soft_margin": 0.2,
        "soft_normalized_entropy": 0.4,
        "relative_position": 0.5,
        "models": [low_model, *high_models],
    }
    rule = CalibratedComplexityContrastExceptionRule()
    rule.fit([])

    source, prediction = rule.choose(row)

    assert source == "complexity-contrast alternative"
    assert prediction == "b"
    diagnostic = row["rule_diagnostics"][rule.name]
    assert diagnostic["active"] is True
    assert diagnostic["candidate"]["negative_complexity_weights"]["generalist"] > diagnostic["candidate"]["positive_complexity_weights"]["generalist"]
    assert diagnostic["candidate"]["positive_complexity_weights"]["specialist_4"] > diagnostic["candidate"]["negative_complexity_weights"]["specialist_4"]
    assert diagnostic["candidate"]["top_three_support"] == 4
    assert diagnostic["decision"]["support"] == 4


def test_only_independent_boost_stack_is_active_and_complex_integrations_are_archived() -> None:
    investigator = VotingInvestigator(specs=fixed_specs())
    selector_rows = investigator.diagnose([[event("a"), event("a")]], split="selector-calibration")
    test_rows = investigator.diagnose([[event("a"), event("b")]], split="test")
    rules = default_hypotheses()
    selector_hypotheses = evaluate_hypotheses(selector_rows, selector_rows, rules=rules)
    evaluate_hypotheses(selector_rows, test_rows, rules=default_hypotheses())

    integrations = evaluate_rule_integrations(
        selector_rows,
        test_rows,
        [hypothesis["name"] for hypothesis in selector_hypotheses],
    )

    names = {result["name"] for result in integrations}
    assert names == {"independent Bag + N-gram boost stack"}
    assert all(result["deployable"] for result in integrations)
    assert all("description" in result and "parameters" in result for result in integrations)
    assert set(test_rows[0]["integration_predictions"]) == names

    archived = evaluate_archived_rule_integrations(
        selector_rows,
        test_rows,
        [hypothesis["name"] for hypothesis in selector_hypotheses],
    )
    archived_names = {result["name"] for result in archived}
    assert "calibration-best rule" in archived_names
    assert "contextual Bayesian family vote" in archived_names
    assert "structural disagreement family router" in archived_names
    assert "sequence-local delayed-feedback Hedge" in archived_names
    assert "uncertainty-gated family consensus" in archived_names
    assert "delayed-feedback Hedge portfolio" in archived_names
    assert names.isdisjoint(archived_names)


def test_independent_stack_is_the_deployment_contract() -> None:
    investigator = VotingInvestigator(specs=fixed_specs())
    selector_rows = investigator.diagnose([[event("a"), event("b"), event("b")]], split="selector-calibration")
    test_rows = investigator.diagnose([[event("c"), event("b"), event("b")]], split="test")

    def rules() -> list[DecisionRule]:
        return default_hypotheses()

    selector_hypotheses = evaluate_hypotheses(selector_rows, selector_rows, rules=rules())
    hypotheses = evaluate_hypotheses(selector_rows, test_rows, rules=rules())
    integrations = evaluate_rule_integrations(
        selector_rows,
        test_rows,
        [hypothesis["name"] for hypothesis in selector_hypotheses],
    )
    mandatory = integrations[0]

    assert mandatory["deployment_priority"] is True
    assert mandatory["name"] == "independent Bag + N-gram boost stack"
    assert mandatory["parameters"]["calibrated"] is False
    assert mandatory["parameters"]["priority_order"] == []
    audit = mandatory["parameters"]["enforcement_audit"]
    assert audit["negative_multiplier_events"] == 0
    assert audit["cross_rule_target_overlap_events"] == 0
    best = select_best_deployable([*hypotheses, *integrations])
    assert best["name"] == mandatory["name"]

    summary = build_summary(
        dataset_name="fixture",
        specs=fixed_specs(),
        train_rows=0,
        calibration_rows=selector_rows,
        test_rows=test_rows,
        hypotheses=hypotheses,
        rule_integrations=integrations,
    )
    contract = summary["independent_boost_contract"]
    assert contract["minimum_complexity_models"] == ["model_a"]
    assert contract["unique_minimum_complexity_model"] is True
    assert contract["default_generalist_is_bag"] is False
    assert contract["independent_stack_present"] is True
    assert contract["independent_stack_is_best_deployable"] is True
    assert contract["negative_multiplier_events"] == 0
    assert all(row["best_deployable_method"] == mandatory["name"] for row in test_rows)
    assert all(
        row["best_deployable_prediction"] == row["integration_predictions"][mandatory["name"]] for row in test_rows
    )


def test_independent_stack_does_not_require_a_selector_calibration_partition() -> None:
    investigator = VotingInvestigator(specs=fixed_specs())
    test_rows = investigator.diagnose([[event("c"), event("b")]], split="test")
    hypotheses = evaluate_hypotheses([], test_rows, rules=default_hypotheses())

    integrations = evaluate_rule_integrations(
        [],
        test_rows,
        [hypothesis["name"] for hypothesis in hypotheses],
    )

    assert [result["name"] for result in integrations] == ["independent Bag + N-gram boost stack"]
    assert integrations[0]["deployment_priority"] is True
    assert integrations[0]["parameters"]["calibrated"] is False
    assert integrations[0]["parameters"]["priority_order"] == []
    assert integrations[0]["parameters"]["enforcement_audit"]["negative_multiplier_events"] == 0


def test_saved_results_can_initialize_dashboard(tmp_path: Path) -> None:  # noqa: PLR0915
    specs = fixed_specs()
    investigator = VotingInvestigator(specs=specs)
    calibration = investigator.diagnose([[event("a")]], split="calibration")
    test_rows = investigator.diagnose([[event("a"), event("b")]], split="test")
    hypotheses = evaluate_hypotheses(
        calibration,
        test_rows,
        rules=[TransientGeneralizationBoostRule(trigger_mode="all soft errors")],
    )
    summary = build_summary(
        dataset_name="fixture",
        specs=specs,
        train_rows=0,
        calibration_rows=calibration,
        test_rows=test_rows,
        hypotheses=hypotheses,
    )
    summary["soft_failure_analysis"] = build_soft_failure_analysis(calibration, test_rows)
    archived_name = "highest confidence"
    summary["hypotheses"].append(
        {
            "name": archived_name,
            "family": "confidence",
            "accuracy": 0.0,
            "correct": 0,
            "total": len(test_rows),
            "selected_models": {},
        }
    )
    summary["rule_integrations"].append({"name": "positive-gain family vote", "accuracy": 0.0, "deployable": True})
    summary["soft_failure_analysis"]["rule_impacts"].append(
        {"name": archived_name, "family": "confidence", "resulting_accuracy": 0.0}
    )
    for row in test_rows:
        row["rule_predictions"][archived_name] = row["soft_prediction"]
        row["rule_models"][archived_name] = "model_a"
    # Simulate a result file produced by the former direct-routing rule. The
    # dashboard must migrate these stale event predictions without retraining.
    transient_name = hypotheses[0]["name"]
    test_rows[1]["rule_predictions"][transient_name] = test_rows[1]["soft_prediction"]
    test_rows[1]["rule_models"][transient_name] = "transient low-complexity weighted pool"
    test_rows[1]["rule_diagnostics"] = {}
    assert [strategy["name"] for strategy in summary["strategies"]] == [
        "soft voting",
        "adaptive voting",
        "cheating voting",
    ]
    assert summary["rule_scenarios"]
    assert "hard_failures" not in summary
    save_results(tmp_path, summary, test_rows)

    loaded_summary, loaded_rows = load_results(tmp_path)
    app = create_dashboard(tmp_path)

    assert loaded_summary["dataset"] == "fixture"
    assert len(loaded_rows) == 2
    assert archived_name not in {rule["name"] for rule in loaded_summary["hypotheses"]}
    assert {item["name"] for item in loaded_summary["rule_integrations"]} == {"independent Bag + N-gram boost stack"}
    assert archived_name not in loaded_rows[0]["rule_predictions"]
    assert transient_name not in loaded_rows[1]["rule_predictions"]
    active_bag_name = TransientBagFavoritismRule().name
    assert loaded_rows[1]["rule_diagnostics"][active_bag_name]["model_multipliers"] == {"model_a": 7.0}
    assert loaded_rows[1]["best_deployable_prediction"] == "a"
    assert loaded_summary["independent_boost_contract"]["independent_stack_present"] is True
    assert loaded_summary["independent_boost_contract"]["negative_multiplier_events"] == 0
    headline_labels = {row["scenario"] for row in _headline_results(loaded_summary)}
    assert {
        "Transient Bag favoritism",
    } <= headline_labels
    best_sequence_candidate = _best_available_sequence_candidate(loaded_summary, loaded_rows)
    sequence_figure = _sequence_figure(loaded_rows, [spec.name for spec in specs], best_sequence_candidate)
    assert best_sequence_candidate is not None
    assert "Best deployable" in {trace.name for trace in sequence_figure.data}
    trace_names = [trace.name for trace in sequence_figure.data]
    assert trace_names.index("Adaptive voting") == trace_names.index("Soft voting") + 1
    best_trace = next(trace for trace in sequence_figure.data if trace.name == "Best deployable")
    assert "N-gram streak multipliers" in best_trace.hovertemplate
    assert "Combined multipliers" in best_trace.hovertemplate
    assert "Baseline soft distribution (top 3)" in best_trace.hovertemplate
    assert "Final modifier-weighted distribution (top 3)" in best_trace.hovertemplate
    assert "Rule-driven change" in best_trace.hovertemplate
    assert any("transient Bag favoritism" in item[13] for item in best_trace.customdata)
    tracked_figure = _sequence_figure(
        loaded_rows,
        [spec.name for spec in specs],
        best_sequence_candidate,
        [active_bag_name],
    )
    assert f"Rule active: {active_bag_name}" in {trace.name for trace in tracked_figure.data}
    soft_trace = next(trace for trace in sequence_figure.data if trace.name == "Soft voting")
    assert "Final soft distribution (top 3)" in soft_trace.hovertemplate
    model_trace = next(trace for trace in sequence_figure.data if trace.name == "model_a")
    assert "State access string" in model_trace.hovertemplate
    assert "State frequency" in model_trace.hovertemplate
    assert "State prediction accuracy" in model_trace.hovertemplate
    assert "Next activities (top 3)" in model_trace.hovertemplate
    assert "Prefix" not in model_trace.hovertemplate
    assert best_trace.line.width == 3
    cheating_trace = next(trace for trace in sequence_figure.data if trace.name == "Cheating voting")
    assert cheating_trace.line.dash == "dot"
    assert model_trace.customdata[0][7] == loaded_rows[0]["models"][0]["state_access_string"]

    mismatch_rows = deepcopy(loaded_rows)
    mismatch_candidate = {"name": "comparison candidate"}
    for row in mismatch_rows:
        row.setdefault("scenario_predictions", {})[mismatch_candidate["name"]] = "different from cheating"
    mismatch_figure = _sequence_figure(mismatch_rows, [spec.name for spec in specs], mismatch_candidate)
    assert len(mismatch_figure.layout.shapes) == len(mismatch_rows)
    assert mismatch_figure.layout.shapes[0].type == "rect"
    assert "2 mismatches" in _sequence_options(mismatch_rows, mismatch_candidate)[0]["label"]
    assert app.server.test_client().get("/").status_code == 200
    layout = app.server.test_client().get("/_dash-layout").get_json()
    assert "Soft-vote analysis" in str(layout)
    assert "Rule-set scenarios" in str(layout)
    assert "Integration methods" in str(layout)
    assert "Headline benchmark results" in str(layout)
    assert "Hard voting" not in str(layout)
    assert "info-button" in str(layout)
    assert "overview-panel overview-main-panel" in str(layout)


def test_dataset_named_results_initialize_global_dashboard(tmp_path: Path) -> None:
    specs = fixed_specs()
    investigator = VotingInvestigator(specs=specs)
    calibration = investigator.diagnose([[event("a")]], split="calibration")
    test_rows = investigator.diagnose([[event("a"), event("b")]], split="test")
    hypotheses = evaluate_hypotheses(calibration, test_rows, rules=[GlobalAccuracyRule()])
    for dataset in ("Sepsis_Cases", "fixture_b"):
        summary = build_summary(
            dataset_name=dataset,
            specs=specs,
            train_rows=0,
            calibration_rows=calibration,
            test_rows=test_rows,
            hypotheses=hypotheses,
        )
        summary["run_config"] = {"data_prop": 0.9, "windows": [2, 3], "seed": 0}
        save_results(dataset_result_dir(tmp_path, dataset), summary, test_rows)

    app = create_dashboard(tmp_path)
    layout = app.server.test_client().get("/_dash-layout").get_json()
    rendered = str(layout)

    assert app.server.test_client().get("/").status_code == 200
    assert "Cross-dataset voting investigation" in rendered
    assert "Global comparison" in rendered
    assert "Per-rule impact across datasets" in rendered
    assert "Selected dataset" in rendered
    assert "Sepsis_Cases" in rendered
    assert "fixture_b" in rendered
    assert "global-dataset-selector" in rendered
    assert "'value': 'Sepsis_Cases'" in rendered
