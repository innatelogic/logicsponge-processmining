"""Summarize paired held-out effects of social-choice rank aggregation rules."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

RULE_FAMILY = "social-choice rank aggregation"


def exact_discordance_pvalue(recoveries: int, harms: int) -> float:
    """Return the two-sided exact sign/McNemar p-value for paired disagreements."""
    discordant = recoveries + harms
    if not discordant:
        return 1.0
    lower_tail = sum(
        math.comb(discordant, successes) for successes in range(min(recoveries, harms) + 1)
    ) / (2**discordant)
    return min(1.0, 2.0 * lower_tail)


def result_rows(results_root: Path) -> list[dict[str, Any]]:
    """Load one row per social-choice rule and saved experiment run."""
    rows = []
    for summary_path in sorted(results_root.glob("*/summary.json")):
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        soft_accuracy = next(
            strategy["accuracy"]
            for strategy in summary["strategies"]
            if strategy["name"] == "soft voting"
        )
        hypothesis_by_name = {
            hypothesis["name"]: hypothesis
            for hypothesis in summary["hypotheses"]
            if hypothesis["family"] == RULE_FAMILY
        }
        for impact in summary["soft_failure_analysis"]["rule_impacts"]:
            if impact["family"] != RULE_FAMILY:
                continue
            hypothesis = hypothesis_by_name[impact["name"]]
            rows.append(
                {
                    "dataset": summary["dataset"],
                    "seed": summary["run_config"]["seed"],
                    "test_events": summary["test_events"],
                    "rule": impact["name"],
                    "soft_accuracy": soft_accuracy,
                    "calibration_accuracy": hypothesis["calibration_accuracy"],
                    "test_accuracy": impact["resulting_accuracy"],
                    "delta": impact["net_accuracy_delta"],
                    "triggered": impact["triggered"],
                    "recoveries": impact["recoveries"],
                    "harms": impact["harms"],
                    "net": impact["net_correct"],
                    "pvalue": exact_discordance_pvalue(impact["recoveries"], impact["harms"]),
                }
            )
    return rows


def markdown_table(rows: list[dict[str, Any]]) -> str:
    """Render exact run-level results as Markdown."""
    header = (
        "| Dataset | Seed | Rule | Calibration | Test | Delta vs soft | "
        "Overrides | Recoveries | Harms | Net | Exact p |\n"
        "|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|"
    )
    body = [
        (
            f"| {row['dataset']} | {row['seed']} | {row['rule']} | "
            f"{row['calibration_accuracy']:.2%} | {row['test_accuracy']:.2%} | "
            f"{row['delta']:+.2%} | {row['triggered']} | {row['recoveries']} | "
            f"{row['harms']} | {row['net']:+d} | {row['pvalue']:.4f} |"
        )
        for row in rows
    ]
    return "\n".join([header, *body])


def main() -> None:
    """Print a reproducible Markdown summary for an experiment result root."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "results",
        type=Path,
        help="Directory containing one DATASET_SEED/summary.json child per run.",
    )
    args = parser.parse_args()
    print(markdown_table(result_rows(args.results)))


if __name__ == "__main__":
    main()
