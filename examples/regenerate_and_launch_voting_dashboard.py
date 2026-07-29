"""
Regenerate voting experiments with current rules, verify them, and launch the dashboard.

The default result root is separate from historical runs, so every dataset shown
in the launched dashboard is guaranteed to have been produced by the current
code and to contain the required social-choice rank aggregation rules.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from logicsponge.processmining.voting_dashboard import run_dashboard
from logicsponge.processmining.voting_investigation import dataset_result_dir, run_investigation

DEFAULT_DATASETS = ("Sepsis_Cases", "Helpdesk", "BPI_Challenge_2013")
DEFAULT_RESULTS_ROOT = Path("results/social-choice-rank-dashboard")
REQUIRED_RULES = frozenset(
    {
        "Borda positional rank aggregation",
        "Copeland pairwise rank aggregation",
        "Maximin pairwise rank aggregation",
    }
)
logger = logging.getLogger(__name__)


def verify_rules(summary_path: Path) -> None:
    """Fail before dashboard launch when a regenerated summary lacks a required rule."""
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    saved_rules = {hypothesis["name"] for hypothesis in summary.get("hypotheses", [])}
    missing = REQUIRED_RULES - saved_rules
    if missing:
        names = ", ".join(sorted(missing))
        msg = f"{summary_path} is missing required rules: {names}"
        raise RuntimeError(msg)


def parser() -> argparse.ArgumentParser:
    """Build the command-line interface."""
    command = argparse.ArgumentParser(description=__doc__)
    command.add_argument(
        "--data",
        nargs="+",
        default=list(DEFAULT_DATASETS),
        help="Datasets to regenerate sequentially.",
    )
    command.add_argument("--data-prop", type=float, default=1.0)
    command.add_argument(
        "--windows",
        default="2,3,4,5,6",
        help="Comma-separated N-gram windows.",
    )
    command.add_argument("--seed", type=int, default=0)
    command.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS_ROOT)
    command.add_argument("--port", type=int, default=8050)
    command.add_argument(
        "--no-dashboard",
        action="store_true",
        help="Regenerate and verify results without starting the server.",
    )
    return command


def main() -> None:
    """Run every requested experiment and launch the verified result root."""
    args = parser().parse_args()
    windows = tuple(int(value) for value in args.windows.split(",") if value.strip())
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    for dataset in args.data:
        output_dir = dataset_result_dir(args.results_root, dataset)
        logger.info("Regenerating %s into %s", dataset, output_dir)
        run_investigation(
            dataset_name=dataset,
            data_prop=args.data_prop,
            windows=windows,
            output_dir=output_dir,
            seed=args.seed,
        )
        verify_rules(output_dir / "summary.json")
        logger.info("Verified all social-choice rules for %s", dataset)

    if args.no_dashboard:
        logger.info("All experiments regenerated and verified under %s", args.results_root.resolve())
        return

    logger.info("Launching dashboard at http://127.0.0.1:%d", args.port)
    logger.info("Press Ctrl+C to stop it.")
    run_dashboard(args.results_root, port=args.port)


if __name__ == "__main__":
    main()
