"""
Interactive dashboard to explore correlations from completed streaming runs.

This script:
- Locates the latest run directory under `results/` (or a user-specified one)
- Reads per-model prediction CSVs from `predictions/`
- Builds an interactive Dash dashboard in your browser with:
    - Experiment selector (switch between available runs)
    - Strategy selector (tested model)
    - Reference selector (multi-select to mask reference curves)
    - Three time-series graphs: correlation, anticorrelation, similarity

Assumptions about CSV format (written by PredictionCSVWriter/ActualCSVWriter):
- `predictions/actual.csv` with columns: step, case_id, timestamp, actual
- `predictions/<model>.csv` with columns: step, case_id, timestamp, actual, predicted, top_k, correct

Usage:
    python examples/compute_correlations.py
    python examples/compute_correlations.py --run-dir results/2025-10-28_20-56_Sepsis_Cases

Requirements: dash, pandas, numpy
"""

from __future__ import annotations

import argparse
import webbrowser
from functools import lru_cache
from pathlib import Path
from threading import Timer
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objs as go
from dash import Dash, Input, Output, dcc, html, no_update

from logicsponge.processmining.config import DEFAULT_CONFIG


def find_latest_run(results_root: Path) -> Path | None:
    """
    Return the latest run directory under results_root by name (RUN_ID sort).

    RUN_ID format is YYYY-MM-DD_HH-MM_<dataset>, so lexicographic sort works.
    """
    if not results_root.exists():
        return None
    subdirs = [d for d in results_root.iterdir() if d.is_dir()]
    if not subdirs:
        return None
    # Sort by name descending (latest first)
    subdirs.sort(key=lambda p: p.name, reverse=True)
    return subdirs[0]


def find_all_runs(results_root: Path) -> list[Path]:
    """Return all run directories under results_root sorted by name descending (latest first)."""
    if not results_root.exists():
        return []
    subdirs = [d for d in results_root.iterdir() if d.is_dir()]
    subdirs.sort(key=lambda p: p.name, reverse=True)
    return subdirs


def load_sequences(pred_dir: Path) -> tuple[list[str], dict[str, list[str]]]: # noqa: C901
    """
    Load baseline actual and per-model predicted sequences from CSV files.

    Returns (actual_list, model_to_pred_list). Lists are trimmed to a common min length.
    """
    actual_path = pred_dir / "actual.csv"
    if not actual_path.exists():
        msg = f"Missing baseline file: {actual_path}"
        raise FileNotFoundError(msg)

    # Load actuals
    df_actual = pd.read_csv(actual_path)
    if "actual" not in df_actual.columns:
        msg = f"Baseline file missing 'actual' column: {actual_path}"
        raise ValueError(msg)
    # Ensure sorted by step if available
    if "step" in df_actual.columns:
        df_actual = df_actual.sort_values("step")
    actual_list = df_actual["actual"].astype(str).tolist()

    # Load model predictions
    model_series: dict[str, list[str]] = {}
    for csv_file in sorted(pred_dir.glob("*.csv")):
        name = csv_file.stem
        if name == "actual":
            continue
        try:
            pred_df = pd.read_csv(csv_file)
            if "predicted" not in pred_df.columns:
                # skip files without predicted column
                continue
            if "step" in pred_df.columns:
                pred_df = pred_df.sort_values("step")
            preds = pred_df["predicted"].astype(str).tolist()
            model_series[name] = preds
        except (OSError, ValueError, pd.errors.ParserError, FileNotFoundError) as e:
            # Skip malformed files
            print(f"Warning: could not read predictions from {csv_file}: {e}")
            continue

    # Trim all lists to common min length
    if not model_series:
        msg = f"No model prediction CSVs found in: {pred_dir}"
        raise RuntimeError(msg)
    min_len = min([len(actual_list)] + [len(v) for v in model_series.values()])
    actual_list = actual_list[:min_len]
    for k in list(model_series.keys()):
        model_series[k] = model_series[k][:min_len]

    return actual_list, model_series


@lru_cache(maxsize=2048)
def compute_pair_cumsums(
    tested: str, reference: str, *, actual: tuple[str, ...], models: tuple[tuple[str, tuple[str, ...]], ...]
) -> dict[str, np.ndarray]:
    """
    Compute cumulative counts arrays for a given (tested, reference) pair.

    We pack the sequences in the cache key via tuples to make it hashable for lru_cache.
    Returns a dict with keys:
      - ref_correct, both_correct, ref_incorrect, tested_correct_when_ref_incorrect, similarity_matches
    Each value is a numpy array of shape (n,) with cumulative counts up to index i (inclusive, 0-based).
    """
    # Unpack model sequences from the tuple mapping
    models_dict = {name: list(seq) for name, seq in models}
    act = list(actual)
    test_seq = list(models_dict[tested])
    # Support using baseline 'actual' as a reference model
    ref_seq = act if reference == "actual" else list(models_dict[reference])
    n = min(len(act), len(test_seq), len(ref_seq))

    # Convert to numpy arrays of dtype object for safe equality checks
    a = np.asarray(act[:n], dtype=object)
    t = np.asarray(test_seq[:n], dtype=object)
    r = np.asarray(ref_seq[:n], dtype=object)

    # Remove positions where either tested or reference produced an empty prediction
    # Empty prediction symbol is defined in DEFAULT_CONFIG
    empty_symbol = DEFAULT_CONFIG.get("empty_symbol")
    if empty_symbol is not None:
        valid_mask = (t != empty_symbol) & (r != empty_symbol)
        # Apply mask to all three arrays
        a = a[valid_mask]
        t = t[valid_mask]
        r = r[valid_mask]
    # Recompute n after filtering
    n = len(a)

    eq_ref_base = (r == a)
    eq_test_base = (t == a)
    eq_test_ref = (t == r)

    both_correct_mask = eq_ref_base & eq_test_base
    ref_incorrect_mask = ~eq_ref_base
    tested_correct_when_ref_incorrect_mask = ref_incorrect_mask & eq_test_base

    # Cumulative sums
    ref_correct_cs = np.cumsum(eq_ref_base.astype(np.int64))
    both_correct_cs = np.cumsum(both_correct_mask.astype(np.int64))
    ref_incorrect_cs = np.cumsum(ref_incorrect_mask.astype(np.int64))
    t_when_r_incorrect_cs = np.cumsum(tested_correct_when_ref_incorrect_mask.astype(np.int64))
    similarity_cs = np.cumsum(eq_test_ref.astype(np.int64))
    tested_correct_cs = np.cumsum(eq_test_base.astype(np.int64))

    return {
        "ref_correct": ref_correct_cs,
        "both_correct": both_correct_cs,
        "ref_incorrect": ref_incorrect_cs,
        "tested_correct_when_ref_incorrect": t_when_r_incorrect_cs,
        "similarity_matches": similarity_cs,
        "tested_correct": tested_correct_cs,
    }


def sample_metrics_from_cumsums(
    cums: dict[str, np.ndarray], stride: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Given cumulative counts and a stride, return (t_points, corr, anticorr, sim) arrays.

    Deprecated in favor of sample_metrics_with_horizon but kept for backward compatibility.
    """
    n = len(cums["ref_correct"]) if cums else 0
    if n == 0:
        return np.array([]), np.array([]), np.array([]), np.array([])

    if stride <= 0:
        stride = 100
    t_points = np.arange(stride, n + 1, stride)
    # Use -1 index for cumulative up to t
    idxs = t_points - 1

    ref_correct = cums["ref_correct"][idxs].astype(float)
    both_correct = cums["both_correct"][idxs].astype(float)
    ref_incorrect = cums["ref_incorrect"][idxs].astype(float)
    t_when_r_inc = cums["tested_correct_when_ref_incorrect"][idxs].astype(float)
    similarity_matches = cums["similarity_matches"][idxs].astype(float)

    with np.errstate(divide="ignore", invalid="ignore"):
        corr = np.where(ref_correct > 0, both_correct / ref_correct, np.nan)
        anticorr = np.where(ref_incorrect > 0, t_when_r_inc / ref_incorrect, np.nan)
        sim = similarity_matches / t_points

    return t_points, corr, anticorr, sim


def sample_metrics_with_horizon(
    cums: dict[str, np.ndarray], stride: int, horizon: str | int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Sample metrics at a given stride with optional sliding window horizon.

    Returns (t_points, corr, anticorr, sim, acc).

    - horizon == "INF" (or any case-insensitive variant) uses cumulative-from-start metrics.
    - horizon as an int (e.g., 100, 500, 1000) computes metrics over the last N predictions.
    """
    n = len(cums.get("ref_correct", [])) if cums else 0
    if n == 0:
        return np.array([]), np.array([]), np.array([]), np.array([]), np.array([])

    if stride <= 0:
        stride = 100
    t_points = np.arange(stride, n + 1, stride)
    idxs = t_points - 1  # convert to 0-based inclusive index

    # Prepare cumulative arrays and a zero-prepended variant for window diffs
    def prep(cs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        cs = cs.astype(np.int64)
        return cs, np.concatenate(([0], cs))

    ref_correct, ref_correct0 = prep(cums["ref_correct"])  # type: ignore[index]
    both_correct, both_correct0 = prep(cums["both_correct"])  # type: ignore[index]
    ref_incorrect, ref_incorrect0 = prep(cums["ref_incorrect"])  # type: ignore[index]
    t_when_r_inc, t_when_r_inc0 = prep(cums["tested_correct_when_ref_incorrect"])  # type: ignore[index]
    similarity_matches, similarity_matches0 = prep(cums["similarity_matches"])  # type: ignore[index]
    tested_correct, tested_correct0 = prep(cums["tested_correct"])  # type: ignore[index]

    if isinstance(horizon, str) and horizon.upper() == "INF":
        # cumulative-from-start behavior
        ref_correct_vals = ref_correct[idxs]
        both_correct_vals = both_correct[idxs]
        ref_incorrect_vals = ref_incorrect[idxs]
        t_when_r_inc_vals = t_when_r_inc[idxs]
        similarity_vals = similarity_matches[idxs]
        tested_correct_vals = tested_correct[idxs]

        denom_steps = t_points.astype(float)
        with np.errstate(divide="ignore", invalid="ignore"):
            corr = np.where(ref_correct_vals > 0, both_correct_vals / ref_correct_vals, np.nan)
            anticorr = np.where(ref_incorrect_vals > 0, t_when_r_inc_vals / ref_incorrect_vals, np.nan)
            sim = similarity_vals / denom_steps
            acc = tested_correct_vals / denom_steps
        return t_points, corr, anticorr, sim, acc

    # sliding window with integer horizon
    try:
        win = int(horizon)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        win = 100

    # Convert idxs to 1-based for use with zero-prepended arrays
    idxs1 = idxs + 1
    prev_idxs1 = np.clip(idxs1 - win, 0, None)

    # Window sums by subtracting cumulative at (t-win)
    ref_correct_w = ref_correct0[idxs1] - ref_correct0[prev_idxs1]
    both_correct_w = both_correct0[idxs1] - both_correct0[prev_idxs1]
    ref_incorrect_w = ref_incorrect0[idxs1] - ref_incorrect0[prev_idxs1]
    t_when_r_inc_w = t_when_r_inc0[idxs1] - t_when_r_inc0[prev_idxs1]
    similarity_w = similarity_matches0[idxs1] - similarity_matches0[prev_idxs1]
    tested_correct_w = tested_correct0[idxs1] - tested_correct0[prev_idxs1]

    # Effective window length near the start grows from 1..win
    eff_win = np.minimum(t_points, win).astype(float)
    with np.errstate(divide="ignore", invalid="ignore"):
        corr = np.where(ref_correct_w > 0, both_correct_w / ref_correct_w, np.nan)
        anticorr = np.where(ref_incorrect_w > 0, t_when_r_inc_w / ref_incorrect_w, np.nan)
        sim = similarity_w / eff_win
        acc = tested_correct_w / eff_win

    return t_points, corr, anticorr, sim, acc


def build_app(results_root: Path, run_dir: Path) -> Dash:  # noqa: C901, PLR0915
    """Create and return the Dash app with an experiment selector for runs under results_root."""
    # Prepare initial data from the selected run
    pred_dir = run_dir / "predictions"
    if not pred_dir.is_dir():
        msg = f"Predictions folder not found in: {run_dir}"
        raise FileNotFoundError(msg)

    actual_list, model_series = load_sequences(pred_dir)
    strategies = sorted(model_series.keys())
    # Allow referencing the baseline as a reference model
    reference_candidates = [*strategies, "actual"]
    default_tested = "ngram_3" if "ngram_3" in strategies else (strategies[0] if strategies else None)
    if default_tested is None:
        msg = "No strategies found to visualize."
        raise RuntimeError(msg)

    # Pack sequences into tuples for cacheable functions
    actual_tuple = tuple(actual_list)

    # Discover available experiments (runs)
    all_runs = find_all_runs(results_root)
    run_options = [{"label": p.name, "value": str(p)} for p in all_runs]
    default_run_value = str(run_dir)

    app = Dash(__name__)
    app.title = f"Correlations - {run_dir.name}"

    app.layout = html.Div(
        style={"fontFamily": "sans-serif", "margin": "12px"},
        children=[
            html.H2(id="run-name", children=f"Run: {run_dir.name}"),
            html.Div(
                style={"display": "flex", "gap": "12px", "flexWrap": "wrap", "alignItems": "center"},
                children=[
                    html.Div([
                        html.Label("Experiment"),
                        dcc.Dropdown(
                            id="experiment",
                            options=run_options,  # type: ignore[arg-type]
                            value=default_run_value,
                            clearable=False,
                        ),
                    ], style={"minWidth": "320px"}),
                    html.Div([
                        html.Label("Tested strategy"),
                        dcc.Dropdown(
                            id="tested-model",
                            options=[{"label": s, "value": s} for s in strategies],
                            value=default_tested,
                            clearable=False,
                        ),
                    ], style={"minWidth": "280px"}),
                    html.Div([
                        html.Label("Reference strategies (mask/unmask)"),
                        dcc.Dropdown(
                            id="reference-models",
                            options=[{"label": s, "value": s} for s in reference_candidates],
                            value=[s for s in reference_candidates if s != default_tested],
                            multi=True,
                        ),
                    ], style={"minWidth": "360px", "flexGrow": 1}),
                    html.Div([
                        html.Label("Sampling stride (iterations)"),
                        dcc.Slider(
                            id="stride",
                            min=10,
                            max=2000,
                            step=None,
                            marks={10: "10", 50: "50", 100: "100", 200: "200", 500: "500", 1000: "1000", 2000: "2000"},
                            value=100,
                        ),
                    ], style={"minWidth": "360px", "maxWidth": "520px"}),
                    html.Div([
                        html.Label("Metric horizon"),
                        dcc.Dropdown(
                            id="horizon",
                            options=[
                                {"label": "INF (cumulative)", "value": "INF"},
                                {"label": "100 (last 100)", "value": 100},
                                {"label": "500 (last 500)", "value": 500},
                                {"label": "1000 (last 1000)", "value": 1000},
                            ],
                            value="INF",
                            clearable=False,
                        ),
                    ], style={"minWidth": "240px"}),
                ],
            ),
            html.Hr(),
            html.Div([
                html.H3("Correlation (tested vs reference)"),
                dcc.Graph(id="graph-corr"),
            ]),
            html.Div([
                html.H3("Anticorrelation (tested vs reference)"),
                dcc.Graph(id="graph-anticorr"),
            ]),
            html.Div([
                html.H3("Similarity (tested vs reference)"),
                dcc.Graph(id="graph-sim"),
            ]),
            html.Div([
                html.H3("Accuracy (tested vs actual)"),
                dcc.Graph(id="graph-acc"),
            ]),
            # Hidden stores for sequences to use in callbacks
            dcc.Store(id="store-actual", data=list(actual_tuple)),
            dcc.Store(id="store-models", data={k: list(v) for k, v in model_series.items()}),
        ],
    )

    @app.callback(
        Output("store-actual", "data"),
        Output("store-models", "data"),
        Output("tested-model", "options"),
        Output("tested-model", "value"),
        Output("reference-models", "options"),
        Output("reference-models", "value"),
        Output("run-name", "children"),
        Input("experiment", "value"),
    )
    def on_experiment_change(experiment_path: str) -> tuple[Any, Any, Any, Any, Any, Any, Any]:
        """When the experiment (run) changes, reload sequences and update selectors."""
        try:
            run_path = Path(experiment_path)
            pred_dir_local = run_path / "predictions"
            actual_l, model_s = load_sequences(pred_dir_local)
            strats = sorted(model_s.keys())
            if not strats:
                msg = "No strategies found in selected run"
                # Avoid raising an exception inside the inner callback; return no_update so the UI remains stable
                print(f"Warning: {msg} in {experiment_path}")
                return no_update, no_update, no_update, no_update, no_update, no_update, no_update
            default_test = "ngram_3" if "ngram_3" in strats else strats[0]
            ref_cands = [*strats, "actual"]

            return (
                list(actual_l),
                {k: list(v) for k, v in model_s.items()},
                [{"label": s, "value": s} for s in strats],
                default_test,
                [{"label": s, "value": s} for s in ref_cands],
                [s for s in ref_cands if s != default_test],
                f"Run: {run_path.name}",
            )
        except Exception as exc:  # noqa: BLE001 - display failure but don't crash UI
            print(f"Warning: failed to switch experiment to {experiment_path}: {exc}")
            return no_update, no_update, no_update, no_update, no_update, no_update, no_update

    # Horizon selector added: INF for cumulative, or a numeric window (e.g., 100, 500, 1000)
    @app.callback(
        Output("graph-corr", "figure"),
        Output("graph-anticorr", "figure"),
        Output("graph-sim", "figure"),
        Output("graph-acc", "figure"),
        Input("tested-model", "value"),
        Input("reference-models", "value"),
        Input("stride", "value"),
        Input("store-actual", "data"),
        Input("store-models", "data"),
        Input("horizon", "value"),
    )
    def update_graphs(  # noqa: PLR0913
        tested: str,
        references: list[str],
        stride: int,
        actual_data: list[str],
        models_data: dict[str, list[str]],
        horizon_value: str | int,
    ) -> tuple[go.Figure, go.Figure, go.Figure, go.Figure]:
        # Repack for cache lookup
        actual_t = tuple(actual_data)
        models_t = tuple((k, tuple(v)) for k, v in models_data.items())

        # Ensure references default to all except tested
        if not references:
            # Default to all models plus the baseline 'actual', except the tested model
            references = [*models_data.keys(), "actual"]
            references = [k for k in references if k != tested]

        corr_traces = []
        anticorr_traces = []
        sim_traces = []
        acc_traces = []

        for ref in references:
            if ref == tested:
                continue
            try:
                cums = compute_pair_cumsums(tested, ref, actual=actual_t, models=models_t)
                t_points, corr, anticorr, sim, _ = sample_metrics_with_horizon(cums, int(stride), horizon_value)
            except (KeyError, ValueError) as exc:
                # Skip problematic pairs but log to console
                print(f"Warning: failed to compute series for tested={tested} ref={ref}: {exc}")
                continue

            corr_traces.append(go.Scatter(x=t_points, y=corr, mode="lines", name=ref))
            anticorr_traces.append(go.Scatter(x=t_points, y=anticorr, mode="lines", name=ref))
            sim_traces.append(go.Scatter(x=t_points, y=sim, mode="lines", name=ref))

        # Accuracy is independent of reference; compute once against any cums based on tested vs actual
        # Use tested-vs-actual by calling with reference="actual" sentinel inside helper
        try:
            cums_acc = compute_pair_cumsums(tested, "actual", actual=actual_t, models=models_t)
            t_points_acc, _, _, _, acc_vals = sample_metrics_with_horizon(cums_acc, int(stride), horizon_value)
            acc_traces.append(go.Scatter(x=t_points_acc, y=acc_vals, mode="lines", name="accuracy"))
        except Exception as exc:  # noqa: BLE001 - keep UI responsive
            print(f"Warning: failed to compute accuracy for tested={tested}: {exc}")

        layout_common = {
            "xaxis": {"title": "Iteration"},
            "yaxis": {"title": "Ratio", "range": [0, 1]},
            "margin": {"l": 40, "r": 10, "t": 20, "b": 40},
            "legend": {"orientation": "h", "y": -0.2},
        }

        fig_corr = go.Figure(data=corr_traces, layout=go.Layout(**layout_common))
        fig_corr.update_layout(title=f"Correlation vs references (tested: {tested})")
        fig_anticorr = go.Figure(data=anticorr_traces, layout=go.Layout(**layout_common))
        fig_anticorr.update_layout(title=f"Anticorrelation vs references (tested: {tested})")
        fig_sim = go.Figure(data=sim_traces, layout=go.Layout(**layout_common))
        fig_sim.update_layout(title=f"Similarity vs references (tested: {tested})")

        fig_acc = go.Figure(data=acc_traces, layout=go.Layout(**layout_common))
        fig_acc.update_layout(title=f"Accuracy (tested vs actual): {tested}")

        return fig_corr, fig_anticorr, fig_sim, fig_acc

    # Store tuples for cache key usage inside callbacks via closure variables
    return app


def main(argv: list[str] | None = None) -> int:
    """Entry point: parse args, prepare the app, and serve the dashboard."""
    parser = argparse.ArgumentParser(description="Interactive correlation dashboard for streaming results")
    parser.add_argument(
        "--run-dir",
        type=str,
        default=None,
        help="Path to a specific run directory (results/<RUN_ID>). If not provided, the latest is used.",
    )
    parser.add_argument("--port", type=int, default=8050, help="Port to run the dashboard on")
    parser.add_argument("--open", dest="auto_open", action="store_true", help="Open the browser automatically")
    args = parser.parse_args(argv)

    project_root = Path(__file__).resolve().parents[1]
    results_root = project_root / "results"

    run_dir = Path(args.run_dir) if args.run_dir else find_latest_run(results_root)
    if run_dir is None or not run_dir.exists():
        print(f"No run directory found under {results_root}. Run predict_streaming.py first.")
        return 1

    try:
        app = build_app(results_root, run_dir)
    except (FileNotFoundError, RuntimeError, ValueError) as e:
        print(f"Failed to prepare dashboard for {run_dir}: {e}")
        return 2

    url = f"http://127.0.0.1:{args.port}/"
    if args.auto_open:
        Timer(1.0, lambda: webbrowser.open(url)).start()

    # Run the server
    app.run(debug=False, port=args.port, use_reloader=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
