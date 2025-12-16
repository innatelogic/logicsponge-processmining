from __future__ import annotations

import csv
import logging
import math
import shutil
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rcParams

logger = logging.getLogger(__name__)


# ----------------------------------------------------------------------
# Matplotlib LaTeX-style font configuration (NO LaTeX dependency)
# ----------------------------------------------------------------------
def configure_latex_fonts() -> None:
    """
    Configure matplotlib to use LaTeX-like Computer Modern fonts.

    Does not require a LaTeX installation.
    """
    rcParams.update({
        "font.family": "serif",
        # Prefer Computer Modern Roman if available; otherwise fall back
        # to commonly-available serif fonts to avoid matplotlib warnings.
        "font.serif": [
            "Computer Modern Roman",
            "DejaVu Serif",
            "Times New Roman",
            "serif",
        ],
        "mathtext.fontset": "cm",
        "axes.unicode_minus": False,
        # Reduced a bit from the previous very-large sizes for better layout
        "axes.labelsize": 20,
        "axes.titlesize": 24,
        "legend.fontsize": 18,
        "xtick.labelsize": 16,
        "ytick.labelsize": 16,
    })


# ----------------------------------------------------------------------
# Core plotting logic
# ----------------------------------------------------------------------
def plot_accuracy_by_window_from_csv(csv_path: Path, desktop_label: str | None = None) -> None:
    """
    Generate accuracy-vs-window-size plot from a single summary CSV.

    Output PNG is written next to the CSV.
    """
    # family -> window_size -> list[accuracy]
    data: dict[str, dict[int, list[float]]] = {
        "ngram": defaultdict(list),
        "lstm": defaultdict(list),
        "transformer": defaultdict(list),
    }

    logger.debug("Processing summary CSV: %s", csv_path)
    with csv_path.open(newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        for row in reader:
            if not row or not row[0].strip():
                continue

            name = row[0].strip()

            # Accuracy column (percentage)
            try:
                acc = float(row[1])
            except (IndexError, ValueError):
                continue

            # ---------------- NGRAM ----------------
            if name.startswith("ngram_"):
                try:
                    k = int(name.split("_", 1)[1])
                    data["ngram"][k - 1].append(acc)
                except ValueError:
                    pass

            # ---------------- LSTM ----------------
            elif name.startswith("LSTM_win"):
                try:
                    w = int(name.replace("LSTM_win", ""))
                    data["lstm"][w].append(acc)
                except ValueError:
                    pass

            # ------------- TRANSFORMER -------------
            elif name.startswith("transformer_win"):
                try:
                    w = int(name.replace("transformer_win", ""))
                    data["transformer"][w].append(acc)
                except ValueError:
                    pass

    def mean_or_nan(vals: list[float]) -> float:
        return float(np.nanmean(vals)) if vals else float("nan")

    all_windows = sorted(
        set(data["ngram"]) |
        set(data["lstm"]) |
        set(data["transformer"])
    )

    if not all_windows:
        logger.debug("No windowed data found in %s; skipping plot.", csv_path)
        return

    xs = np.array(all_windows, dtype=float)
    ngram_vals = [mean_or_nan(data["ngram"].get(w, [])) for w in all_windows]
    lstm_vals = [mean_or_nan(data["lstm"].get(w, [])) for w in all_windows]
    transformer_vals = [
        mean_or_nan(data["transformer"].get(w, [])) for w in all_windows
    ]

    # ------------------------------------------------------------------
    # Plot
    # ------------------------------------------------------------------
    # Reduce width slightly while keeping same height for better layout
    plt.figure(figsize=(8, 5))
    plt.plot(xs, ngram_vals, marker="o", label="NGram")
    plt.plot(xs, transformer_vals, marker="s", label="Transformer")
    plt.plot(xs, lstm_vals, marker="^", label="LSTM")

    plt.xscale("log", base=2)

    # Powers-of-two ticks
    min_x = max(xs.min(), 1)
    max_x = xs.max()
    min_exp = math.floor(math.log2(min_x))
    max_exp = math.ceil(math.log2(max_x))
    ticks = [2**e for e in range(min_exp, max_exp + 1)]
    plt.xticks(ticks, [str(t) for t in ticks])
    plt.minorticks_off()

    plt.xlabel("Window size")
    plt.ylabel("Accuracy (%)")
    plt.ylim(0, 100)
    plt.title("Model accuracy vs window size")
    plt.grid(axis="y", linestyle="--", alpha=0.3)
    plt.grid(axis="x", linestyle="--", alpha=0.3)
    plt.legend()
    plt.tight_layout()

    out_png = csv_path.with_name(csv_path.stem.replace("summary", "accuracy_by_window") + ".png")
    plt.savefig(out_png, dpi=300)
    plt.close()
    logger.info("Wrote plot PNG for '%s': %s", csv_path.name, out_png)

    # Also copy the created PNG into a folder on the user's Desktop.
    # Use the provided `desktop_label` (root folder name) when available.
    try:
        folder_name = desktop_label if desktop_label else "logicsponge_plots"
        desktop_folder = Path.home() / "Desktop" / folder_name
        desktop_folder.mkdir(parents=True, exist_ok=True)
        desktop_target = desktop_folder / out_png.name
        shutil.copy2(out_png, desktop_target)
        logger.info("Also copied PNG to Desktop folder: %s", desktop_target)
    except Exception as e:  # pragma: no cover - non-critical
        logger.warning("Failed to copy PNG to Desktop folder: %s", e)


# ----------------------------------------------------------------------
# Folder traversal
# ----------------------------------------------------------------------
def process_root_folder(root_folder: Path) -> None:
    """
    Recursively process all *summary.csv files under root_folder.

    The Desktop copy folder uses only the final path component of
    `root_folder` (e.g. passing "results/final-fraction-1" ->
    Desktop folder `final-fraction-1`).
    """
    # Coerce to Path in case a string was passed; use only the last part.
    root_folder = Path(root_folder)
    desktop_label = root_folder.name if root_folder.name else root_folder.resolve().name

    for csv_file in root_folder.rglob("*summary.csv"):
        plot_accuracy_by_window_from_csv(csv_file, desktop_label=desktop_label)


# ----------------------------------------------------------------------
# CLI entry point
# ----------------------------------------------------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate accuracy-by-window plots from summary CSVs"
    )
    parser.add_argument(
        "root_folder",
        type=Path,
        help="Root folder containing subfolders with *summary.csv files",
    )

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    configure_latex_fonts()
    process_root_folder(args.root_folder)
