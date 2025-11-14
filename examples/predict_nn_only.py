"""
Quick NN-only evaluation script to validate windowed vs non-windowed accuracies
without running miners or voting strategies.

Usage:
  python examples/predict_nn_only.py --data Sepsis_Cases --epochs 30 --window 10

If --window is omitted, runs non-windowed.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import torch
from torch import nn, optim

from logicsponge.processmining.config import DEFAULT_CONFIG
from logicsponge.processmining.data_utils import (
    add_start_to_sequences,
    add_stop_to_sequences,
    data_statistics,
    split_sequence_data,
    transform_to_seqs,
)
from logicsponge.processmining.neural_networks import (
    LSTMModel,
    PreprocessData,
    TransformerModel,
    evaluate_rnn,
    train_rnn,
)
from logicsponge.processmining.utils import (
    parse_cli_args,
    prepare_synthetic_dataset,
    resolve_dataset_from_args,
)


def build_model(name: str, device: torch.device, cfg: dict[str, Any]):
    if name == "LSTM":
        model = LSTMModel(
            vocab_size=cfg.get("vocab_size", 64),
            embedding_dim=cfg.get("embedding_dim", 64),
            hidden_dim=cfg.get("hidden_dim", 128),
            use_one_hot=True,
            device=device,
        )
    elif name == "transformer":
        model = TransformerModel(
            vocab_size=cfg.get("vocab_size", 64),
            embedding_dim=cfg.get("embedding_dim", 64),
            hidden_dim=cfg.get("hidden_dim", 128),
            attention_heads=1,
            use_one_hot=True,
            device=device,
        )
    else:
        raise ValueError(f"Unknown model {name}")
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    return model, optimizer, criterion


def main() -> None:
    args = parse_cli_args()

    # Data resolve (support synthetic like other scripts)
    if getattr(args, "data", None) and str(args.data).lower().startswith("synthetic"):
        res = prepare_synthetic_dataset(args, [1, 1, 1, 0, 0, 0], total_activities=10000)
        data_name, dataset, dataset_test_opt = res if res is not None else resolve_dataset_from_args(args)
    else:
        data_name, dataset, dataset_test_opt = resolve_dataset_from_args(args)
    dataset_test = dataset_test_opt if dataset_test_opt is not None else dataset

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data = transform_to_seqs(dataset)
    n_acts, _ = data_statistics(data)

    run_cfg = {
        "nn": {"lr": 0.001, "batch_size": 8, "epochs": int(getattr(args, "epochs", 30) or 30)},
        "lstm": {"vocab_size": n_acts + 8, "embedding_dim": n_acts + 8, "hidden_dim": 128},
        "transformer": {"vocab_size": n_acts + 8, "embedding_dim": n_acts + 8, "hidden_dim": 512},
    }

    # Splits
    train_set, remainder = split_sequence_data(transform_to_seqs(dataset), 0.3, random_shuffle=True, seed=0)
    val_set, test_set = split_sequence_data(remainder, 0.5, random_shuffle=True, seed=0)

    # Add STOP for RNN training
    train_set = add_stop_to_sequences(train_set, DEFAULT_CONFIG["stop_symbol"])
    val_set = add_stop_to_sequences(val_set, DEFAULT_CONFIG["stop_symbol"])
    test_set = add_stop_to_sequences(test_set, DEFAULT_CONFIG["stop_symbol"])  # keep parity with batch script

    # Add START for RNNs
    start_symbol = DEFAULT_CONFIG["start_symbol"]
    train_set = add_start_to_sequences(train_set, start_symbol)
    val_set = add_start_to_sequences(val_set, start_symbol)
    test_set = add_start_to_sequences(test_set, start_symbol)

    proc = PreprocessData()
    train_tensor = proc.preprocess_data(train_set)
    val_tensor = proc.preprocess_data(val_set)
    test_tensor = proc.preprocess_data(test_set)

    window_size = getattr(args, "window", None)
    if window_size is not None:
        try:
            window_size = int(window_size)
        except Exception:
            window_size = None

    results: dict[str, Any] = {}
    for name in ("LSTM", "transformer"):
        model, opt, crit = build_model(name, device, run_cfg[name.lower()])
        t0 = time.time()
        model = train_rnn(
            model,
            train_tensor,
            val_tensor,
            crit,
            opt,
            batch_size=run_cfg["nn"]["batch_size"],
            epochs=run_cfg["nn"]["epochs"],
            window_size=window_size,
            left_pad=False,
        )
        train_time = time.time() - t0

        stats, perplexities, eval_time, prediction_vector = evaluate_rnn(
            model,
            test_tensor,
            max_k=3,
            idx_to_activity=proc.idx_to_activity,
            window_size=window_size,
            left_pad=False,
        )
        results_key = f"{name}_win{window_size}" if window_size is not None else name
        results[results_key] = {
            "accuracy": float(stats["accuracy"]),
            "total": stats["total_predictions"],
            "correct": stats["correct_predictions"],
            "topk": stats["top_k_correct_preds"],
            "train_time_sec": train_time,
            "eval_time_sec": eval_time,
            "pred_len": len(prediction_vector),
        }

    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
