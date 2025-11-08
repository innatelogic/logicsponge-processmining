"""Module to evaluate and compare different process mining models."""

import json
import logging
import time
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from typing import Any

import matplotlib as mpl
import numpy as np
import pandas as pd
import torch
from torch import nn, optim
from tqdm import tqdm

# ruff: noqa: E402
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
)

# Load run configuration (learning rates, batch sizes, epochs for NN and RL)
config_file_path = Path(__file__).parent / "predict_config.json"
default_run_config = {
    "nn": {"lr": 0.001, "batch_size": 8, "epochs": 20},
    "rl": {"lr": 0.001, "batch_size": 8, "epochs": 20, "gamma": 0.99},
    "lstm": {"vocab_size": 32, "embedding_dim": 32, "hidden_dim": 128, "output_dim": 32},
    "transformer": {"vocab_size": 32, "embedding_dim": 32, "hidden_dim": 128, "output_dim": 32},
    "qlearning": {"vocab_size": 32, "embedding_dim": 32, "hidden_dim": 512, "output_dim": 32},
}

# Write the default run configuration into `predict_config.json` in the repo folder.
# We intentionally write the defaults here (instead of loading) so that users get a
# populated config file they can edit. If writing fails we fall back to in-memory defaults.
try:
    with config_file_path.open("w") as _f:
        json.dump(default_run_config, _f, indent=2)
    run_config = default_run_config
except OSError as _e:
    logging.getLogger(__name__).debug("Could not write default config to %s: %s", config_file_path, _e)
    run_config = default_run_config

from logicsponge.processmining.algorithms_and_structures import (
    BayesianClassifier,
)
from logicsponge.processmining.batch_helpers import (
    build_and_save_comparison_matrices,
    build_strategies,
    prefix_evaluate_rnn,
    record_model_results,
    write_prediction_vectors,
)
from logicsponge.processmining.config import DEFAULT_CONFIG
from logicsponge.processmining.data_utils import (
    add_input_symbols,
    add_start_to_sequences,
    add_stop_to_sequences,
    data_statistics,
    interleave_sequences,
    split_sequence_data,
    transform_to_seqs,
)
from logicsponge.processmining.miners import BasicMiner
from logicsponge.processmining.neural_networks import (
    LSTMModel,
    PreprocessData,
    QNetwork,
    TransformerModel,
    evaluate_rl,
    evaluate_rnn,
    train_rl,
    train_rnn,
)
from logicsponge.processmining.test_data import data_name, dataset, dataset_test
from logicsponge.processmining.utils import (
    add_file_log_handler,
    compute_perplexity_stats,
    save_run_config,
)

SEC_TO_MICRO = 1_000_000


def lstm_model() -> tuple[LSTMModel, optim.Optimizer, nn.Module]:
    """Initialize and return an LSTM model, optimizer, and loss function."""
    model = LSTMModel(
        vocab_size=run_config.get("lstm", {}).get("vocab_size", 64),
        embedding_dim=run_config.get("lstm", {}).get("embedding_dim", 64),
        hidden_dim=run_config.get("lstm", {}).get("hidden_dim", 128),
        use_one_hot=True,
        device=device
    )
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=run_config.get("nn", {}).get("lr", 0.001))
    return model, optimizer, criterion


def transformer_model(attention_heads: int, pos_encoding: str | None = None) -> tuple[TransformerModel, optim.Optimizer, nn.Module]:
    """Initialize and return a Transformer model, optimizer, and loss function."""
    # Keep backward-compatible signature but allow optional positional encoding
    def _build_model(pos_encoding: str | None = None) -> TransformerModel:
        kwargs: dict[str, object] = dict(
            seq_input_dim=64,  # instead of max_seq_length + 2
            vocab_size=run_config.get("transformer", {}).get("vocab_size", 64),
            embedding_dim=run_config.get("transformer", {}).get("embedding_dim", 64),
            hidden_dim=run_config.get("transformer", {}).get("hidden_dim", 128),
            output_dim=run_config.get("transformer", {}).get("output_dim", 64),
            attention_heads=attention_heads,
            use_one_hot=True,
            device=device,
        )
        if pos_encoding is not None:
            # TransformerModel in this repo expects `pos_encoding_type` kwarg
            kwargs["pos_encoding_type"] = pos_encoding
        return TransformerModel(**kwargs)

    model = _build_model(pos_encoding)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=run_config.get("nn", {}).get("lr", 0.001))
    return model, optimizer, criterion

def process_neural_model(  # noqa: PLR0913
    name: str,
    iteration_data: dict,
    all_metrics: dict,
    nn_train_set_transformed: torch.Tensor,
    nn_val_set_transformed: torch.Tensor,
    nn_test_set_transformed: torch.Tensor,
    epochs: int = 20,
    *,
    window_size: int | None = None,
) -> None:
    """
    Train and evaluate a NN model.

    When `window_size` is provided, the model is trained/evaluated on the last
    `window_size` tokens of each sequence (mimicking the qlearning windowing).
    Results are stored under a display name `f"{name}_win{window_size}"` so they
    can be compared alongside the RL models.
    """
    display_name = f"{name}_win{window_size}" if window_size is not None else name

    # Support transformer positional-encoding variants named like
    # "transformer_pos_<encoding>" and optionally windowed variants
    # (windowing is controlled by the `window_size` argument passed
    # by the caller; the name may include a trailing "_winX" but we
    # prefer the explicit `window_size` parameter).
    if name.startswith("transformer_pos_"):
        enc = name[len("transformer_pos_") :]
        # strip trailing window suffix if present in the name
        if "_win" in enc:
            enc = enc.split("_win")[0]
        model, optimizer, criterion = transformer_model(attention_heads=1, pos_encoding=enc)
    else:
        match name:
            case "LSTM":
                model, optimizer, criterion = lstm_model()
            case "transformer":
                model, optimizer, criterion = transformer_model(attention_heads=1)
            case "transformer_2heads":
                model, optimizer, criterion = transformer_model(attention_heads=2)
            case _:
                msg = "Unknown NN model."
                raise ValueError(msg)

    # Train the model on (optionally) windowed prefixes
    start_time = time.time()
    model = train_rnn(
        model,
        nn_train_set_transformed,
        nn_val_set_transformed,
        criterion,
        optimizer,
        batch_size=run_config.get("nn", {}).get("batch_size", 8),
        epochs=epochs,
        window_size=window_size,
    )
    end_time = time.time()
    training_time = (end_time - start_time) * SEC_TO_MICRO / (TRAIN_EVENTS + VAL_EVENTS)

    # If a window_size is provided we want to evaluate the model in "prefix" mode
    # (one prediction per prefix), so that the resulting flattened prediction vector
    # aligns with the baseline `actual` vector and with how RL/qlearning is evaluated.
    if window_size is None:
        stats, perplexities, eval_time, prediction_vector = evaluate_rnn(
            model,
            nn_test_set_transformed,
            max_k=config["top_k"],
            idx_to_activity=nn_processor.idx_to_activity,
            window_size=None,
        )
    else:
        # Refactored: use helper preserving original behavior
        stats, perplexities, eval_time, prediction_vector = prefix_evaluate_rnn(
            model=model,
            sequences=nn_test_set_transformed,
            idx_to_activity=nn_processor.idx_to_activity,
            max_k=config["top_k"],
            window_size=window_size,
        )

    # Store neural model prediction vector in global memory under display name
    prediction_vectors_memory.setdefault(display_name, []).append(prediction_vector)

    # Sanity/length alignment check: ensure NN produced the same number of
    # predictions as the baseline `actual` vector for this iteration. The
    # `actual` vector is appended earlier in the outer loop; here we compare
    # against the most recent `actual` entry so misalignments are detected
    # early and fail fast.
    actual_iters = prediction_vectors_memory.get("actual", [])
    if actual_iters:
        # The current iteration's baseline is expected to be the last one
        actual_len = len(actual_iters[-1])
        if len(prediction_vector) != actual_len:
            msg = (
                f"Length mismatch for NN model '{display_name}': preds={len(prediction_vector)} "
                f"vs actual={actual_len}. This indicates evaluation misalignment."
            )
            logger.exception(msg)
            raise ValueError(msg)

    perplexity_stats = compute_perplexity_stats(perplexities)
    eval_time *= SEC_TO_MICRO / TEST_EVENTS

    if (
        not isinstance(stats["top_k_correct_preds"], list)
        or not isinstance(stats["total_predictions"], int)
        or not isinstance(stats["accuracy"], float)
    ):
        msg = f"{display_name} stats are not in the expected format."
        raise TypeError(msg)

    # Unified recording
    record_model_results(
        display_name=display_name,
        stats={
            "accuracy": stats["accuracy"],
            "total_predictions": stats["total_predictions"],
            "correct_predictions": stats["correct_predictions"],
            "top_k_correct_preds": stats["top_k_correct_preds"],
        },
        perplexity_stats=perplexity_stats,
        eval_time_micro=eval_time,
        train_time_micro=training_time,
        num_states=None,
        top_k=config["top_k"],
        iteration_data=iteration_data,
        all_metrics=all_metrics,
        stats_to_log=stats_to_log,
    )


# ============================================================
# Generate a list of ngrams to test
# ============================================================
VOTING_NGRAMS = [(2, 3, 4), (2, 3, 5, 8), (2, 3, 4, 5)]  # (2, 3, 5, 6), (2, 3, 5, 7), (2, 3, 4, 7)

SELECT_BEST_ARGS = ["prob"]  # ["acc", "prob", "prob x acc"]

WINDOW_RANGE = [1, 2, 3, 4, 5, 6, 7, 8]  # , 9, 10, 12, 14, 16]

NGRAM_NAMES = [f"ngram_{i + 1}" for i in WINDOW_RANGE]
# ] + [
#     f"ngram_{i+1}_recovery" for i in WINDOW_RANGE
# ]
# ] + [
#     f"ngram_{i+1}_shorts" for i in WINDOW_RANGE
# ]

# ============================================================

# Positional encodings for transformer variants (parity with predict_streaming)
# Add any encodings used in predict_streaming here so batch experiments include them.
POS_ENCODINGS = [
    ("learnable_relative", None),
]



def qnetwork_model() -> tuple[QNetwork, optim.Optimizer, nn.Module]:
    """Initialize a QNetwork model for RL training."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = QNetwork(
        vocab_size=run_config.get("qlearning", {}).get("vocab_size", 32),
        embedding_dim=run_config.get("qlearning", {}).get("embedding_dim", 32),
        hidden_dim=run_config.get("qlearning", {}).get("hidden_dim", 1024),
        device=device,
    ).to(device)
    optimizer = optim.Adam(model.parameters(), lr=run_config.get("rl", {}).get("lr", 0.001))
    criterion = nn.MSELoss()  # Use MSE for Q-learning

    return model, optimizer, criterion


@dataclass
class RLModelResults:
    """Results from training/evaluating an RL model."""

    acc: float
    top_2: float
    top_3: float
    perplexities: list[float]
    train_time: float


def process_rl_model(
    _name: str,
    window_size: int | None,
    _iteration_data: dict[str, Any],
    nn_train_set_transformed: torch.Tensor,
    nn_val_set_transformed: torch.Tensor,
    nn_eval_set_transformed: torch.Tensor,
    epochs: int = 20,
) -> tuple[dict[str, Any], list[Any], float, list[Any], float]:
    """
    Train and evaluate a Q-learning model with specified window size.

    Evaluation is performed on the same sequences as other models (test set),
    transformed to tensor format, and intentionally WITHOUT an added START token
    so that prediction vectors align with the common "actual" baseline.
    """
    start_time = time.time()

    # Initialize model
    model, optimizer, criterion = qnetwork_model()

    # Train model
    model = train_rl(
        model=model,
        train_sequences=nn_train_set_transformed,
        val_sequences=nn_val_set_transformed,
        criterion=criterion,
        optimizer=optimizer,
        batch_size=run_config.get("rl", {}).get("batch_size", 4),
        epochs=epochs,
        window_size=window_size,
        gamma=run_config.get("rl", {}).get("gamma", 0.99),  # Standard RL discount factor
    )

    train_time = time.time() - start_time

    # Evaluate on the common evaluation set (same sequences as other models)
    model.eval()
    with torch.no_grad():
        # evaluate_rl returns: metrics, perplexities, eval_time, prediction_vector
        metrics, eval_pp, eval_time, prediction_vector = evaluate_rl(
            model=model,
            sequences=nn_eval_set_transformed,
            max_k=3,
            idx_to_activity=nn_processor.idx_to_activity,
            window_size=window_size,
        )

    # Return the relevant outputs so the caller can integrate them into iteration records
    return metrics, eval_pp, eval_time, prediction_vector, train_time

mpl.use("Agg")

pd.set_option("display.max_columns", None)  # Show all columns
pd.set_option("display.expand_frame_repr", False)  # Prevent line-wrapping # noqa: FBT003

logger = logging.getLogger(__name__)

# Log the resolved run configuration
try:
    logger.info("Run config:\n%s", json.dumps(run_config, indent=2))
except (TypeError, OSError):
    # If logging the config fails for any reason (non-serializable value), write a simple fallback message
    logger.info("Run config: (could not serialize run_config)")

RUN_ID = time.strftime("%Y-%m-%d_%H-%M", time.localtime()) + f"_{data_name}"
stats_to_log = []


# Build a run-specific results directory similar to predict_streaming.py
run_results_dir = Path(f"results/{RUN_ID}_batch")
run_results_dir.mkdir(parents=True, exist_ok=True)

# Persist the resolved run configuration into the run-specific results folder so
# the config used for this experiment is stored alongside outputs. This also
# makes `config_file_path` point into the experiment folder.
config_copy_path = run_results_dir / "predict_config.json"
if save_run_config(run_config, config_copy_path):
    config_file_path = config_copy_path
else:
    logger.debug("Could not write run config to %s; continuing without saving.", run_results_dir)

# Stats and predictions live inside the run folder
stats_file_path = run_results_dir / f"{RUN_ID}_stats_batch.json"
predictions_dir = run_results_dir / "predictions"
predictions_dir.mkdir(parents=True, exist_ok=True)

# Log file inside run folder
log_file_path = run_results_dir / f"{RUN_ID}_log.txt"
try:
    # Remove any existing file handlers first
    for handler in logging.root.handlers[:]:
        if isinstance(handler, logging.FileHandler):
            handler.flush()
            handler.close()
            logging.root.removeHandler(handler)

    # Use helper to add a file handler
    file_handler = add_file_log_handler(log_file_path, fmt="%(message)s")
    if file_handler is None:
        logger.debug("Could not create log file %s; continuing with console logging.", log_file_path)
except OSError:
    logger.debug("Could not create log file %s; continuing with console logging.", log_file_path)

if torch.backends.mps.is_available():
    # device = torch.device("mps")
    device = torch.device("cpu")
    logger.info("Using cpu.")
elif torch.cuda.is_available():
    msg = f"Using cuda: {torch.cuda.get_device_name(0)}."
    logger.info(msg)
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
    logger.info("Using cpu.")


torch.manual_seed(123)
torch.cuda.manual_seed(123)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

ML_TRAINING = True
NN_TRAINING = True
ALERGIA_TRAINING = False
SHOW_DELAYS = False
RL_TRAINING = True

# ============================================================
# Determine start and stop symbols
# ============================================================

start_symbol = DEFAULT_CONFIG["start_symbol"]
stop_symbol = DEFAULT_CONFIG["stop_symbol"]

# ============================================================
# Data preparation
# ============================================================

nn_processor = PreprocessData()
data = transform_to_seqs(dataset)
n_activities, max_seq_length = data_statistics(data)

data_test = transform_to_seqs(dataset_test)

# ============================================================
# Define the number of iterations
# ============================================================

N_ITERATIONS = 1

# Store metrics across iterations
all_metrics: dict = {
    name: {
        "accuracies": [],
        "pp_arithmetic_mean": [],
        "pp_harmonic_mean": [],
        "pp_median": [],
        "pp_q1": [],
        "pp_q3": [],
        "top-2": [],
        "top-3": [],
        "num_states": [],
        "train_time": [],
        "pred_time": [],
        "mean_delay_error": [],
        "mean_actual_delay": [],
        "mean_normalized_error": [],
        "num_delay_predictions": [],
    }
    for name in [
        "fpt",
        "bag",
        *list(NGRAM_NAMES),
        "fallback fpt->ngram",
        # "fallback ngram_8->ngram_2",
        # "fallback ngram_8->ngram_3",
        # "fallback ngram_8->ngram_4",
        # "fallback ngram_10->ngram_2",
        # "fallback ngram_13->ngram_2",
        # "fallback ngram_8->...->1",
        # "complex fallback",
        "hard voting",
        # *[
        #     f"adaptive voting {grams} {select_best_arg}"
        #     for select_best_arg in SELECT_BEST_ARGS
        #     for grams in VOTING_NGRAMS
        # ],
        *[f"soft voting {grams}" for grams in VOTING_NGRAMS],
        *[f"soft voting {grams}*" for grams in VOTING_NGRAMS],
        "alergia",
    "LSTM",
    "transformer",
    "transformer_2heads",
    # Add transformer positional-encoding base variants
    *[f"transformer_pos_{pe}" for pe, _ in POS_ENCODINGS],
    "qlearning",
    *[f"LSTM_win{w}" for w in WINDOW_RANGE],
    *[f"transformer_win{w}" for w in WINDOW_RANGE],
    # Add windowed transformer positional-encoding variants
    *[f"transformer_pos_{pe}_win{w}" for pe, _ in POS_ENCODINGS for w in WINDOW_RANGE],
    *[f"qlearning_win{w}" for w in WINDOW_RANGE],
        "bayesian train",
        "bayesian test",
        "bayesian t+t",
        "bayesian test nonsingle",
        "bayesian t+t nonsingle",
    ]
}



# Before the iteration loop: initialize memory for storing prediction vectors per strategy
prediction_vectors_memory: dict = {}
# Add initial debug/info
logger.info(" Initialized prediction_vectors_memory (will store per-strategy prediction vectors)")
logger.debug("Expected iterations: %s", N_ITERATIONS)

# Repeat the experiment N_ITERATIONS times
for iteration in range(N_ITERATIONS):
    # Dictionary to store this iteration's metrics
    iteration_metrics: dict[str, Any] = {}

    # RL models will be trained/evaluated later (after NN preprocessing) to ensure train/val splits are available

    # Store iteration metrics in global metrics
    for name, metrics in iteration_metrics.items():
        all_metrics[name]["accuracies"].append(metrics["acc"])
        all_metrics[name]["top-2"].append(metrics["top-2"])
        all_metrics[name]["top-3"].append(metrics["top-3"])
        all_metrics[name]["pp_arithmetic_mean"].append(compute_perplexity_stats(metrics["perplexities"])["arithmetic_mean"])
        all_metrics[name]["pp_harmonic_mean"].append(compute_perplexity_stats(metrics["perplexities"])["harmonic_mean"])
        all_metrics[name]["pp_median"].append(compute_perplexity_stats(metrics["perplexities"])["median"])
        all_metrics[name]["pp_q1"].append(compute_perplexity_stats(metrics["perplexities"])["q1"])
        all_metrics[name]["pp_q3"].append(compute_perplexity_stats(metrics["perplexities"])["q3"])
        all_metrics[name]["train_time"].append(metrics["train_time"])
    msg = f"Starting iteration {iteration + 1}/{N_ITERATIONS}..."
    logger.info(msg)

    # ============================================================
    # Data Splitting
    # ============================================================

    if data_name != "Synthetic_Train":
        train_set_transformed, remainder = split_sequence_data(data, 0.3, random_shuffle=True, seed=iteration)
        val_set_transformed, test_set_transformed = split_sequence_data(
            remainder, 0.5, random_shuffle=True, seed=iteration
        )
    else:
        # Warning: Synthetic_Train must not be split, but thus the validation set is not correctly generated
        train_set_transformed, val_set_transformed, test_set_transformed = data, data_test, data_test

    # Train set for process miners
    train_set = interleave_sequences(train_set_transformed, random_index=False)

    config = {
        "top_k": 3,
        "include_stop": True,  # Include stop symbol in the training set, recommmended to set to True
    }

    if config["include_stop"]:
        # Append STOP symbol
        train_set_transformed = add_stop_to_sequences(train_set_transformed, stop_symbol)
        val_set_transformed = add_stop_to_sequences(val_set_transformed, stop_symbol)
        test_set_transformed = add_stop_to_sequences(test_set_transformed, stop_symbol)

    alergia_train_set_transformed = add_input_symbols(train_set_transformed, "in")

    TRAIN_EVENTS = sum(len(lst) for lst in train_set_transformed)
    VAL_EVENTS = sum(len(lst) for lst in val_set_transformed)
    TEST_EVENTS = sum(len(lst) for lst in test_set_transformed)


    # ============================================================
    # Initialize Process Miners
    # ============================================================

    # (miners constructed inside build_strategies helper below)

    # Build and store baseline vector of actual next activities (one entry per event, stringified)
    actual_vector: list[str] = []
    for seq in test_set_transformed:
        actual_vector.extend(str(ev["activity"]) for ev in seq)
    prediction_vectors_memory.setdefault("actual", []).append(actual_vector)
    logger.debug(
        "Stored baseline actual vector | iteration=%d | len=%d | sample=%s",
        iteration + 1,
        len(actual_vector),
        actual_vector[:10],
    )

    # Strategies refactored into helper (miners / ngrams / fallback / voting)
    strategies = build_strategies(
        config=config,
        test_set_transformed=test_set_transformed,
        ngram_names=NGRAM_NAMES,
        voting_ngrams=VOTING_NGRAMS,
    )

    # Bayesian classifier models (not part of strategies dict, but trained separately)
    BAYESIAN_MODELS = {
        "bayesian train": BayesianClassifier(config=config),
        "bayesian test": BayesianClassifier(config=config),
        "bayesian t+t": BayesianClassifier(config=config),
        "bayesian test nonsingle": BayesianClassifier(single_occurence_allowed=False, config=config),
        "bayesian t+t nonsingle": BayesianClassifier(single_occurence_allowed=False, config=config),
    }

    # Initialize training times for strategy keys; Bayesian keys will be added during their training loop
    training_times = dict.fromkeys(strategies, 0.0)

    # ================= Train Process Miners
    miners_start_time = time.time()

    for event in tqdm(train_set, desc="Processing events"):
        for strategy_name, (strategy, _) in strategies.items():
            if "alergia" in strategy_name or "bayesian" in strategy_name:
                continue
            start_time = time.time()
            strategy.update(event)
            end_time = time.time()
            training_times[strategy_name] += end_time - start_time

    for strategy_name in strategies:
        if "alergia" in strategy_name or "bayesian" in strategy_name:
            continue
        training_times[strategy_name] /= len(train_set)

    miners_end_time = time.time()
    elapsed_time = miners_end_time - miners_start_time
    msg = f"Total training time for process miners: {elapsed_time:.4f} seconds"
    logger.info(msg)

    # ================= Train Bayesian Classifiers
    bayesian_start_time = time.time()

    for model_name, model in BAYESIAN_MODELS.items():
        if "train" in model_name:
            start_time = time.time()
            model.initialize_memory(train_set_transformed)
            end_time = time.time()
            training_times[model_name] = (end_time - start_time) / TRAIN_EVENTS
        elif "test" in model_name:
            start_time = time.time()
            model.initialize_memory(test_set_transformed)
            end_time = time.time()
            training_times[model_name] = (end_time - start_time) / TEST_EVENTS
        elif "t+t" in model_name:
            start_time = time.time()
            model.initialize_memory(train_set_transformed + test_set_transformed)
            end_time = time.time()
            training_times[model_name] = (end_time - start_time) / (TRAIN_EVENTS + TEST_EVENTS)

    bayesian_end_time = time.time()
    elapsed_time = bayesian_end_time - bayesian_start_time
    msg = f"Training time for Bayesian Classifiers: {elapsed_time:.4f} seconds"
    logger.info(msg)

    for strategy_name in strategies:
        training_times[strategy_name] *= SEC_TO_MICRO  # Convert to microseconds

    # ============================================================
    # Evaluation
    # ============================================================

    # Store the statistics for each iteration and also print them out
    iteration_data: dict = {
        "Model": [],
        "PP Arithm": [],
        "PP Harmo": [],
        "PP Median": [],
        "PP Q1": [],
        "PP Q3": [],
        "Correct (%)": [],
        "Wrong (%)": [],
        "Empty (%)": [],
        "Top-2": [],
        "Top-3": [],
        "Pred Time": [],
        "Train Time": [],
        "Good Preds": [],
        "Tot Preds": [],
        "Nb States": [],
        # "Mean Delay Error": [],
        # "Mean Actual Delay": [],
        # "Mean Normalized Error": [],
        # "Delay Predictions": [],
    }
    # for k in range(1, config["top_k"]):
    #     iteration_data[f"Top-{k+1}"] = []

    for strategy_name, (strategy, test_data) in strategies.items():
        # if "hard" in strategy_name:
        #     continue
        # if "bayesian" in strategy_name:
        #     continue
        # if "voting" in strategy_name or "ngram" in strategy_name:
        #     continue
        # if not strategy_name.startswith("ngram_"):
        #     continue

        msg = f"Evaluating {strategy_name}..."
        logger.info(msg)

        evaluation_time, prediction_vector = strategy.evaluate(
            test_data,
            mode="incremental",
            debug=(data_name == "Synthetic_Train"),
            compute_perplexity=("hard" not in strategy_name and "qlearning" not in strategy_name),
        )
        evaluation_time *= SEC_TO_MICRO / TEST_EVENTS

        # Store prediction vector for this strategy and iteration (keep ordering across iterations)
        prediction_vectors_memory.setdefault(strategy_name, []).append(prediction_vector)

        # Print debugging info about the stored prediction vector
        try:
            sample = prediction_vector[:10]
        except (TypeError, AttributeError, IndexError):
            sample = repr(prediction_vector)[:200]
        logger.debug(
            "Stored preds | strategy=%s | iteration=%d | preds_len=%d | sample=%s",
            strategy_name,
            iteration + 1,
            len(prediction_vector) if hasattr(prediction_vector, "__len__") else -1,
            sample,
        )
        logger.debug("prediction_vector repr (first 400 chars): %s", repr(prediction_vector)[:400])

        stats = strategy.stats

        total = stats["total_predictions"]

        per_state_stats = stats.get("per_state_stats", {})
        # Convert each value in the dictionary (PerStateStats) to a dict
        for key, value in per_state_stats.items():
            per_state_stats[key] = value.to_dict()

        # logging moved into record_model_results to avoid duplication

        num_states = strategy.get_num_states() if isinstance(strategy, BasicMiner) else None

        if "pp_arithmetic_mean" not in stats:
            # Preserve runtime behavior placeholders; explicitly ignore strict type mismatch
            stats["pp_arithmetic_mean"] = None  # type: ignore[assignment]
            stats["pp_harmonic_mean"] = None    # type: ignore[assignment]
            stats["pp_median"] = None           # type: ignore[assignment]
            stats["pp_q1"] = None               # type: ignore[assignment]
            stats["pp_q3"] = None               # type: ignore[assignment]

        # Unified recording for miners (moved after delay/accuracy computations below)

        # (helper functions removed as they were unused)

        # Timing information
        delay_count = stats["num_delay_predictions"]
        if delay_count > 0:
            mean_delay_error = stats["delay_error_sum"] / delay_count
            mean_actual_delay = stats["actual_delay_sum"] / delay_count
            mean_normalized_error = stats["normalized_error_sum"] / delay_count
        else:
            mean_delay_error = None
            mean_actual_delay = None
            mean_normalized_error = None

        # iteration_data["Mean Delay Error"].append(mean_delay_error)
        # iteration_data["Mean Actual Delay"].append(mean_actual_delay)
        # iteration_data["Mean Normalized Error"].append(mean_normalized_error)
        # iteration_data["Delay Predictions"].append(delay_count)

        # Calculate accuracy for helper
        accuracy = stats["correct_predictions"] / total if total > 0 else 0

        # Unified recording for miners
        record_model_results(
            display_name=strategy_name,
            stats={
                "accuracy": accuracy,
                "total_predictions": total,
                "correct_predictions": stats["correct_predictions"],
                "wrong_predictions": stats["wrong_predictions"],
                "empty_predictions": stats["empty_predictions"],
                "top_k_correct_preds": stats["top_k_correct_preds"],
            },
            perplexity_stats={
                "pp_harmonic_mean": stats.get("pp_harmonic_mean"),
                "pp_arithmetic_mean": stats.get("pp_arithmetic_mean"),
                "pp_median": stats.get("pp_median"),
                "pp_q1": stats.get("pp_q1"),
                "pp_q3": stats.get("pp_q3"),
            },
            eval_time_micro=evaluation_time,
            train_time_micro=training_times[strategy_name],
            num_states=num_states,
            top_k=config["top_k"],
            iteration_data=iteration_data,
            all_metrics=all_metrics,
            stats_to_log=stats_to_log,
            mean_delay_error=mean_delay_error,
            mean_actual_delay=mean_actual_delay,
            mean_normalized_error=mean_normalized_error,
            num_delay_predictions=delay_count,
            per_state_stats=per_state_stats,
        )

    # LSTM + Transformer + RL Evaluation
    if ML_TRAINING:
        # For RNNs: Transform sequences with start/stop symbols first
        nn_train_set_transformed = train_set_transformed
        nn_val_set_transformed = val_set_transformed
        nn_test_set_transformed = test_set_transformed

        # Add START symbol for RNN models
        nn_train_set_transformed = add_start_to_sequences(nn_train_set_transformed, start_symbol)
        nn_val_set_transformed = add_start_to_sequences(nn_val_set_transformed, start_symbol)
        nn_test_set_transformed = add_start_to_sequences(nn_test_set_transformed, start_symbol)

        # Preprocess into tensor format
        nn_train_set_transformed = nn_processor.preprocess_data(nn_train_set_transformed)
        nn_val_set_transformed = nn_processor.preprocess_data(nn_val_set_transformed)
        nn_test_set_transformed = nn_processor.preprocess_data(nn_test_set_transformed)

        # RL evaluation must align exactly with the "actual" flattened vector used elsewhere.
        # Basic miners predict one token per event, including the first event of each sequence
        # (conditioned on the initial state). To match that, RL needs a START token so that
        # iterating prefixes k=1..valid_len-1 yields targets covering all original events.
        # Therefore, prepend START to test sequences for RL evaluation only.
        rl_eval_set_with_start = add_start_to_sequences(test_set_transformed, start_symbol)
        rl_eval_set_transformed = nn_processor.preprocess_data(rl_eval_set_with_start)

        if NN_TRAINING:
            for name in ["LSTM", "transformer", "transformer_2heads"]:
                msg = f"Training and evaluating {name} model..."
                logger.info(msg)
                process_neural_model(
                    name=name,
                    iteration_data=iteration_data,
                    all_metrics=all_metrics,
                    nn_train_set_transformed=nn_train_set_transformed,
                    nn_val_set_transformed=nn_val_set_transformed,
                    nn_test_set_transformed=nn_test_set_transformed,
                )
            # Also train/evaluate windowed variants similar to RL qlearning windows
            for w in WINDOW_RANGE:
                for name in ["LSTM", "transformer"]:
                    msg = f"Training and evaluating {name} with window={w}..."
                    logger.info(msg)
                    process_neural_model(
                        name=name,
                        iteration_data=iteration_data,
                        all_metrics=all_metrics,
                        nn_train_set_transformed=nn_train_set_transformed,
                        nn_val_set_transformed=nn_val_set_transformed,
                        nn_test_set_transformed=nn_test_set_transformed,
                        epochs=default_run_config.get("nn", {}).get("epochs", 20),
                        window_size=w,
                    )
            # After NN evaluation in this iteration, print a short summary of NN entries
            for nn_name in ("LSTM", "transformer", "transformer_2heads"):
                vecs = prediction_vectors_memory.get(nn_name, [])
                if vecs:
                    last = vecs[-1]
                    logger.debug(
                        "NN stored preds | model=%s | iter=%d | preds_len=%d | sample=%s",
                        nn_name,
                        iteration + 1,
                        len(last) if hasattr(last, "__len__") else -1,
                        last[:10] if hasattr(last, "__getitem__") else repr(last)[:200],
                    )
                else:
                    logger.debug("NN stored preds | model=%s | iter=%d | (no preds stored)", nn_name, iteration + 1)

        # RL (QNetwork) evaluation in batch mode (no RLMiner). Use process_rl_model to run training/eval
        if RL_TRAINING:
            for w in [None, *WINDOW_RANGE] : # Include both no-window and windowed variants
                rl_name = f"qlearning_win{w}" if w is not None else "qlearning"
                logger.info("Training and evaluating %s model...", rl_name)

                metrics, perplexities, eval_time, prediction_vector, training_time = process_rl_model(
                    rl_name,
                    w,
                    iteration_data,
                    nn_train_set_transformed,
                    nn_val_set_transformed,
                    rl_eval_set_transformed,
                    epochs=default_run_config["rl"]["epochs"],
                )

                # Store prediction vector like other strategies
                prediction_vectors_memory.setdefault(rl_name, []).append(prediction_vector)

                # Assert alignment with baseline actual vector for this iteration
                actual_iters = prediction_vectors_memory.get("actual", [])
                actual_len = len(actual_iters[iteration]) if len(actual_iters) > iteration else None
                if actual_len is not None and len(prediction_vector) != actual_len:
                    msg = (
                        f"Length mismatch for {rl_name} at iter {iteration + 1}: "
                        f"preds={len(prediction_vector)} vs actual={actual_len}."
                    )
                    logger.exception("RL prediction vector length mismatch: %s", msg)
                    raise ValueError(msg)

                # Keep the same placeholders for RL perplexity as before
                perplexity_stats = {
                    "pp_harmonic_mean": None,
                    "pp_arithmetic_mean": None,
                    "pp_median": None,
                    "pp_q1": None,
                    "pp_q3": None,
                }

                eval_time *= SEC_TO_MICRO / TEST_EVENTS

                if (
                    not isinstance(metrics["top_k_correct_preds"], list)
                    or not isinstance(metrics["total_predictions"], int)
                    or not isinstance(metrics["accuracy"], float)
                ):
                    msg = f"{rl_name} stats are not in the expected format."
                    raise TypeError(msg)

                record_model_results(
                    display_name=rl_name,
                    stats={
                        "accuracy": metrics["accuracy"],
                        "total_predictions": metrics["total_predictions"],
                        "correct_predictions": metrics["correct_predictions"],
                        "top_k_correct_preds": metrics["top_k_correct_preds"],
                    },
                    perplexity_stats=perplexity_stats,
                    eval_time_micro=eval_time,
                    train_time_micro=training_time,
                    num_states=None,
                    top_k=config["top_k"],
                    iteration_data=iteration_data,
                    all_metrics=all_metrics,
                    stats_to_log=stats_to_log,
                )

    # Create a DataFrame for the iteration and log it
    iteration_df = pd.DataFrame(iteration_data).round(2)
    # Iteration-level summary of stored prediction vectors
    summary_counts = {k: len(v) for k, v in prediction_vectors_memory.items()}
    logger.info("After iteration %d summary counts (per strategy): %s", iteration + 1, summary_counts)
    # Show small sample for a few strategies to verify alignment
    sample_strategies = list(iteration_data["Model"])[:5]
    for s in sample_strategies:
        vecs = prediction_vectors_memory.get(s, [])
        last = vecs[-1] if vecs else []
        # logger.info("[ITER DEBUG] sample preds | strategy=%s | len=%d | sample=%s", s, len(last), (last[:8] if hasattr(last, "__getitem__") else repr(last)[:200])) # noqa: E501
    logger.info("Iteration %d results:\n%s", iteration + 1, iteration_df)

# ============================================================
# Calculate and Show Final Results
# ============================================================


with stats_file_path.open("w") as f:
    json.dump(stats_to_log, f, indent=4)

# Write prediction vectors (one CSV per strategy and per-iteration) into the run-specific predictions dir
write_prediction_vectors(
    prediction_vectors_memory=prediction_vectors_memory,
    run_id=RUN_ID,
    predictions_dir=predictions_dir,
    logger=logger,
)

results: dict = {
    "Model": [],
    "Mean Accuracy (%)": [],
    "Std": [],
    "Top-2 (%)": [],
    "Top-3 (%)": [],
    "PP Arithm": [],
    "PP Harmo": [],
    "PP Median": [],
    "PP Q1": [],
    "PP Q3": [],
    "States": [],
    "Pred Time": [],
    "Train Time": [],
    "Delay Error": [],
    "Actual Delay": [],
    "Normalized Error": [],
    "Delay Predictions": [],
}

for model_name, stats in all_metrics.items():
    results["Model"].append(model_name)

    key_labels = {
        "PP Arithm": "pp_arithmetic_mean",
        "PP Harmo": "pp_harmonic_mean",
        "PP Median": "pp_median",
        "PP Q1": "pp_q1",
        "PP Q3": "pp_q3",
        "Top-2 (%)": "top-2",
        "Top-3 (%)": "top-3",
        "Pred Time": "pred_time",
        "Train Time": "train_time",
    }

    for label, key_name in key_labels.items():
        if len(stats[key_name]) > 0 and None not in stats[key_name]:
            results[label].append(np.mean(stats[key_name]))
        else:
            results[label].append(None)

    if len(stats["accuracies"]) > 0:
        mean_acc = np.mean(stats["accuracies"]) * 100
        std_acc = float(np.std(stats["accuracies"])) * 100
    else:
        mean_acc = None
        std_acc = None

    results["Mean Accuracy (%)"].append(mean_acc)
    results["Std"].append(std_acc)

    if len(stats["num_states"]) > 0 and None not in stats["num_states"]:
        final_num_states = stats["num_states"][-1]
    else:
        final_num_states = None

    results["States"].append(final_num_states)

    if len(stats["mean_delay_error"]) > 0 and None not in stats["mean_delay_error"]:
        mean_delay_error = timedelta(seconds=float(np.mean(stats["mean_delay_error"])))
    else:
        mean_delay_error = None

    if len(stats["mean_actual_delay"]) > 0 and None not in stats["mean_actual_delay"]:
        mean_actual_delay = timedelta(seconds=float(np.mean(stats["mean_actual_delay"])))
    else:
        mean_actual_delay = None

    if len(stats["mean_normalized_error"]) > 0 and None not in stats["mean_normalized_error"]:
        mean_normalized_error = np.mean(stats["mean_normalized_error"])
    else:
        mean_normalized_error = None

    if len(stats["num_delay_predictions"]) > 0 and None not in stats["num_delay_predictions"]:
        num_delay_predictions = np.mean(stats["num_delay_predictions"])
    else:
        num_delay_predictions = None

    def format_timedelta(td: timedelta | None) -> str | None:
        """Format a timedelta object into a string representation."""
        if td is None:
            return None
        days = td.days
        hours, rem = divmod(td.seconds, 3600)
        minutes, seconds = divmod(rem, 60)
        return f"{days}d {hours:02d}h {minutes:02d}m {seconds:02d}s"

    # Assuming mean_delay_error and mean_actual_delay are timedelta objects
    results["Delay Error"].append(format_timedelta(mean_delay_error))
    results["Actual Delay"].append(format_timedelta(mean_actual_delay))

    # Normalized Error and Delay Predictions (assuming they are not timedelta)
    results["Normalized Error"].append(mean_normalized_error)  # Round to two decimals
    results["Delay Predictions"].append(num_delay_predictions)  # Keep as-is

# Create a DataFrame and print it
data = pd.DataFrame(results).round(2)

if not SHOW_DELAYS:
    # Remove the delay columns
    data = data.drop(columns=["Delay Error", "Actual Delay", "Normalized Error", "Delay Predictions"])
msg = "\n" + str(data)
logger.info(msg)

# === Cross-reference table of comparison ratios between all models ===
build_and_save_comparison_matrices(
    prediction_vectors_memory=prediction_vectors_memory,
    run_id=RUN_ID,
    out_dir=stats_file_path.parent,
    logger=logger,
)
