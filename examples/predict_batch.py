"""Module to evaluate and compare different process mining models."""

import json
import logging
import time
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from typing import Any

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from aalpy.learning_algs import run_Alergia
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
    Bag,
    BayesianClassifier,
    FrequencyPrefixTree,
    NGram,
    Parikh,
)
from logicsponge.processmining.config import DEFAULT_CONFIG
from logicsponge.processmining.data_utils import (
    add_input_symbols,
    add_start_to_sequences,
    add_stop_to_sequences,
    data_statistics,
    interleave_sequences,
    retain_sequences_of_length_x_than,
    split_sequence_data,
    transform_to_seqs,
)
from logicsponge.processmining.models import (
    AdaptiveVoting,
    Alergia,
    BasicMiner,
    Fallback,
    HardVoting,
    Relativize,
    SoftVoting,
)
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
from logicsponge.processmining.utils import RED_TO_GREEN_CMAP, compare_models_comparison, compute_perplexity_stats

SEC_TO_MICRO = 1_000_000


def lstm_model() -> tuple[LSTMModel, optim.Optimizer, nn.Module]:
    """Initialize and return an LSTM model, optimizer, and loss function."""
    model = LSTMModel(
        vocab_size=run_config.get("lstm", {}).get("vocab_size", 64),
        embedding_dim=run_config.get("lstm", {}).get("embedding_dim", 64),
        hidden_dim=run_config.get("lstm", {}).get("hidden_dim", 128),
        output_dim=run_config.get("lstm", {}).get("output_dim", 64),
        use_one_hot=True,
        device=device
    )
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=run_config.get("nn", {}).get("lr", 0.001))
    return model, optimizer, criterion


def transformer_model() -> tuple[TransformerModel, optim.Optimizer, nn.Module]:
    """Initialize and return a Transformer model, optimizer, and loss function."""
    model = TransformerModel(
        seq_input_dim=max_seq_length + 2,
        vocab_size=run_config.get("transformer", {}).get("vocab_size", 64),
        embedding_dim=run_config.get("transformer", {}).get("embedding_dim", 64),
        hidden_dim=run_config.get("transformer", {}).get("hidden_dim", 128),
        output_dim=run_config.get("transformer", {}).get("output_dim", 64),
        use_one_hot=True,
        device=device,
    )  # +2 for start and stop symbols
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=run_config.get("nn", {}).get("lr", 0.001))
    return model, optimizer, criterion


def process_neural_model(  # noqa: C901, PLR0912, PLR0913, PLR0915
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

    match name:
        case "LSTM":
            model, optimizer, criterion = lstm_model()
        case "transformer":
            model, optimizer, criterion = transformer_model()
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
        # Local prefix-mode evaluation for RNN/Transformer: iterate prefixes and
        # make a single prediction per prefix (cropping the input to the last
        # `window_size` tokens before passing to the model). This mirrors
        # evaluate_rl behavior and ensures alignment with `actual`.
        eval_start = time.time()
        correct = 0
        total = 0
        top_k_correct = [0] * config["top_k"]
        prediction_vector = []

        model.eval()
        model_device = getattr(model, "device", None)

        with torch.no_grad():
            for i in range(nn_test_set_transformed.shape[0]):
                seq = nn_test_set_transformed[i]
                valid_len = int((seq != 0).sum().item())
                # Skip sequences too short to have a prefix
                if valid_len < 2:  # noqa: PLR2004
                    continue
                # iterate prefixes (k = 1..valid_len-1) to produce one pred per event
                for k in range(1, valid_len):
                    prefix = seq[:k].unsqueeze(0)  # [1, k]
                    if prefix.shape[1] > window_size:
                        prefix = prefix[:, -window_size:]
                    if model_device is not None:
                        prefix = prefix.to(device=model_device)

                    outputs = model(prefix)
                    # outputs shape: [batch=1, seq_len=maybe<=window_size, vocab]
                    # We take the last timestep's logits as the prediction for the next token
                    last_logits = outputs[:, -1, :].squeeze(0) if outputs.dim() == 3 else outputs.squeeze(0)  # noqa: PLR2004

                    topk = torch.topk(last_logits, k=config["top_k"])
                    pred_idx = int(topk.indices[0].item())

                    # Map to activity names
                    if nn_processor.idx_to_activity and pred_idx in nn_processor.idx_to_activity:
                        prediction_vector.append(str(nn_processor.idx_to_activity[pred_idx]))
                    else:
                        prediction_vector.append(str(pred_idx))

                    target_idx = int(seq[k].item())
                    total += 1
                    if pred_idx == target_idx:
                        correct += 1
                        for j in range(config["top_k"]):
                            top_k_correct[j] += 1
                    else:
                        # count inclusion in top-k
                        for j in range(config["top_k"]):
                            if target_idx in {int(x) for x in topk.indices[: j + 1].tolist()}:
                                top_k_correct[j] += 1

        eval_time = time.time() - eval_start
        stats = {
            "accuracy": (correct / total) if total > 0 else 0.0,
            "total_predictions": total,
            "correct_predictions": correct,
            "top_k_correct_preds": top_k_correct,
        }
        perplexities = []

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

    # Append data to the iteration data dictionary
    iteration_data["Model"].append(display_name)

    iteration_data["PP Harmo"].append(perplexity_stats["pp_harmonic_mean"])
    iteration_data["PP Arithm"].append(perplexity_stats["pp_arithmetic_mean"])
    iteration_data["PP Median"].append(perplexity_stats["pp_median"])
    iteration_data["PP Q1"].append(perplexity_stats["pp_q1"])
    iteration_data["PP Q3"].append(perplexity_stats["pp_q3"])

    iteration_data["Correct (%)"].append(stats["accuracy"] * 100)
    iteration_data["Wrong (%)"].append(100 - stats["accuracy"] * 100)
    iteration_data["Empty (%)"].append(0.0)

    for k in range(1, config["top_k"]):
        iteration_data[f"Top-{k + 1}"].append(stats["top_k_correct_preds"][k] / stats["total_predictions"] * 100)

    iteration_data["Pred Time"].append(eval_time)
    iteration_data["Train Time"].append(training_time)

    iteration_data["Good Preds"].append(stats["correct_predictions"])
    iteration_data["Tot Preds"].append(stats["total_predictions"])
    iteration_data["Nb States"].append(None)

    stats_to_log.append(
        {
            "strategy": display_name,
            "strategy_accuracy": stats["accuracy"] * 100,
            "strategy_perplexity": perplexity_stats["pp_harmonic_mean"],
            "strategy_eval_time": eval_time,
        }
    )

    # Ensure the all_metrics dict has an entry for display_name
    if display_name not in all_metrics:
        all_metrics[display_name] = {
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

    all_metrics[display_name]["accuracies"].append(stats["accuracy"])
    all_metrics[display_name]["pp_arithmetic_mean"].append(perplexity_stats["pp_arithmetic_mean"])
    all_metrics[display_name]["pp_harmonic_mean"].append(perplexity_stats["pp_harmonic_mean"])
    all_metrics[display_name]["pp_median"].append(perplexity_stats["pp_median"])
    all_metrics[display_name]["pp_q1"].append(perplexity_stats["pp_q1"])
    all_metrics[display_name]["pp_q3"].append(perplexity_stats["pp_q3"])
    for k in range(1, config["top_k"]):
        all_metrics[display_name][f"top-{k + 1}"].append(iteration_data[f"Top-{k + 1}"][-1])

    all_metrics[display_name]["pred_time"].append(eval_time)
    all_metrics[display_name]["train_time"].append(training_time)

    all_metrics[display_name]["num_states"].append(0)
    all_metrics[display_name]["mean_delay_error"].append(None)
    all_metrics[display_name]["mean_actual_delay"].append(None)
    all_metrics[display_name]["mean_normalized_error"].append(None)
    all_metrics[display_name]["num_delay_predictions"].append(None)


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


def qnetwork_model() -> tuple[QNetwork, optim.Optimizer, nn.Module]:
    """Initialize a QNetwork model for RL training."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = QNetwork(
        vocab_size=run_config.get("qlearning", {}).get("vocab_size", 32),
        embedding_dim=run_config.get("qlearning", {}).get("embedding_dim", 32),
        hidden_dim=run_config.get("qlearning", {}).get("hidden_dim", 1024),
        output_dim=run_config.get("qlearning", {}).get("output_dim", 32),
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
run_results_dir = Path(f"results/{RUN_ID}")
run_results_dir.mkdir(parents=True, exist_ok=True)

# Persist the resolved run configuration into the run-specific results folder so
# the config used for this experiment is stored alongside outputs. This also
# makes `config_file_path` point into the experiment folder.
try:
    config_copy_path = run_results_dir / "predict_config.json"
    with config_copy_path.open("w") as _f:
        json.dump(run_config, _f, indent=2)
    config_file_path = config_copy_path
except OSError:
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

    # Reconfigure the logger to write to the run-specific file
    file_handler = logging.FileHandler(log_file_path)
    formatter = logging.Formatter("%(message)s")
    file_handler.setFormatter(formatter)
    logging.root.addHandler(file_handler)
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
        "qlearning",
        *[f"LSTM_win{w}" for w in WINDOW_RANGE],
        *[f"transformer_win{w}" for w in WINDOW_RANGE],
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
        # ML_TRAINING = False

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

    fpt = BasicMiner(algorithm=FrequencyPrefixTree(), config=config)

    bag = BasicMiner(algorithm=Bag(), config=config)

    parikh = BasicMiner(algorithm=Parikh(upper_bound=2), config=config)

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

    # NGram models
    NGRAM_MODELS: dict[str, BasicMiner] = {}
    for ngram_name in NGRAM_NAMES:
        window_length = int(ngram_name.split("_")[1]) - 1
        recovery_lengths = list(range(window_length, -1, -1))

        if "recovery" in ngram_name or "shorts" in ngram_name:
            NGRAM_MODELS[ngram_name] = BasicMiner(
                algorithm=NGram(
                    window_length=window_length,
                    recover_lengths=recovery_lengths,
                ),
                config=config,
            )
        else:
            # Use the default NGram algorithm without recovery
            NGRAM_MODELS[ngram_name] = BasicMiner(
                algorithm=NGram(window_length=window_length, recover_lengths=[]),
                config=config,
            )
        # logger.debug(f"Stats of {ngram_name}: {NGRAM_MODELS[ngram_name].stats}")

    # ngram_8_test = BasicMiner(algorithm=NGram(window_length=7), config=config)
    # ngram_8_test_train = BasicMiner(algorithm=NGram(window_length=7), config=config)

    fallback = Fallback(
        models=[
            BasicMiner(algorithm=FrequencyPrefixTree(min_total_visits=10)),
            BasicMiner(algorithm=NGram(window_length=4)),
        ],
        config=config,
    )

    fallback_ngram8to2 = Fallback(
        models=[
            BasicMiner(algorithm=NGram(window_length=7)),
            BasicMiner(algorithm=NGram(window_length=1)),
        ],
        config=config,
    )

    fallback_ngram8to3 = Fallback(
        models=[
            BasicMiner(algorithm=NGram(window_length=7)),
            BasicMiner(algorithm=NGram(window_length=2)),
        ],
        config=config,
    )

    fallback_ngram8to4 = Fallback(
        models=[
            BasicMiner(algorithm=NGram(window_length=7)),
            BasicMiner(algorithm=NGram(window_length=3)),
        ],
        config=config,
    )

    fallback_ngram10to2 = Fallback(
        models=[
            BasicMiner(algorithm=NGram(window_length=9)),
            BasicMiner(algorithm=NGram(window_length=1)),
        ],
        config=config,
    )

    fallback_ngram13to2 = Fallback(
        models=[
            BasicMiner(algorithm=NGram(window_length=12)),
            BasicMiner(algorithm=NGram(window_length=1)),
        ],
        config=config,
    )

    fallback_ngram8to_ooo = Fallback(
        models=[
            BasicMiner(algorithm=NGram(window_length=7, min_total_visits=10)),
            BasicMiner(algorithm=NGram(window_length=6, min_total_visits=10)),
            BasicMiner(algorithm=NGram(window_length=5)),
            BasicMiner(algorithm=NGram(window_length=4)),
            BasicMiner(algorithm=NGram(window_length=3)),
            BasicMiner(algorithm=NGram(window_length=2)),
            BasicMiner(algorithm=NGram(window_length=1)),
            BasicMiner(algorithm=NGram(window_length=0)),
        ],
        config=config,
    )

    complex_fallback = Fallback(
        models=[
            BasicMiner(algorithm=NGram(window_length=9, min_total_visits=10, min_max_prob=0.9)),
            BasicMiner(algorithm=NGram(window_length=8, min_total_visits=10, min_max_prob=0.9)),
            BasicMiner(algorithm=NGram(window_length=7, min_total_visits=10, min_max_prob=0.8)),
            BasicMiner(algorithm=NGram(window_length=6, min_total_visits=10, min_max_prob=0.7)),
            BasicMiner(algorithm=NGram(window_length=5, min_total_visits=10, min_max_prob=0.6)),
            BasicMiner(algorithm=NGram(window_length=4, min_total_visits=10, min_max_prob=0.0)),
            BasicMiner(
                algorithm=NGram(window_length=3, min_total_visits=10, min_max_prob=0.0),
            ),
            BasicMiner(algorithm=NGram(window_length=2, min_total_visits=10, min_max_prob=0.0)),
            BasicMiner(algorithm=NGram(window_length=1)),
        ],
        config=config,
    )

    hard_voting = HardVoting(
        models=[
            BasicMiner(algorithm=Bag()),
            BasicMiner(algorithm=FrequencyPrefixTree(min_total_visits=10)),
            BasicMiner(algorithm=NGram(window_length=2)),
            BasicMiner(algorithm=NGram(window_length=3)),
            BasicMiner(algorithm=NGram(window_length=4)),
            # BasicMiner(algorithm=NGram(window_length=5)),
            # BasicMiner(algorithm=NGram(window_length=6)),
        ],
        config=config,
    )

    optional_num_ngrams = 3
    adaptive_voting_list = [
        AdaptiveVoting(
            models=[
                BasicMiner(algorithm=Bag()),
                BasicMiner(algorithm=FrequencyPrefixTree(min_total_visits=10)),
                BasicMiner(algorithm=NGram(window_length=grams[0], min_total_visits=10)),
                BasicMiner(algorithm=NGram(window_length=grams[1], min_total_visits=10)),
                BasicMiner(algorithm=NGram(window_length=grams[2], min_total_visits=10)),
            ]
            + (
                [BasicMiner(algorithm=NGram(window_length=grams[optional_num_ngrams], min_total_visits=10))]
                if len(grams) > optional_num_ngrams
                else []
            ),
            select_best=select_best_arg,
            config=config,
        )
        for grams in VOTING_NGRAMS
        for select_best_arg in SELECT_BEST_ARGS
    ]

    # soft_voting = SoftVoting(
    #     models=[
    #         BasicMiner(algorithm=Bag()),
    #         BasicMiner(algorithm=FrequencyPrefixTree(min_total_visits=10)),
    #         BasicMiner(algorithm=NGram(window_length=2)),
    #         BasicMiner(algorithm=NGram(window_length=3)),
    #         BasicMiner(algorithm=NGram(window_length=4)),
    #         # BasicMiner(algorithm=NGram(window_length=5)),
    #         # BasicMiner(algorithm=NGram(window_length=6)),
    #     ],
    #     config=config,
    # )

    soft_voting_list = [
        SoftVoting(
            models=[
                BasicMiner(algorithm=Bag()),
                BasicMiner(algorithm=FrequencyPrefixTree(min_total_visits=10)),
                BasicMiner(algorithm=NGram(window_length=grams[0])),
                BasicMiner(algorithm=NGram(window_length=grams[1])),
                BasicMiner(algorithm=NGram(window_length=grams[2])),
            ]
            + (
                [BasicMiner(algorithm=NGram(window_length=grams[optional_num_ngrams]))]
                if len(grams) > optional_num_ngrams
                else []
            ),
            config=config,
        )
        for grams in VOTING_NGRAMS
    ] + [
        SoftVoting(
            models=[
                BasicMiner(algorithm=Bag()),
                BasicMiner(algorithm=FrequencyPrefixTree(min_total_visits=10)),
                BasicMiner(algorithm=NGram(window_length=grams[0], min_total_visits=10)),
                BasicMiner(algorithm=NGram(window_length=grams[1], min_total_visits=10)),
                BasicMiner(algorithm=NGram(window_length=grams[2], min_total_visits=10)),
            ]
            + (
                [BasicMiner(algorithm=NGram(window_length=grams[optional_num_ngrams], min_total_visits=10))]
                if len(grams) > optional_num_ngrams
                else []
            ),
        )
        for grams in VOTING_NGRAMS
    ]

    relativize = Relativize(
        models=[
            BasicMiner(algorithm=FrequencyPrefixTree(min_total_visits=10)),
            BasicMiner(algorithm=NGram(window_length=3)),
        ],
        config=config,
    )

    # ============================================================
    BAYESIAN_MODELS = {
        "bayesian train": BayesianClassifier(config=config),
        "bayesian test": BayesianClassifier(config=config),
        "bayesian t+t": BayesianClassifier(config=config),
        "bayesian test nonsingle": BayesianClassifier(single_occurence_allowed=False, config=config),
        "bayesian t+t nonsingle": BayesianClassifier(single_occurence_allowed=False, config=config),
    }

    # All strategies (without LSTM)
    ngram_strategies = {
        ngram_name: (
            ngram_model,
            test_set_transformed
            if "shorts" not in ngram_name
            else retain_sequences_of_length_x_than(test_set_transformed, 10, mode="lower"),
        )
        for ngram_name, ngram_model in NGRAM_MODELS.items()
    }
    # Repeat each tuple in VOTING_NGRAMS 3 times for adaptive voting strategies
    repeated_ngrams = [gram for grams in VOTING_NGRAMS for gram in [grams] * len(SELECT_BEST_ARGS)]
    adaptive_voting_strategies = {
        f"adaptive voting {grams} {adaptive_voting.select_best}": (adaptive_voting, test_set_transformed)
        for grams, adaptive_voting in zip(repeated_ngrams, adaptive_voting_list, strict=False)
    }

    soft_voting_strategies1 = {
        f"soft voting {grams}": (soft_voting_test, test_set_transformed)
        for grams, soft_voting_test in zip(VOTING_NGRAMS, soft_voting_list[: len(VOTING_NGRAMS)], strict=False)
    }
    soft_voting_strategies2 = {
        f"soft voting {grams}*": (soft_voting_test, test_set_transformed)
        for grams, soft_voting_test in zip(VOTING_NGRAMS, soft_voting_list[len(VOTING_NGRAMS) :], strict=False)
    }
    soft_voting_strategies = {**soft_voting_strategies1, **soft_voting_strategies2}
    bayesian_strategies = {} #{model_name: (model, test_set_transformed) for model_name, model in BAYESIAN_MODELS.items()} # noqa: E501
    strategies = {
        "fpt": (fpt, test_set_transformed),
        "bag": (bag, test_set_transformed),
        **ngram_strategies,
        # "ngram_12": (ngram_12, retain_sequences_of_length_x_than(test_set_transformed, 10, mode="lower")),
        # "ngram_15": (ngram_15, retain_sequences_of_length_x_than(test_set_transformed, 10, mode="lower")),
        # "ngram_18": (ngram_18, retain_sequences_of_length_x_than(test_set_transformed, 10, mode="lower")),
        "fallback fpt->ngram": (fallback, test_set_transformed),
        # "fallback ngram_8->ngram_2": (fallback_ngram8to2, test_set_transformed),
        # "fallback ngram_8->ngram_3": (fallback_ngram8to3, test_set_transformed),
        # "fallback ngram_8->ngram_4": (fallback_ngram8to4, test_set_transformed),
        # "fallback ngram_10->ngram_2": (fallback_ngram10to2, test_set_transformed),
        # "fallback ngram_13->ngram_2": (fallback_ngram13to2, test_set_transformed),
        # "fallback ngram_8->...->1": (fallback_ngram8to_ooo, test_set_transformed),
        # "complex fallback": (complex_fallback, test_set_transformed),
        "hard voting": (hard_voting, test_set_transformed),
        # **adaptive_voting_strategies,
        **soft_voting_strategies,
        **bayesian_strategies,
    }

    training_times = dict.fromkeys(strategies, 0.0)

    # ============= Train Alergia
    if ALERGIA_TRAINING:
        alergia_start_time = time.time()

        algorithm = run_Alergia(alergia_train_set_transformed, automaton_type="smm", eps=0.5, print_info=True)
        smm = Alergia(algorithm=algorithm)
        strategies["alergia"] = (smm, test_set_transformed)

        alergia_end_time = time.time()
        alergia_elapsed_time = alergia_end_time - alergia_start_time
        msg = f"Training time for Alergia: {alergia_elapsed_time:.4f} seconds"
        logger.info(msg)
        training_times = dict.fromkeys(strategies, 0.0)
        training_times["alergia"] = alergia_elapsed_time


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
        correct_percentage = (stats["correct_predictions"] / total * 100) if total > 0 else 0
        wrong_percentage = (stats["wrong_predictions"] / total * 100) if total > 0 else 0
        empty_percentage = (stats["empty_predictions"] / total * 100) if total > 0 else 0

        top_k_accuracies = (
            [(top_k_correct / total * 100) for top_k_correct in stats["top_k_correct_preds"]]
            if total > 0
            else [0] * len(stats["top_k_correct_preds"])
        )

        per_state_stats = stats.get("per_state_stats", {})
        # Convert each value in the dictionary (PerStateStats) to a dict
        for key, value in per_state_stats.items():
            per_state_stats[key] = value.to_dict()

        stats_to_log.append(
            {
                "strategy": strategy_name,
                "strategy_accuracy": correct_percentage,
                "strategy_perplexity": stats["pp_harmonic_mean"],
                "strategy_eval_time": evaluation_time,
                "per_state_stats": per_state_stats,
            }
        )

        num_states = strategy.get_num_states() if isinstance(strategy, BasicMiner) else None

        if "pp_arithmetic_mean" not in stats:
            stats["pp_arithmetic_mean"] = None
            stats["pp_harmonic_mean"] = None
            stats["pp_median"] = None
            stats["pp_q1"] = None
            stats["pp_q3"] = None

        # Append data to the iteration data dictionary
        iteration_data["Model"].append(strategy_name)

        iteration_data["PP Harmo"].append(stats["pp_harmonic_mean"])
        iteration_data["PP Arithm"].append(stats["pp_arithmetic_mean"])
        iteration_data["PP Median"].append(stats["pp_median"])
        iteration_data["PP Q1"].append(stats["pp_q1"])
        iteration_data["PP Q3"].append(stats["pp_q3"])

        iteration_data["Correct (%)"].append(correct_percentage)
        iteration_data["Wrong (%)"].append(wrong_percentage)
        iteration_data["Empty (%)"].append(empty_percentage)
        for k in range(1, config["top_k"]):
            iteration_data[f"Top-{k + 1}"].append(top_k_accuracies[k])

        iteration_data["Pred Time"].append(evaluation_time)
        iteration_data["Train Time"].append(training_times[strategy_name])

        iteration_data["Good Preds"].append(stats["correct_predictions"])
        iteration_data["Tot Preds"].append(total)
        iteration_data["Nb States"].append(num_states)

        # Get the mean of a dictionary
        def weighted_mean_of_dict(stat_dict: dict) -> float:
            """Calculate the mean of a dictionary where keys are the values and values are the weights."""
            weighted_sum = sum(key * val for key, val in stat_dict.items())
            total_val = sum(stat_dict.values())
            return weighted_sum / total_val if total_val != 0 else 0

        def mean_of_dict(my_dict: dict) -> float:
            """Calculate the mean of a dictionary."""
            return sum(my_dict.values()) / len(my_dict) if my_dict else 0

        def div_dict(stat_dict: dict, total_dict: dict) -> dict:
            """Divide each value in stat_dict by the corresponding value in total_dict."""
            output = {}
            for key, value in stat_dict.items():
                total = total_dict[key]
                if total == 0:
                    msg = f"Total for key {key} is zero, cannot calculate mean."
                    raise ValueError(msg)
                output[key] = value / total
            return output

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

        # Calculate and append accuracy to all_metrics for final statistics
        accuracy = stats["correct_predictions"] / total if total > 0 else 0

        all_metrics[strategy_name]["accuracies"].append(accuracy)
        all_metrics[strategy_name]["pp_arithmetic_mean"].append(stats["pp_arithmetic_mean"])
        all_metrics[strategy_name]["pp_harmonic_mean"].append(stats["pp_harmonic_mean"])
        all_metrics[strategy_name]["pp_median"].append(stats["pp_median"])
        all_metrics[strategy_name]["pp_q1"].append(stats["pp_q1"])
        all_metrics[strategy_name]["pp_q3"].append(stats["pp_q3"])
        for k in range(1, config["top_k"]):
            all_metrics[strategy_name][f"top-{k + 1}"].append(top_k_accuracies[k])
        all_metrics[strategy_name]["num_states"].append(num_states)
        all_metrics[strategy_name]["pred_time"].append(evaluation_time)
        all_metrics[strategy_name]["train_time"].append(training_times[strategy_name])

        all_metrics[strategy_name]["mean_delay_error"].append(mean_delay_error)
        all_metrics[strategy_name]["mean_actual_delay"].append(mean_actual_delay)
        all_metrics[strategy_name]["mean_normalized_error"].append(mean_normalized_error)
        all_metrics[strategy_name]["num_delay_predictions"].append(delay_count)

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
            for name in ["LSTM", "transformer"]:
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
            for nn_name in ("LSTM", "transformer"):
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

        logger.info("Training and evaluating qlearning model...")
        process_rl_model(
            "qlearning",
            None,
            iteration_data,
            nn_train_set_transformed,
            nn_val_set_transformed,
            nn_test_set_transformed,
            epochs=default_run_config["rl"]["epochs"],
        )

        # RL (QNetwork) evaluation in batch mode (no RLMiner). Use process_rl_model to run training/eval
        for w in WINDOW_RANGE:
            rl_name = f"qlearning_win{w}"
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

            iteration_data["Model"].append(rl_name)

            iteration_data["PP Harmo"].append(perplexity_stats["pp_harmonic_mean"])
            iteration_data["PP Arithm"].append(perplexity_stats["pp_arithmetic_mean"])
            iteration_data["PP Median"].append(perplexity_stats["pp_median"])
            iteration_data["PP Q1"].append(perplexity_stats["pp_q1"])
            iteration_data["PP Q3"].append(perplexity_stats["pp_q3"])

            iteration_data["Correct (%)"].append(metrics["accuracy"] * 100)
            iteration_data["Wrong (%)"].append(100 - metrics["accuracy"] * 100)
            iteration_data["Empty (%)"].append(0.0)

            for k in range(1, config["top_k"]):
                iteration_data[f"Top-{k + 1}"].append(
                    metrics["top_k_correct_preds"][k] / metrics["total_predictions"] * 100
                )

            iteration_data["Pred Time"].append(eval_time)
            iteration_data["Train Time"].append(training_time)

            iteration_data["Good Preds"].append(metrics["correct_predictions"])
            iteration_data["Tot Preds"].append(metrics["total_predictions"])
            iteration_data["Nb States"].append(None)

            stats_to_log.append(
                {
                    "strategy": rl_name,
                    "strategy_accuracy": metrics["accuracy"] * 100,
                    "strategy_perplexity": perplexity_stats["pp_harmonic_mean"],
                    "strategy_eval_time": eval_time,
                }
            )

            all_metrics[rl_name]["accuracies"].append(metrics["accuracy"])
            all_metrics[rl_name]["pp_arithmetic_mean"].append(perplexity_stats["pp_arithmetic_mean"])
            all_metrics[rl_name]["pp_harmonic_mean"].append(perplexity_stats["pp_harmonic_mean"])
            all_metrics[rl_name]["pp_median"].append(perplexity_stats["pp_median"])
            all_metrics[rl_name]["pp_q1"].append(perplexity_stats["pp_q1"])
            all_metrics[rl_name]["pp_q3"].append(perplexity_stats["pp_q3"])
            for k in range(1, config["top_k"]):
                all_metrics[rl_name][f"top-{k + 1}"].append(iteration_data[f"Top-{k + 1}"][-1])

            all_metrics[rl_name]["pred_time"].append(eval_time)
            all_metrics[rl_name]["train_time"].append(training_time)

            all_metrics[rl_name]["num_states"].append(0)
            all_metrics[rl_name]["mean_delay_error"].append(None)
            all_metrics[rl_name]["mean_actual_delay"].append(None)
            all_metrics[rl_name]["mean_normalized_error"].append(None)
            all_metrics[rl_name]["num_delay_predictions"].append(None)

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
try:
    for model_name, iterations in prediction_vectors_memory.items():
        # Combined dataframe for all iterations for this model
        combined_rows = []
        for it_idx, pred_vec in enumerate(iterations, start=1):
            # pred_vec is expected to be an iterable of predictions (one per event)
            # Try to fetch the matching actual vector for this iteration (if available)
            actual_iters = prediction_vectors_memory.get("actual", [])
            actual_vec = None
            if model_name != "actual" and len(actual_iters) >= it_idx:
                actual_vec = actual_iters[it_idx - 1]

            try:
                preds = list(pred_vec)

                if actual_vec is not None:
                    # Align lengths: truncate to the shortest to avoid misalignment
                    actuals = list(actual_vec)
                    min_len = min(len(preds), len(actuals))
                    preds = preds[:min_len]
                    actuals = actuals[:min_len]
                    df_iter = pd.DataFrame({"index": list(range(len(preds))), "prediction": preds, "actual": actuals})
                else:
                    df_iter = pd.DataFrame({"index": list(range(len(preds))), "prediction": preds})
            except (ValueError, TypeError, OSError):
                # Fallback: coerce into a single-column representation; include actual as string if available
                if actual_vec is not None:
                    try:
                        actuals = list(actual_vec)
                        df_iter = pd.DataFrame({"prediction": [str(pred_vec)], "actual": [str(actuals)]})
                    except (ValueError, TypeError, OSError):
                        df_iter = pd.DataFrame({"prediction": [str(pred_vec)]})
                else:
                    df_iter = pd.DataFrame({"prediction": [str(pred_vec)]})

            iter_csv_path = predictions_dir / f"{RUN_ID}_{model_name}_iter{it_idx}.csv"
            df_iter.to_csv(iter_csv_path, index=False)

            # Append rows for the combined CSV
            df_iter_insert = df_iter.copy()
            df_iter_insert.insert(0, "iteration", it_idx)
            combined_rows.append(df_iter_insert)

        # Save combined CSV for the model (all iterations stacked)
        if combined_rows:
            combined_df = pd.concat(combined_rows, ignore_index=True)
            combined_csv_path = predictions_dir / f"{RUN_ID}_{model_name}.csv"
            combined_df.to_csv(combined_csv_path, index=False)
    logger.info("Saved prediction vectors for %d strategies to %s", len(prediction_vectors_memory), predictions_dir)
except (OSError, RuntimeError, ValueError, TypeError) as e:
    logger.debug("Could not write prediction vectors to CSV: %s", e, exc_info=True)

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
try:
    model_keys = list(prediction_vectors_memory.keys())
    # Build empty DataFrame for overall ratios

    correlation_df = pd.DataFrame(index=model_keys, columns=model_keys, dtype=float)
    anticorrelation_df = pd.DataFrame(index=model_keys, columns=model_keys, dtype=float)
    similarity_df = pd.DataFrame(index=model_keys, columns=model_keys, dtype=float)
    iterations_df = pd.DataFrame(index=model_keys, columns=model_keys, dtype=float)

    for tested in model_keys:
        for reference in model_keys:
            try:
                res = compare_models_comparison(
                    prediction_vectors_memory,
                    tested_model=tested,
                    reference_model=reference,
                    baseline_model="actual",
                    include_empty=False
                )
                correlation = res.get("correlation", None)
                anticorrelation = res.get("anticorrelation", None)
                similarity = res.get("similarity", None)
                iterations_used = res.get("counts", {}).get("iterations_used", None)
            except (ValueError, KeyError, TypeError, ZeroDivisionError, IndexError) as e:
                logger.debug("Compare failed for tested=%s ref=%s: %s", tested, reference, e)
                correlation = None
                anticorrelation = None
                similarity = None
                iterations_used = None

            correlation_df.loc[tested, reference] = (correlation * 100.0) if correlation is not None else np.nan
            anticorrelation_df.loc[tested, reference] = (
                (anticorrelation * 100.0)
                if anticorrelation is not None
                else np.nan
            )
            similarity_df.loc[tested, reference] = (similarity * 100.0) if similarity is not None else np.nan

            iterations_df.loc[tested, reference] = iterations_used if iterations_used is not None else np.nan

    # Log the ratio table similar to all_metrics
    logger.info("\n[COMPARISON] Cross-reference (percent) - rows=tested, cols=reference:\n%s", correlation_df.round(2))
    logger.debug("\n[COMPARISON DEBUG] Iterations used per cell:\n%s", iterations_df)

    # Save cross-reference CSV for later analysis (use the same results dir as stats_file_path)
    correlation_csv_path = stats_file_path.parent / f"{RUN_ID}_correlation_matrix.csv"
    anticorrelation_csv_path = stats_file_path.parent / f"{RUN_ID}_anticorrelation_matrix.csv"
    similarity_csv_path = stats_file_path.parent / f"{RUN_ID}_similarity_matrix.csv"
    correlation_csv_path.parent.mkdir(parents=True, exist_ok=True)
    correlation_df.to_csv(correlation_csv_path)
    anticorrelation_df.to_csv(anticorrelation_csv_path)
    similarity_df.to_csv(similarity_csv_path)
    logger.info(
        "[COMPARISON] Saved cross-reference CSV to: %s, %s, %s",
        correlation_csv_path.resolve(), anticorrelation_csv_path.resolve(), similarity_csv_path.resolve()
    )

    # Optional: save a PNG heatmap automatically (use results dir)
    correlation_heatmap_png = stats_file_path.parent / f"{RUN_ID}_correlation_matrix.png"
    anticorrelation_heatmap_png = stats_file_path.parent / f"{RUN_ID}_anticorrelation_matrix.png"
    similarity_heatmap_png = stats_file_path.parent / f"{RUN_ID}_similarity_matrix.png"
    for heatmap_png, df, title in [
        (correlation_heatmap_png, correlation_df, "Correlation: tested vs reference"),
        (anticorrelation_heatmap_png, anticorrelation_df, "Anticorrelation: tested vs reference"),
        (similarity_heatmap_png, similarity_df, "Similarity: tested vs reference"),
    ]:
        try:
            plt.figure(figsize=(max(8, len(df.columns) * 0.5), max(6, len(df.index) * 0.5)))
            sns.heatmap(df.astype(float), annot=False, fmt=".1f", cmap=RED_TO_GREEN_CMAP, cbar_kws={"label": "Percent"})
            plt.title(title)
            plt.tight_layout()
            plt.savefig(heatmap_png, dpi=150)
            plt.close()
            logger.info("[COMPARISON] Saved summary heatmap to: %s", heatmap_png.resolve())
        except (OSError, RuntimeError, ValueError) as e:
            logger.debug("Could not auto-save heatmap image: %s", e, exc_info=True)

            # --- Additional plots: for each ngram (one curve) show values across qlearning windows ---
    try:
        from logicsponge.processmining.utils import plot_ngrams_vs_prefix

        # Q-learning windows
        plot_ngrams_vs_prefix(
            correlation_df,
            NGRAM_NAMES,
            "qlearning_win",
            xlabel="Q-learning window size",
            title="Correlation",
            out_png=stats_file_path.parent / f"{RUN_ID}_ngrams_vs_qlearning_correlation.png",
        )
        plot_ngrams_vs_prefix(
            anticorrelation_df,
            NGRAM_NAMES,
            "qlearning_win",
            xlabel="Q-learning window size",
            title="Anticorrelation",
            out_png=stats_file_path.parent / f"{RUN_ID}_ngrams_vs_qlearning_anticorrelation.png",
        )
        plot_ngrams_vs_prefix(
            similarity_df,
            NGRAM_NAMES,
            "qlearning_win",
            xlabel="Q-learning window size",
            title="Similarity",
            out_png=stats_file_path.parent / f"{RUN_ID}_ngrams_vs_qlearning_similarity.png",
        )

        # Transformer windows
        plot_ngrams_vs_prefix(
            correlation_df,
            NGRAM_NAMES,
            "transformer_win",
            xlabel="Model window size",
            title="Correlation",
            out_png=stats_file_path.parent / f"{RUN_ID}_ngrams_vs_transformer_win_transformer_correlation.png",
        )
        plot_ngrams_vs_prefix(
            anticorrelation_df,
            NGRAM_NAMES,
            "transformer_win",
            xlabel="Model window size",
            title="Anticorrelation",
            out_png=stats_file_path.parent / f"{RUN_ID}_ngrams_vs_transformer_win_transformer_anticorrelation.png",
        )
        plot_ngrams_vs_prefix(
            similarity_df,
            NGRAM_NAMES,
            "transformer_win",
            xlabel="Model window size",
            title="Similarity",
            out_png=stats_file_path.parent / f"{RUN_ID}_ngrams_vs_transformer_win_transformer_similarity.png",
        )

        # LSTM windows
        plot_ngrams_vs_prefix(
            correlation_df,
            NGRAM_NAMES,
            "LSTM_win",
            xlabel="Model window size",
            title="Correlation",
            out_png=stats_file_path.parent / f"{RUN_ID}_ngrams_vs_LSTM_win_lstm_correlation.png",
        )
        plot_ngrams_vs_prefix(
            anticorrelation_df,
            NGRAM_NAMES,
            "LSTM_win",
            xlabel="Model window size",
            title="Anticorrelation",
            out_png=stats_file_path.parent / f"{RUN_ID}_ngrams_vs_LSTM_win_lstm_anticorrelation.png",
        )
        plot_ngrams_vs_prefix(
            similarity_df,
            NGRAM_NAMES,
            "LSTM_win",
            xlabel="Model window size",
            title="Similarity",
            out_png=stats_file_path.parent / f"{RUN_ID}_ngrams_vs_LSTM_win_lstm_similarity.png",
        )
    except Exception:
        logger.exception("Could not generate ngrams vs qlearning plots")
except (ValueError, KeyError, TypeError, ZeroDivisionError, IndexError, OSError, RuntimeError):
    logger.exception("Failed to build cross-reference comparison table")


