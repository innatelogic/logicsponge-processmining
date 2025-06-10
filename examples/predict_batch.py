"""Module to evaluate and compare different process mining models."""

import json
import logging
import time
from datetime import timedelta
from pathlib import Path

import matplotlib as mpl
import numpy as np
import pandas as pd
import torch
from aalpy.learning_algs import run_Alergia
from torch import nn, optim
from tqdm import tqdm

# ruff: noqa: E402
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
)

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
    TransformerModel,
    evaluate_rnn,
    evaluate_transformer,
    train_rnn,
    train_transformer,
)
from logicsponge.processmining.test_data import data_name, dataset, dataset_test
from logicsponge.processmining.utils import compute_perplexity_stats

SEC_TO_MICRO = 1_000_000

# ============================================================
# Generate a list of ngrams to test
# ============================================================
VOTING_NGRAMS = [(2, 3, 4), (2, 3, 5, 8), (2, 3, 4, 5)] # (2, 3, 5, 6), (2, 3, 5, 7), (2, 3, 4, 7)

SELECT_BEST_ARGS = ["prob"] # ["acc", "prob", "prob x acc"]

WINDOW_RANGE = [0, 1, 2, 3, 4, 5, 6, 7, 8] #, 9, 10, 12, 14, 16]

NGRAM_NAMES = [f"ngram_{i + 1}" for i in WINDOW_RANGE]
# ] + [
#     f"ngram_{i+1}_recovery" for i in WINDOW_RANGE
# ]
# ] + [
#     f"ngram_{i+1}_shorts" for i in WINDOW_RANGE
# ]

# ============================================================

mpl.use("Agg")

pd.set_option("display.max_columns", None)  # Show all columns
pd.set_option("display.expand_frame_repr", False)  # Prevent line-wrapping # noqa: FBT003

logger = logging.getLogger(__name__)

RUN_ID = int(time.time())
stats_to_log = []


# Create an ID for the current run
stats_file_path = Path(f"results/stats_batch_{RUN_ID}.json")
log_file_path = Path(f"results/log_{RUN_ID}.txt")
log_file_path.parent.mkdir(parents=True, exist_ok=True)
with log_file_path.open("w") as f:
    for handler in logging.root.handlers[:]:
        if isinstance(handler, logging.FileHandler):
            handler.flush()
            handler.close()
            logging.root.removeHandler(handler)

    # Reconfigure the logger to write to the file
    file_handler = logging.FileHandler(log_file_path)
    formatter = logging.Formatter("%(message)s")
    file_handler.setFormatter(formatter)
    logging.root.addHandler(file_handler)

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

NN_TRAINING = True
SKIP_LSTM = False
SKIP_TRANSFORMER = True
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
n_activities = data_statistics(data)

data_test = transform_to_seqs(dataset_test)

# ============================================================
# Define the number of iterations
# ============================================================

n_iterations = 5

# Store metrics across iterations
all_metrics = {
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
        "bayesian train",
        "bayesian test",
        "bayesian t+t",
        "bayesian test nonsingle",
        "bayesian t+t nonsingle",
    ]
}

# Repeat the experiment n_iterations times
for iteration in range(n_iterations):
    msg = f"Starting iteration {iteration + 1}/{n_iterations}..."
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
        NN_TRAINING = False

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

    # data_statistics(test_set_transformed)

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
                algorithm=NGram(
                    window_length=window_length, recover_lengths=[]
                ),
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
            BasicMiner(
                algorithm=NGram(
                    window_length=9, min_total_visits=10, min_max_prob=0.9
                )
            ),
            BasicMiner(
                algorithm=NGram(
                    window_length=8, min_total_visits=10, min_max_prob=0.9
                )
            ),
            BasicMiner(
                algorithm=NGram(
                    window_length=7, min_total_visits=10, min_max_prob=0.8
                )
            ),
            BasicMiner(
                algorithm=NGram(
                    window_length=6, min_total_visits=10, min_max_prob=0.7
                )
            ),
            BasicMiner(
                algorithm=NGram(
                    window_length=5, min_total_visits=10, min_max_prob=0.6
                )
            ),
            BasicMiner(
                algorithm=NGram(
                    window_length=4, min_total_visits=10, min_max_prob=0.0
                )
            ),
            BasicMiner(
                algorithm=NGram(
                    window_length=3, min_total_visits=10, min_max_prob=0.0
                ),
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
            ] + (
                [
                    BasicMiner(algorithm=NGram(window_length=grams[optional_num_ngrams], min_total_visits=10))
                ]
                if len(grams) > optional_num_ngrams else []
            ),
            select_best=select_best_arg,
            config=config
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

    soft_voting_list = (
        [
            SoftVoting(
                models=[
                    BasicMiner(algorithm=Bag()),
                    BasicMiner(algorithm=FrequencyPrefixTree(min_total_visits=10)),
                    BasicMiner(algorithm=NGram(window_length=grams[0])),
                    BasicMiner(algorithm=NGram(window_length=grams[1])),
                    BasicMiner(algorithm=NGram(window_length=grams[2])),
                ] + (
                    [
                        BasicMiner(algorithm=NGram(window_length=grams[optional_num_ngrams]))
                    ] if len(grams) > optional_num_ngrams else []
                ),
                config=config,
            )
            for grams in VOTING_NGRAMS
        ]
        +
        [
            SoftVoting(
                models=[
                    BasicMiner(algorithm=Bag()),
                    BasicMiner(algorithm=FrequencyPrefixTree(min_total_visits=10)),
                    BasicMiner(algorithm=NGram(window_length=grams[0], min_total_visits=10)),
                    BasicMiner(algorithm=NGram(window_length=grams[1], min_total_visits=10)),
                    BasicMiner(algorithm=NGram(window_length=grams[2], min_total_visits=10)),
                ]
                + (
                    [
                        BasicMiner(algorithm=NGram(window_length=grams[optional_num_ngrams], min_total_visits=10))
                    ]
                    if len(grams) > optional_num_ngrams else []
                ),
            )
            for grams in VOTING_NGRAMS
        ]
    )

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

    # ============= Train Alergia
    alergia_start_time = time.time()

    algorithm = run_Alergia(alergia_train_set_transformed, automaton_type="smm", eps=0.5, print_info=True)
    smm = Alergia(algorithm=algorithm)

    alergia_end_time = time.time()
    alergia_elapsed_time = alergia_end_time - alergia_start_time
    msg = f"Training time for Alergia: {alergia_elapsed_time:.4f} seconds"
    logger.info(msg)

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
        for grams, soft_voting_test in zip(VOTING_NGRAMS, soft_voting_list[:len(VOTING_NGRAMS)], strict=False)
    }
    soft_voting_strategies2 = {
        f"soft voting {grams}*": (soft_voting_test, test_set_transformed)
        for grams, soft_voting_test in zip(VOTING_NGRAMS, soft_voting_list[len(VOTING_NGRAMS):], strict=False)
    }
    soft_voting_strategies = {**soft_voting_strategies1, **soft_voting_strategies2}
    bayesian_strategies = {model_name: (model, test_set_transformed) for model_name, model in BAYESIAN_MODELS.items()}
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
        "alergia": (smm, test_set_transformed),
        **bayesian_strategies,
    }

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
        training_times[strategy_name] *= SEC_TO_MICRO # Convert to microseconds

    # ============================================================
    # Evaluation
    # ============================================================

    # Store the statistics for each iteration and also print them out
    iteration_data = {
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

        evaluation_time = strategy.evaluate(test_data, mode="incremental", debug=(data_name == "Synthetic_Train"), compute_perplexity="hard" not in strategy_name)
        evaluation_time *= SEC_TO_MICRO / TEST_EVENTS

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
                "per_state_stats": per_state_stats
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

    # LSTM Evaluation
    if NN_TRAINING:
        msg = "Training and evaluating LSTM model..."
        logger.info(msg)

        # For RNNs: Append START symbol
        nn_train_set_transformed = add_start_to_sequences(train_set_transformed, start_symbol)
        nn_val_set_transformed = add_start_to_sequences(val_set_transformed, start_symbol)
        nn_test_set_transformed = add_start_to_sequences(test_set_transformed, start_symbol)

        nn_train_set_transformed = nn_processor.preprocess_data(nn_train_set_transformed)
        nn_val_set_transformed = nn_processor.preprocess_data(nn_val_set_transformed)
        nn_test_set_transformed = nn_processor.preprocess_data(nn_test_set_transformed)

        vocab_size = 50  # Assume an upper bound on the number of activities

        # Initialize the model, criterion, and optimizer
        embedding_dim = 50
        hidden_dim = 128
        output_dim = vocab_size  # Output used to predict the next activity

        model = LSTMModel(vocab_size, embedding_dim, hidden_dim, output_dim, device=device, use_one_hot=True)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        # Train the LSTM on the train set with batch size and sequence-to-sequence targets
        start_time = time.time()
        model = train_rnn(
            model, nn_train_set_transformed, nn_val_set_transformed, criterion, optimizer, batch_size=8, epochs=20
        ) if not SKIP_LSTM else model
        end_time = time.time()
        training_time = (end_time - start_time) * SEC_TO_MICRO / (TRAIN_EVENTS + VAL_EVENTS)

        lstm_stats, lstm_perplexities, lstm_eval_time = evaluate_rnn(
            model, nn_test_set_transformed, dataset_type="Test", max_k=config["top_k"]
        )
        lstm_perplexity_stats = compute_perplexity_stats(lstm_perplexities)
        lstm_eval_time *= SEC_TO_MICRO / TEST_EVENTS

        # if SHOW_DELAYS:
        #     # WARNING: LSTM DOES NOT CALCULATES DELAYS SO FAR ??
        #     all_metrics["LSTM"]["mean_delay_error"].append(mean_delay_error)
        #     all_metrics["LSTM"]["mean_actual_delay"].append(mean_actual_delay)
        #     all_metrics["LSTM"]["mean_normalized_error"].append(mean_normalized_error)
        #     all_metrics["LSTM"]["num_delay_predictions"].append(delay_count)

        if (
            not isinstance(lstm_stats["top_k_correct_preds"], list)
            or not isinstance(lstm_stats["total_predictions"], int)
            or not isinstance(lstm_stats["accuracy"], float)
        ):
            msg = "LSTM stats are not in the expected format."
            raise TypeError(msg)
        # Append data to the iteration data dictionary
        iteration_data["Model"].append("LSTM")

        iteration_data["PP Harmo"].append(lstm_perplexity_stats["pp_harmonic_mean"])
        iteration_data["PP Arithm"].append(lstm_perplexity_stats["pp_arithmetic_mean"])
        iteration_data["PP Median"].append(lstm_perplexity_stats["pp_median"])
        iteration_data["PP Q1"].append(lstm_perplexity_stats["pp_q1"])
        iteration_data["PP Q3"].append(lstm_perplexity_stats["pp_q3"])

        iteration_data["Correct (%)"].append(lstm_stats["accuracy"] * 100)
        iteration_data["Wrong (%)"].append(100 - lstm_stats["accuracy"] * 100)
        iteration_data["Empty (%)"].append(0.0)

        for k in range(1, config["top_k"]):
            iteration_data[f"Top-{k + 1}"].append(
                lstm_stats["top_k_correct_preds"][k] / lstm_stats["total_predictions"] * 100
            )

        iteration_data["Pred Time"].append(lstm_eval_time)
        iteration_data["Train Time"].append(training_time)

        iteration_data["Good Preds"].append(lstm_stats["correct_predictions"])
        iteration_data["Tot Preds"].append(lstm_stats["total_predictions"])
        iteration_data["Nb States"].append(None)


        stats_to_log.append(
            {
                "strategy": "LSTM",
                "strategy_accuracy": lstm_stats["accuracy"] * 100,
                "strategy_perplexity": lstm_perplexity_stats["pp_harmonic_mean"],
                "strategy_eval_time": lstm_eval_time,
                # "per_state_stats": None
            }
        )

        all_metrics["LSTM"]["accuracies"].append(lstm_stats["accuracy"])
        all_metrics["LSTM"]["pp_arithmetic_mean"].append(lstm_perplexity_stats["pp_arithmetic_mean"])
        all_metrics["LSTM"]["pp_harmonic_mean"].append(lstm_perplexity_stats["pp_harmonic_mean"])
        all_metrics["LSTM"]["pp_median"].append(lstm_perplexity_stats["pp_median"])
        all_metrics["LSTM"]["pp_q1"].append(lstm_perplexity_stats["pp_q1"])
        all_metrics["LSTM"]["pp_q3"].append(lstm_perplexity_stats["pp_q3"])
        for k in range(1, config["top_k"]):
            all_metrics["LSTM"][f"top-{k + 1}"].append(iteration_data[f"Top-{k + 1}"][-1])

        all_metrics["LSTM"]["pred_time"].append(lstm_eval_time)
        all_metrics["LSTM"]["train_time"].append(training_time)

        all_metrics["LSTM"]["num_states"].append(0)
        all_metrics["LSTM"]["mean_delay_error"].append(None)
        all_metrics["LSTM"]["mean_actual_delay"].append(None)
        all_metrics["LSTM"]["mean_normalized_error"].append(None)
        all_metrics["LSTM"]["num_delay_predictions"].append(None)



        # ============================================================
        # ============================================================

        # Initialize the transformer model
        msg = "Training and evaluating transformer model..."
        logger.info(msg)

        nhead = 2
        num_encoder_layers = 2
        num_decoder_layers = 2
        dim_feedforward = 128
        dropout = 0.1

        model = TransformerModel(
            vocab_size=vocab_size,
            embedding_dim=embedding_dim,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            device=device,
            use_one_hot=True  # or False, depending on your preference
        )

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.0001)  # Lower learning rate for transformer


        # Train the transformer
        start_time = time.time()
        model = train_transformer(
            model, nn_train_set_transformed, nn_val_set_transformed, criterion, optimizer,
            batch_size=8, epochs=20
        ) if not SKIP_TRANSFORMER else model
        end_time = time.time()
        training_time = (end_time - start_time) * SEC_TO_MICRO / (TRAIN_EVENTS + VAL_EVENTS)

        # Evaluate the transformer
        transformer_stats, transformer_perplexities, transformer_eval_time = evaluate_transformer(
            model, nn_test_set_transformed, dataset_type="Test", max_k=config["top_k"]
        )
        transformer_perplexity_stats = compute_perplexity_stats(transformer_perplexities)
        transformer_eval_time *= SEC_TO_MICRO / TEST_EVENTS

        # Validation checks
        if (
            not isinstance(transformer_stats["top_k_correct_preds"], list)
            or not isinstance(transformer_stats["total_predictions"], int)
            or not isinstance(transformer_stats["accuracy"], float)
        ):
            msg = "transformer stats are not in the expected format."
            raise TypeError(msg)

        # Append data to the iteration data dictionary
        iteration_data["Model"].append("transformer")

        iteration_data["PP Harmo"].append(transformer_perplexity_stats["pp_harmonic_mean"])
        iteration_data["PP Arithm"].append(transformer_perplexity_stats["pp_arithmetic_mean"])
        iteration_data["PP Median"].append(transformer_perplexity_stats["pp_median"])
        iteration_data["PP Q1"].append(transformer_perplexity_stats["pp_q1"])
        iteration_data["PP Q3"].append(transformer_perplexity_stats["pp_q3"])

        iteration_data["Correct (%)"].append(transformer_stats["accuracy"] * 100)
        iteration_data["Wrong (%)"].append(100 - transformer_stats["accuracy"] * 100)
        iteration_data["Empty (%)"].append(0.0)

        for k in range(1, config["top_k"]):
            iteration_data[f"Top-{k + 1}"].append(
                transformer_stats["top_k_correct_preds"][k] / transformer_stats["total_predictions"] * 100
            )

        iteration_data["Pred Time"].append(transformer_eval_time)
        iteration_data["Train Time"].append(training_time)

        iteration_data["Good Preds"].append(transformer_stats["correct_predictions"])
        iteration_data["Tot Preds"].append(transformer_stats["total_predictions"])
        iteration_data["Nb States"].append(None)


        # Add to stats_to_log
        stats_to_log.append(
            {
                "strategy": "transformer",
                "strategy_accuracy": transformer_stats["accuracy"] * 100,
                "strategy_perplexity": transformer_perplexity_stats["pp_harmonic_mean"],
                "strategy_eval_time": transformer_eval_time,
            }
        )

        # Add to all_metrics
        all_metrics["transformer"]["accuracies"].append(transformer_stats["accuracy"])
        all_metrics["transformer"]["pp_arithmetic_mean"].append(transformer_perplexity_stats["pp_arithmetic_mean"])
        all_metrics["transformer"]["pp_harmonic_mean"].append(transformer_perplexity_stats["pp_harmonic_mean"])
        all_metrics["transformer"]["pp_median"].append(transformer_perplexity_stats["pp_median"])
        all_metrics["transformer"]["pp_q1"].append(transformer_perplexity_stats["pp_q1"])
        all_metrics["transformer"]["pp_q3"].append(transformer_perplexity_stats["pp_q3"])

        for k in range(1, config["top_k"]):
            all_metrics["transformer"][f"top-{k + 1}"].append(iteration_data[f"Top-{k + 1}"][-1])

        all_metrics["transformer"]["pred_time"].append(transformer_eval_time)
        all_metrics["transformer"]["train_time"].append(training_time)

        all_metrics["transformer"]["num_states"].append(0)
        all_metrics["transformer"]["mean_delay_error"].append(None)
        all_metrics["transformer"]["mean_actual_delay"].append(None)
        all_metrics["transformer"]["mean_normalized_error"].append(None)
        all_metrics["transformer"]["num_delay_predictions"].append(None)

    # Create a DataFrame for the iteration and log it
    iteration_df = pd.DataFrame(iteration_data).round(2)

    # iteration_df = iteration_df.drop(columns=["PP Median", "PP Q1", "PP Q3"])

    msg = f"\nIteration {iteration + 1} stats:\n{iteration_df}"
    logger.info(msg)

# ============================================================
# Calculate and Show Final Results
# ============================================================


with stats_file_path.open("w") as f:
    json.dump(stats_to_log, f, indent=4)

results = {
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
