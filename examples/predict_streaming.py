"""Streaming prediction example for process mining using LogicSponge."""

import gc
import logging
import time
from pathlib import Path

# ruff: noqa: E402
import torch
from torch import nn, optim
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",  # Only log level and message, no date
)

import logicsponge.core as ls
from logicsponge.core import DataItem, DataItemFilter

# from logicsponge.core import dashboard
from logicsponge.processmining.algorithms_and_structures import Bag, FrequencyPrefixTree, NGram
from logicsponge.processmining.config import DEFAULT_CONFIG
from logicsponge.processmining.models import (
    AdaptiveVoting,
    BasicMiner,
    Fallback,
    HardVoting,
    NeuralNetworkMiner,
    SoftVoting,
)
from logicsponge.processmining.neural_networks import LSTMModel, TransformerModel
from logicsponge.processmining.streaming import (
    AddStartSymbol,
    CSVStatsWriter,
    Evaluation,
    IteratorStreamer,
    PrintEval,
    StreamingActivityPredictor,
)
from logicsponge.processmining.test_data import dataset

logger = logging.getLogger(__name__)
RUN_ID = int(time.time())
stats_to_log = []
stats_file_path = Path(f"results/stats_streaming_{RUN_ID}.csv")


# disable circular gc here, since a phase 2 may take minutes
gc.disable()

# def gb_callback_example(phase, info: dict):
#     print("gc", phase, info)
# gc.callbacks.append(gb_callback_example)

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

# ====================================================
# Initialize models
# ====================================================

config = {
    "include_stop": False,
    "include_time": False,  # True is default
}

start_symbol = DEFAULT_CONFIG["start_symbol"]


# ============================================================
# Generate a list of ngrams to test
# ============================================================
SOFT_VOTING_NGRAMS = [
    (2, 3, 5, 8),
    (2, 3, 4, 5),
]  # (2, 3, 6, 8), (2, 3, 5, 6), (2, 3, 4, 6), (2, 3, 6, 7), (2, 3, 7, 8), (2, 3, 6, 8)

WINDOW_RANGE = [0, 1, 2, 3, 4, 5, 6, 7, 8]  # , 9, 10, 12, 14, 16]

NGRAM_NAMES = [f"ngram_{i + 1}" for i in WINDOW_RANGE]
# ] + [
#     f"ngram_{i+1}_recovery" for i in WINDOW_RANGE
# ]
# ] + [
#     f"ngram_{i+1}_shorts" for i in WINDOW_RANGE
# ]

NGRAM_RETURN_TO_INITIAL = True
# ============================================================

fpt = StreamingActivityPredictor(
    strategy=BasicMiner(algorithm=FrequencyPrefixTree(), config=config),
)

bag = StreamingActivityPredictor(
    strategy=BasicMiner(algorithm=Bag(), config=config),
)

# NGram models
NGRAM_MODELS: dict[str, StreamingActivityPredictor] = {}
for ngram_name in NGRAM_NAMES:
    window_length = int(ngram_name.split("_")[1]) - 1
    # Assuming recovery options are not used in streaming for simplicity, matching current streaming file structure
    NGRAM_MODELS[ngram_name] = StreamingActivityPredictor(
        strategy=BasicMiner(
            algorithm=NGram(window_length=window_length, recover_lengths=[]),
            config=config,
        )
    )


fallback = StreamingActivityPredictor(
    strategy=Fallback(
        models=[
            BasicMiner(algorithm=FrequencyPrefixTree(min_total_visits=10)),
            BasicMiner(algorithm=NGram(window_length=4)),
        ],
        config=config,
    )
)

fallback_ngram8to2 = StreamingActivityPredictor(
    strategy=Fallback(
        models=[
            BasicMiner(algorithm=NGram(window_length=7)),
            BasicMiner(algorithm=NGram(window_length=1)),
        ],
        config=config,
    )
)

fallback_ngram8to3 = StreamingActivityPredictor(
    strategy=Fallback(
        models=[
            BasicMiner(algorithm=NGram(window_length=7)),
            BasicMiner(algorithm=NGram(window_length=2)),
        ],
        config=config,
    )
)

fallback_ngram8to4 = StreamingActivityPredictor(
    strategy=Fallback(
        models=[
            BasicMiner(algorithm=NGram(window_length=7)),
            BasicMiner(algorithm=NGram(window_length=3)),
        ],
        config=config,
    )
)

fallback_ngram10to2 = StreamingActivityPredictor(
    strategy=Fallback(
        models=[
            BasicMiner(algorithm=NGram(window_length=9)),
            BasicMiner(algorithm=NGram(window_length=1)),
        ],
        config=config,
    )
)

fallback_ngram13to2 = StreamingActivityPredictor(
    strategy=Fallback(
        models=[
            BasicMiner(algorithm=NGram(window_length=12)),
            BasicMiner(algorithm=NGram(window_length=1)),
        ],
        config=config,
    )
)

fallback_ngram8to_ooo = StreamingActivityPredictor(
    strategy=Fallback(
        models=[
            BasicMiner(algorithm=NGram(window_length=7)),
            BasicMiner(algorithm=NGram(window_length=6)),
            BasicMiner(algorithm=NGram(window_length=5)),
            BasicMiner(algorithm=NGram(window_length=4)),
            BasicMiner(algorithm=NGram(window_length=3)),
            BasicMiner(algorithm=NGram(window_length=2)),
            BasicMiner(algorithm=NGram(window_length=1)),
            BasicMiner(algorithm=NGram(window_length=0)),
        ],
        config=config,
    )
)

complex_fallback = StreamingActivityPredictor(
    strategy=Fallback(
        models=[
            BasicMiner(algorithm=NGram(window_length=9, min_total_visits=10, min_max_prob=0.9)),
            BasicMiner(algorithm=NGram(window_length=8, min_total_visits=10, min_max_prob=0.9)),
            BasicMiner(algorithm=NGram(window_length=7, min_total_visits=10, min_max_prob=0.8)),
            BasicMiner(algorithm=NGram(window_length=6, min_total_visits=10, min_max_prob=0.7)),
            BasicMiner(algorithm=NGram(window_length=5, min_total_visits=10, min_max_prob=0.6)),
            BasicMiner(algorithm=NGram(window_length=4, min_total_visits=10, min_max_prob=0.0)),
            BasicMiner(algorithm=NGram(window_length=3, min_total_visits=10, min_max_prob=0.0)),
            BasicMiner(algorithm=NGram(window_length=2, min_total_visits=10, min_max_prob=0.0)),
            BasicMiner(algorithm=NGram(window_length=1)),
        ],
        config=config,
    )
)

hard_voting = StreamingActivityPredictor(
    strategy=HardVoting(
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
)

soft_voting = StreamingActivityPredictor(
    strategy=SoftVoting(
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
)

soft_voting_star = StreamingActivityPredictor(
    strategy=SoftVoting(
        models=[
            BasicMiner(algorithm=Bag()),
            BasicMiner(algorithm=FrequencyPrefixTree(min_total_visits=10)),
            BasicMiner(algorithm=NGram(window_length=2, min_total_visits=10)),
            BasicMiner(algorithm=NGram(window_length=3, min_total_visits=10)),
            BasicMiner(algorithm=NGram(window_length=4, min_total_visits=10)),
            BasicMiner(algorithm=NGram(window_length=5, min_total_visits=10)),
            # BasicMiner(algorithm=NGram(window_length=6)),
        ],
        config=config,
    )
)

# list_grams = [(2, 3, 6, 8), (2, 3, 5, 6), (2, 3, 5, 8), (2, 3, 4, 6), (2, 3, 6, 7), (2, 3, 7, 8), (2, 3, 6, 8)]
# Use SOFT_VOTING_NGRAMS for consistency
soft_voting_predictors = {
    f"soft_voting_{grams}": StreamingActivityPredictor(
        strategy=SoftVoting(
            models=[
                BasicMiner(algorithm=Bag()),
                BasicMiner(algorithm=FrequencyPrefixTree(min_total_visits=10)),
                BasicMiner(algorithm=NGram(window_length=grams[0])),
                BasicMiner(algorithm=NGram(window_length=grams[1])),
                BasicMiner(algorithm=NGram(window_length=grams[2])),
                BasicMiner(algorithm=NGram(window_length=grams[3])),
            ],
            config=config,
        )
    )
    for grams in SOFT_VOTING_NGRAMS
}

adaptive_voting = StreamingActivityPredictor(
    strategy=AdaptiveVoting(
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
)

# Initialize LSTMs
vocab_size = 50  # An upper bound on the number of activities
embedding_dim = 50
hidden_dim = 128
output_dim = vocab_size
model_lstm = LSTMModel(vocab_size, embedding_dim, hidden_dim, output_dim, device=device, use_one_hot=True)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model_lstm.parameters(), lr=0.001)

lstm = StreamingActivityPredictor(
    strategy=NeuralNetworkMiner(
        model=model_lstm,
        criterion=criterion,
        optimizer=optimizer,
        batch_size=8,
        config=config,
    )
)


# Initialize transformer model
hidden_dim = 128
output_dim = vocab_size  # Output used to predict the next activity

model_transformer = TransformerModel(
    vocab_size=vocab_size,
    embedding_dim=embedding_dim,
    hidden_dim=hidden_dim,
    output_dim=output_dim,
    use_one_hot=True,
)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model_transformer.parameters(), lr=0.0001)  # Lower learning rate for transformer

transformer = StreamingActivityPredictor(
    strategy=NeuralNetworkMiner(
        model=model_transformer,
        criterion=criterion,
        optimizer=optimizer,
        batch_size=8,
        config=config,
    )
)


# ====================================================
# Sponge
# ====================================================

# Model names
models = [
    "fpt",
    "bag",
    *NGRAM_NAMES,  # Add all NGRAM_NAMES
    "fallback",
    "fallback_ngram8to2",
    "fallback_ngram8to3",
    "fallback_ngram8to4",
    "fallback_ngram10to2",
    "fallback_ngram13to2",
    "fallback_ngram8to_ooo",
    "complex_fallback",
    "hard_voting",
    "adaptive_voting",
    "soft_voting",
    "soft_voting_star",
    *list(soft_voting_predictors.keys()),  # Add all soft_voting predictor names
    "lstm",
    "transformer",
]



metrics_attributes = [
    "accuracy",
    "top_k_accuracy",
    "pp_arithmetic_mean",
    "pp_harmonic_mean",
    "pp_median",
    "pp_q1",
    "pp_q3",
]
metrics_list = [f"{model}.{attribute}" for model in models for attribute in metrics_attributes]

train_latency_list = [f"{model}.train_latency_mean" for model in models]
predict_latency_list = [f"{model}.predict_latency_mean" for model in models]

latency_mean_list = [f"{model}.latency_mean" for model in models]

delay_attributes = [
    "mean_delay_error",
    "mean_actual_delay",
    "mean_normalized_error",
    "delay_predictions",
]

delay_list = [f"{model}.{attribute}" for model in models for attribute in delay_attributes]

all_attributes = [
    "index",
    *metrics_list,
    *train_latency_list,
    *predict_latency_list,
    *latency_mean_list,
]  # , *delay_list]

# streamer is defined after len_dataset so tqdm can show total progress


def start_filter(item: DataItem) -> bool:
    """Filter function to check if the activity is not the start symbol."""
    return item["activity"] != start_symbol


# len_dataset = 15214
len_dataset = 262200  # BPI Challenge 2012
# len_dataset = 65000
# len_dataset = 1202267
# len_dataset = 2514266
# len_dataset = 1595923

# Initialize streamer with tqdm progress bar to track dataset processing
streamer = IteratorStreamer(
    data_iterator=iter(tqdm(dataset, total=len_dataset, desc="Streaming", unit="evt"))
)

# sponge = (
#     streamer
#     * ls.KeyFilter(keys=["case_id", "activity", "timestamp"])
#     * AddStartSymbol(start_symbol=start_symbol)
#     * (
#         (fpt * DataItemFilter(data_item_filter=start_filter) * Evaluation("fpt"))
#         # | (bag * DataItemFilter(data_item_filter=start_filter) * Evaluation("bag"))
#         # | (ngram_1 * DataItemFilter(data_item_filter=start_filter) * Evaluation("ngram_1"))
#         # | (ngram_2 * DataItemFilter(data_item_filter=start_filter) * Evaluation("ngram_2"))
#         # | (ngram_3 * DataItemFilter(data_item_filter=start_filter) * Evaluation("ngram_3"))
#         # | (ngram_4 * DataItemFilter(data_item_filter=start_filter) * Evaluation("ngram_4"))
#         # | (ngram_5 * DataItemFilter(data_item_filter=start_filter) * Evaluation("ngram_5"))
#         # | (ngram_6 * DataItemFilter(data_item_filter=start_filter) * Evaluation("ngram_6"))
#         # | (ngram_7 * DataItemFilter(data_item_filter=start_filter) * Evaluation("ngram_7"))
#         # | (ngram_8 * DataItemFilter(data_item_filter=start_filter) * Evaluation("ngram_8"))
#         # | (fallback * DataItemFilter(data_item_filter=start_filter) * Evaluation("fallback"))
#         # | (hard_voting * DataItemFilter(data_item_filter=start_filter) * Evaluation("hard_voting"))
#         # | (soft_voting * DataItemFilter(data_item_filter=start_filter) * Evaluation("soft_voting"))
#         # | (adaptive_voting * DataItemFilter(data_item_filter=start_filter) * Evaluation("adaptive_voting"))
#         | (lstm * DataItemFilter(data_item_filter=start_filter) * Evaluation("lstm"))
#     )
#     * ls.MergeToSingleStream() * ls.Flatten()
#     * ls.AddIndex(key="index", index=1)
#     * ls.KeyFilter(keys=all_attributes)
#     * ls.DataItemFilter(data_item_filter=lambda item: item["index"] % 100 == 0 or item["index"] > len_dataset - 10)
#     * PrintEval()
#     # * ls.Print()
#     # * (dashboard.Plot("Accuracy (%)", x="index", y=accuracy_list))
#     # * (dashboard.Plot("Latency Mean (ms)", x="index", y=latency_mean_list))
# )


sponge = (
    streamer
    * ls.KeyFilter(keys=["case_id", "activity", "timestamp"])
    * (
        (fpt * Evaluation("fpt"))
        | (bag * Evaluation("bag"))
        # Add all NGRAM_MODELS to the pipeline
        | ((NGRAM_MODELS[name] * Evaluation(name)) for name in NGRAM_NAMES)
        | (fallback * Evaluation("fallback"))
        # | (fallback_ngram8to2 * Evaluation("fallback_ngram8to2"))
        # | (fallback_ngram8to3 * Evaluation("fallback_ngram8to3"))
        # | (fallback_ngram8to4 * Evaluation("fallback_ngram8to4"))
        # | (fallback_ngram10to2 * Evaluation("fallback_ngram10to2"))
        # | (fallback_ngram13to2 * Evaluation("fallback_ngram13to2"))
        # | (fallback_ngram8to_ooo * Evaluation("fallback_ngram8to_ooo"))
        # | (complex_fallback * Evaluation("complex_fallback"))
        | (hard_voting * Evaluation("hard_voting"))
        # | (adaptive_voting * Evaluation("adaptive_voting"))
        | (soft_voting * Evaluation("soft_voting"))
        | (soft_voting_star * Evaluation("soft_voting_star"))
        # Add all soft_voting_predictors to the pipeline
        | ((predictor * Evaluation(name)) for name, predictor in soft_voting_predictors.items())
        | (
            AddStartSymbol(start_symbol=start_symbol)
            * lstm
            * DataItemFilter(data_item_filter=start_filter)
            * Evaluation("lstm")
        )
        | (
            AddStartSymbol(start_symbol=start_symbol)
            * transformer
            * DataItemFilter(data_item_filter=start_filter)
            * Evaluation("transformer")
        )
    )
    * ls.MergeToSingleStream()
    * ls.Flatten()
    * ls.AddIndex(key="index", index=1)
    * ls.KeyFilter(keys=all_attributes)
    * ls.DataItemFilter(data_item_filter=lambda item: item["index"] % 100 == 0 or item["index"] == len_dataset - 1)
    * (PrintEval() | CSVStatsWriter(csv_path=stats_file_path))
    # * ls.Print()
    # * (dashboard.Plot("Accuracy (%)", x="index", y=accuracy_list))
    # * (dashboard.Plot("Latency Mean (ms)", x="index", y=latency_mean_list))
)


sponge.start()

# dashboard.show_stats(sponge)
# dashboard.run()
