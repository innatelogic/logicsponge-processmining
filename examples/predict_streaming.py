import gc
import logging

import torch
from torch import nn, optim

# ruff: noqa: E402
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
from logicsponge.processmining.neural_networks import LSTMModel
from logicsponge.processmining.streaming import (
    AddStartSymbol,
    Evaluation,
    IteratorStreamer,
    PrintEval,
    StreamingActivityPredictor,
)
from logicsponge.processmining.test_data import dataset

logger = logging.getLogger(__name__)

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


fpt = StreamingActivityPredictor(
    strategy=BasicMiner(algorithm=FrequencyPrefixTree(), config=config),
)

bag = StreamingActivityPredictor(
    strategy=BasicMiner(algorithm=Bag(), config=config),
)

ngram_1 = StreamingActivityPredictor(
    strategy=BasicMiner(algorithm=NGram(window_length=0), config=config),
)

ngram_2 = StreamingActivityPredictor(
    strategy=BasicMiner(algorithm=NGram(window_length=1), config=config),
)

ngram_3 = StreamingActivityPredictor(
    strategy=BasicMiner(algorithm=NGram(window_length=2), config=config),
)

ngram_4 = StreamingActivityPredictor(
    strategy=BasicMiner(algorithm=NGram(window_length=3), config=config),
)

ngram_5 = StreamingActivityPredictor(
    strategy=BasicMiner(algorithm=NGram(window_length=4), config=config),
)

ngram_6 = StreamingActivityPredictor(
    strategy=BasicMiner(algorithm=NGram(window_length=5), config=config),
)

ngram_7 = StreamingActivityPredictor(
    strategy=BasicMiner(algorithm=NGram(window_length=6), config=config),
)

ngram_8 = StreamingActivityPredictor(
    strategy=BasicMiner(algorithm=NGram(window_length=7), config=config),
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

adaptive_ngram = StreamingActivityPredictor(
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
            BasicMiner(algorithm=NGram(window_length=5)),
            # BasicMiner(algorithm=NGram(window_length=6)),
        ],
        config=config,
    )
)

list_grams = [(2, 3, 6, 8), (2, 3, 5, 6), (2, 3, 5, 8), (2, 3, 4, 6), (2, 3, 6, 7), (2, 3, 7, 8), (2, 3, 6, 8)]
soft_voting_tests = []

for grams in list_grams:
    soft_voting_tests.append(
        StreamingActivityPredictor(
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
    )

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
model = LSTMModel(vocab_size, embedding_dim, hidden_dim, output_dim, device=device, use_one_hot=True)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

lstm = StreamingActivityPredictor(
    strategy=NeuralNetworkMiner(
        model=model,
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
    "ngram_1",
    "ngram_2",
    "ngram_3",
    "ngram_4",
    "ngram_5",
    "ngram_6",
    "ngram_7",
    "ngram_8",
    "fallback",
    "adaptive_ngram",
    "hard_voting",
    "soft_voting",
    "soft_voting_test_1",
    "soft_voting_test_2",
    "soft_voting_test_3",
    "soft_voting_test_4",
    "soft_voting_test_5",
    "soft_voting_test_6",
    "soft_voting_test_7",
    "adaptive_voting",
    "lstm"
]

accuracy_list = [f"{model}.accuracy" for model in models]
latency_mean_list = [f"{model}.latency_mean" for model in models]

delay_attributes = [
    "mean_delay_error",
    "mean_actual_delay",
    "mean_normalized_error",
    "delay_predictions",
]

delay_list = [f"{model}.{attribute}" for model in models for attribute in delay_attributes]

all_attributes = ["index", *accuracy_list, *latency_mean_list, *delay_list]

streamer = IteratorStreamer(data_iterator=dataset)


def start_filter(item: DataItem):
    return item["activity"] != start_symbol


len_dataset = 15214
# len_dataset = 262200
# len_dataset = 65000
# len_dataset = 1202267
# len_dataset = 2514266
# len_dataset = 1595923

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
#     * ls.ToSingleStream(flatten=True)
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
        | (ngram_1 * Evaluation("ngram_1"))
        | (ngram_2 * Evaluation("ngram_2"))
        | (ngram_3 * Evaluation("ngram_3"))
        | (ngram_4 * Evaluation("ngram_4"))
        | (ngram_5 * Evaluation("ngram_5"))
        | (ngram_6 * Evaluation("ngram_6"))
        | (ngram_7 * Evaluation("ngram_7"))
        | (ngram_8 * Evaluation("ngram_8"))
        | (fallback * Evaluation("fallback"))
        | (adaptive_ngram * Evaluation("adaptive_ngram"))
        | (hard_voting * Evaluation("hard_voting"))
        | (soft_voting * Evaluation("soft_voting"))
        | (soft_voting_tests[0] * Evaluation("soft_voting_test_1"))
        | (soft_voting_tests[1] * Evaluation("soft_voting_test_2"))
        | (soft_voting_tests[2] * Evaluation("soft_voting_test_3"))
        | (soft_voting_tests[3] * Evaluation("soft_voting_test_4"))
        | (soft_voting_tests[4] * Evaluation("soft_voting_test_5"))
        | (soft_voting_tests[5] * Evaluation("soft_voting_test_6"))
        | (soft_voting_tests[6] * Evaluation("soft_voting_test_7"))
        | (adaptive_voting * Evaluation("adaptive_voting"))
        | (
            AddStartSymbol(start_symbol=start_symbol)
            * lstm
            * DataItemFilter(data_item_filter=start_filter)
            * Evaluation("lstm")
        )
    )
    * ls.ToSingleStream(flatten=True)
    * ls.AddIndex(key="index", index=1)
    * ls.KeyFilter(keys=all_attributes)
    * ls.DataItemFilter(data_item_filter=lambda item: item["index"] % 100 == 0 or item["index"] > len_dataset - 10)
    * PrintEval()
    # * ls.Print()
    # * (dashboard.Plot("Accuracy (%)", x="index", y=accuracy_list))
    # * (dashboard.Plot("Latency Mean (ms)", x="index", y=latency_mean_list))
)


sponge.start()

# dashboard.show_stats(sponge)
# dashboard.run()
