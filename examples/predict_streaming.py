import gc
import logging

import torch
from torch import nn, optim

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
from logicsponge.processmining.streaming import AddStartSymbol, Evaluation, IteratorStreamer, StreamingActivityPredictor
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
model = LSTMModel(vocab_size, embedding_dim, hidden_dim, output_dim, device=device)
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
    # "ngram_1",
    # "ngram_2",
    "ngram_3",
    "ngram_4",
    "ngram_5",
    # "ngram_6",
    # "ngram_7",
    # "ngram_8",
    "fallback",
    "hard_voting",
    "soft_voting",
    "adaptive_voting",
    "lstm",
]

accuracy_list = [f"{model}.accuracy" for model in models]
latency_mean_list = [f"{model}.latency_mean" for model in models]
latency_max_list = [f"{model}.latency_max" for model in models]
all_attributes = ["index", *accuracy_list, *latency_mean_list]

streamer = IteratorStreamer(data_iterator=dataset)


def start_filter(item: DataItem):
    return item["activity"] != start_symbol


len_dataset = 15214

sponge = (
    streamer
    * ls.KeyFilter(keys=["case_id", "activity"])
    * AddStartSymbol(start_symbol=start_symbol)
    * (
        (fpt * DataItemFilter(data_item_filter=start_filter) * Evaluation("fpt"))
        | (bag * DataItemFilter(data_item_filter=start_filter) * Evaluation("bag"))
        # | (ngram_1 * DataItemFilter(data_item_filter=start_filter) * Evaluation("ngram_1"))
        # | (ngram_2 * DataItemFilter(data_item_filter=start_filter) * Evaluation("ngram_2"))
        | (ngram_3 * DataItemFilter(data_item_filter=start_filter) * Evaluation("ngram_3"))
        | (ngram_4 * DataItemFilter(data_item_filter=start_filter) * Evaluation("ngram_4"))
        | (ngram_5 * DataItemFilter(data_item_filter=start_filter) * Evaluation("ngram_5"))
        # | (ngram_6 * DataItemFilter(data_item_filter=start_filter) * Evaluation("ngram_6"))
        # | (ngram_7 * DataItemFilter(data_item_filter=start_filter) * Evaluation("ngram_7"))
        # | (ngram_8 * DataItemFilter(data_item_filter=start_filter) * Evaluation("ngram_8"))
        | (fallback * DataItemFilter(data_item_filter=start_filter) * Evaluation("fallback"))
        | (hard_voting * DataItemFilter(data_item_filter=start_filter) * Evaluation("hard_voting"))
        | (soft_voting * DataItemFilter(data_item_filter=start_filter) * Evaluation("soft_voting"))
        | (adaptive_voting * DataItemFilter(data_item_filter=start_filter) * Evaluation("adaptive_voting"))
        | (lstm * DataItemFilter(data_item_filter=start_filter) * Evaluation("lstm"))
    )
    * ls.ToSingleStream(flatten=True)
    * ls.AddIndex(key="index", index=1)
    * ls.KeyFilter(keys=all_attributes)
    * ls.DataItemFilter(data_item_filter=lambda item: item["index"] % 100 == 0 or item["index"] > len_dataset - 10)
    * ls.Print()
    # * (dashboard.Plot("Accuracy (%)", x="index", y=accuracy_list))
    # * (dashboard.Plot("Latency Mean (ms)", x="index", y=latency_mean_list))
)

sponge.start()

# dashboard.show_stats(sponge)
# dashboard.run()
