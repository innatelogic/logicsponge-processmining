import logicsponge.core as ls
from logicsponge.processmining.algorithms_and_structures import Bag, FrequencyPrefixTree, NGram
from logicsponge.processmining.models import BasicMiner, SoftVoting
from logicsponge.processmining.streaming import IteratorStreamer, StreamingActivityPredictor
from logicsponge.processmining.test_data import dataset

# ====================================================
# Initialize models
# ====================================================

config = {
    "include_stop": False,
}

model1 = StreamingActivityPredictor(
    strategy=BasicMiner(algorithm=NGram(window_length=5), config=config),
)

model2 = StreamingActivityPredictor(
    strategy=SoftVoting(
        models=[
            BasicMiner(algorithm=Bag()),
            BasicMiner(algorithm=FrequencyPrefixTree(min_total_visits=10)),
            BasicMiner(algorithm=NGram(window_length=2)),
            BasicMiner(algorithm=NGram(window_length=3)),
            BasicMiner(algorithm=NGram(window_length=4)),
        ],
        config=config,
    )
)


# ====================================================
# Sponge
# ====================================================


streamer = IteratorStreamer(data_iterator=dataset)

sponge = (
    streamer
    * ls.KeyFilter(keys=["case_id", "activity", "timestamp"])
    * model2
    * ls.AddIndex(key="index", index=1)
    * ls.Print()
)


sponge.start()
