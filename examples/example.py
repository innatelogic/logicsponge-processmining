import logicsponge.core as ls
from logicsponge.processmining.algorithms_and_structures import Bag, FrequencyPrefixTree, NGram
from logicsponge.processmining.models import BasicMiner, SoftVoting
from logicsponge.processmining.streaming import IteratorStreamer, StreamingActionPredictor
from logicsponge.processmining.test_data import dataset

# ====================================================
# Initialize models
# ====================================================

config = {
    "include_stop": False,
}

model1 = StreamingActionPredictor(
    strategy=BasicMiner(algorithm=NGram(window_length=5), config=config),
)

model2 = StreamingActionPredictor(
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
    * ls.KeyFilter(keys=["case_id", "action"])
    * model2
    * ls.AddIndex(key="index", index=1)
    * ls.Print()
)


sponge.start()


d = {
    'case_id': 'FAA',
    'action': 'Return ER',
    'prediction': {
        'action': 'Return ER',
        'top_k_actions': ['Return ER', 'Leucocytes', 'Release E'],
        'probability': 0.9986388006307096,
        'probs': {
            # [...]
            'Leucocytes': 0.0013611993692904283,
            'Return ER': 0.9986388006307096,
            # [...]
        }
    },
    'latency': 0.06985664367675781,
    'index': 15214
}
