<img src="media/logicsponge.png" alt="LogicSponge Logo" width="350">

**logicsponge-processmining** is a library for process-mining tasks that is built on **logicsponge-core**. Process mining involves a set of tools for modeling, analyzing, and improving business processes.

# In a nutshell

The current implementation includes the following features:
- Event-log prediction in both batch and streaming modes, using frequency prefix trees, n-grams, LSTMs, and ensemble methods.
- Visualization of event streams based on their prefix trees.

# Getting started

We recommend starting with our [logicsponge tutorial](https://github.com/innatelogic/logicsponge) to get acquainted with the basics of how logicsponge processes data streams.   Afterwards, to get started with logicsponge-processmining, install it using pip:

```sh
pip install logicsponge-processmining
```

# Event-log prediction

Event-log prediction involves anticipating events given historical data about a process. In the streaming case, we receive a sequence of events, where each event is a pair
(case ID, activity) consisting of a case ID and an activity (also referred to as action). As events arrive, we train a model incrementally, allowing it to predict the next activity for a given case based on
the sequence of activities observed so far.

logicsponge-processmining offers several predefined models: frequency prefix trees, n-grams, LSTMs, and ensemble methods (soft, hard, and adaptive voting).

Let’s walk through the required imports to understand the structure of the library:

```python
# example.py

import logicsponge.core as ls
from logicsponge.processmining.algorithms_and_structures import Bag, FrequencyPrefixTree, NGram
from logicsponge.processmining.models import BasicMiner, SoftVoting
from logicsponge.processmining.streaming import IteratorStreamer, StreamingActivityPredictor
from logicsponge.processmining.test_data import dataset
```

This imports algorithms like frequency prefix trees and n-grams. These classes also allow you to define your own data structures.

You will then import models:
- `BasicMiner` wraps a single algorithm (e.g., an n-gram) to produce a predictor model.
- `SoftVoting` (and other ensemble methods) takes a list of models and produces a new model that applies soft voting.

Instances of these classes are ready for batch learning. To use them in streaming mode, wrap them with `StreamingActivityPredictor`. Below, we define two models:
- The first is a 6-gram (look-back window size of 5).
- The second combines several algorithms using soft voting.

By configuring `"include_stop": False`, stop predictions are ignored, and probabilities are normalized. This is often suitable in streaming settings unless explicit stop activities are present.

```python
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
```

Next, we set up the sponge to stream data from a dataset and apply a model. For clarity, a key filter is applied first.  

The dataset can be any iterator. For illustration, we use the **Sepsis dataset** available at [4TU.ResearchData](https://data.4tu.nl/datasets/33632f3c-5c48-40cf-8d8f-2db57f5a6ce7). When you run the Python script, you will be prompted to download it.


```python
streamer = IteratorStreamer(data_iterator=dataset)

sponge = (
    streamer
    * ls.KeyFilter(keys=["case_id", "activity"])
    * model2
    * ls.AddIndex(key="index", index=1)
    * ls.Print()
)


sponge.start()
```

A single prediction might look like this. In addition to the actual case_id and activity, it provides:
- The most likely predicted activity.
- The top-3 activities.
- The probability distribution over all possible activities.

```python
{
    'case_id': 'FAA',
    'activity': 'Return ER',
    'prediction': {
        'activity': 'Return ER',
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
```
