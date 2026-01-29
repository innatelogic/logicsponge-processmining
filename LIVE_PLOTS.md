# Live Plot Feature for Active Model Trace

## Overview
The `--live-plots` flag enables a live web dashboard that visualizes the evolution of the active model index during streaming prediction. This allows you to see in real-time which model is selected at each step of the prediction process.

## Usage

Run the streaming prediction with the `--live-plots` flag:

```bash
python examples/predict_streaming.py --data Sepsis_Cases --live-plots
```

## Access the Dashboard

Once the script is running and the first data is being processed, open your web browser and navigate to:

```
http://localhost:5000
```

The dashboard will:
- Automatically refresh every 5 seconds to show the latest data
- Display one plot per strategy (adaptive voting variants and promotion variants)
- Show the evolution of the active model index (y-axis) over event index (x-axis)
- Help visualize model selection patterns and promotions

## What You'll See

For each strategy, you'll see a plot showing:
- **X-axis**: Event index (position in the event stream)
- **Y-axis**: Active model index (which model in the ensemble is selected)
- **Line plot**: Evolution of model selection over time

For example, in a Promotion strategy, you might see the model index stay at 0 initially, then jump to 1 once the second model outperforms the first sufficiently, visualizing the promotion process.

## Requirements

For live plots to work, Flask and Matplotlib must be installed:

```bash
pip install flask matplotlib
```

If these packages are not installed, the live plot server will not start, but the streaming prediction will continue normally.

## Performance Notes

- The Flask server runs in a background daemon thread
- Plot generation happens on-demand for each page refresh
- CSV file reads are efficient even as the file grows
- The server has minimal overhead on the main prediction process
