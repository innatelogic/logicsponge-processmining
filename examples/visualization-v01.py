"""Visualize process mining models using Dash and Cytoscape."""

from typing import Any

import dash
import dash_cytoscape as cyto
import matplotlib as mpl
from dash import State as DashState
from dash import dcc, html
from dash.dependencies import Input, Output

from logicsponge.processmining.algorithms_and_structures import FrequencyPrefixTree, NGram
from logicsponge.processmining.test_data import dataset

mpl.use("Agg")

# ============================================================
# Data preparation
# ============================================================

# Create process mining objects
pm_ngram = NGram(window_length=3)
pm_fpt = FrequencyPrefixTree()

# Dictionary to store all PM models
pm_models = {"ngram": pm_ngram, "fpt": pm_fpt}

# Ensure `dataset` is an iterator
dataset = iter(dataset)  # Convert to an iterator if not already one


# ============================================================
# Generate graph
# ============================================================


def generate_cytoscape_elements(pm: NGram | FrequencyPrefixTree) -> list[dict[str, Any]]:
    """Generate Cytoscape elements from the process mining model."""
    elements = []
    for state_id in pm.state_info:
        is_initial = "true" if pm.initial_state and pm.initial_state == state_id else "false"
        elements.append(
            {
                "data": {
                    "id": str(state_id),
                    "label": f"{pm.state_info[state_id]['total_visits']}/{pm.state_info[state_id]['active_visits']}",
                    "initial": is_initial,
                }
            }
        )
    for from_state, transitions in pm.transitions.items():
        for activity_name, to_state in transitions.items():
            edge_color = (
                "#FF0000"
                if (
                    pm.last_transition
                    and from_state == pm.last_transition[0]
                    and activity_name == pm.last_transition[1]
                    and to_state == pm.last_transition[2]
                )
                else "#53585F"
            )
            elements.append(
                {
                    "data": {"source": str(from_state), "target": str(to_state), "label": activity_name},
                    "style": {"line-color": edge_color},
                }
            )
    return elements


# Dash App
app = dash.Dash(__name__)

# Initial elements to show something on load
initial_elements = generate_cytoscape_elements(pm_ngram)

# Stylesheet for the graphs
graph_stylesheet = [
    {
        "selector": "node",
        "style": {
            "label": "data(label)",
            "background-color": "#D9E7EF",
            "color": "black",
            "text-valign": "center",
            "text-halign": "center",
            "font-size": "12px",
            "border-color": "#9FB4CD",
            "border-width": "1.5px",
        },
    },
    {"selector": 'node[initial="true"]', "style": {"background-color": "#9FB4CD"}},
    {
        "selector": "edge",
        "style": {
            "label": "data(label)",
            "curve-style": "bezier",
            "target-arrow-shape": "triangle",
            "target-arrow-size": 20,
            "target-arrow-color": "#53585F",
            "line-color": "#53585F",
            "width": 3,
            "font-size": "18px",
            "text-background-color": "white",
            "text-background-opacity": 1,
            "text-background-shape": "round-rectangle",
            "text-background-padding": "2px",
            "color": "black",
        },
    },
    {
        "selector": "edge.highlighted",
        "style": {
            "width": 3,
            "line-color": "#A9514C",
            "target-arrow-color": "#A9514C",
        },
    },
]

app.layout = html.Div(
    [
        html.H1("Process Mining Dashboard"),
        # Tabs component
        dcc.Tabs(
            id="model-tabs",
            value="ngram",
            children=[
                dcc.Tab(label="NGram", value="ngram"),
                dcc.Tab(label="Frequency Prefix Tree", value="fpt"),
            ],
        ),
        # Content that will be updated based on tab selection
        html.Div(
            id="tab-content",
            children=[
                dcc.Interval(id="interval-component", interval=1000, n_intervals=0),
                dcc.Store(id="previous-node-count-ngram", data=len(pm_ngram.state_info)),
                dcc.Store(id="previous-node-count-fpt", data=len(pm_fpt.state_info)),
                dcc.Store(id="active-model", data="ngram"),
                cyto.Cytoscape(
                    id="automaton-graph",
                    layout={
                        "name": "breadthfirst",
                        "directed": True,
                    },
                    style={"width": "100%", "height": "600px"},
                    elements=initial_elements,
                    stylesheet=graph_stylesheet,
                ),
            ],
        ),
    ]
)


@app.callback(Output("active-model", "data"), [Input("model-tabs", "value")])
def update_active_model(tab_value: str) -> str:
    """Update the active model based on the selected tab."""
    return tab_value


@app.callback(
    [
        Output("automaton-graph", "elements"),
        Output("automaton-graph", "layout"),
        Output("interval-component", "disabled"),
        Output("previous-node-count-ngram", "data"),
        Output("previous-node-count-fpt", "data"),
    ],
    [Input("interval-component", "n_intervals")],
    [
        DashState("previous-node-count-ngram", "data"),
        DashState("previous-node-count-fpt", "data"),
        DashState("active-model", "data"),
    ],
)
def update_graph(
    _unused: Any,  # noqa: ANN401
    previous_node_count_ngram: Any,  # noqa: ANN401
    previous_node_count_fpt: Any,  # noqa: ANN401
    active_model: str,
) -> tuple:
    """Update the graph based on the interval and active model."""
    try:
        # Fetch the next element from the dataset iterator
        event = next(dataset)
    except StopIteration:
        # Stop the interval if the dataset is exhausted
        return dash.no_update, dash.no_update, True, previous_node_count_ngram, previous_node_count_fpt

    # Update all process mining models with the new data
    for model in pm_models.values():
        model.update(event)

    # Get the active model
    pm = pm_models[active_model]

    # Generate updated Cytoscape elements for the active model
    elements = generate_cytoscape_elements(pm)

    # Track node counts for each model
    current_node_count_ngram = len(pm_models["ngram"].state_info)
    current_node_count_fpt = len(pm_models["fpt"].state_info)

    # Determine which previous count to use based on active model
    previous_node_count = previous_node_count_ngram if active_model == "ngram" else previous_node_count_fpt
    current_node_count = current_node_count_ngram if active_model == "ngram" else current_node_count_fpt

    # Adjust layout if node count changes
    if current_node_count > previous_node_count:
        layout = {"name": "breadthfirst", "directed": True, "animate": True}
    else:
        layout = {"name": "preset"}

    return elements, layout, False, current_node_count_ngram, current_node_count_fpt


if __name__ == "__main__":
    app.run(debug=True)
