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
pm_ngram = NGram(window_length=2)
pm_fpt = FrequencyPrefixTree()

# Dictionary to store all PM models
pm_models: dict[str, NGram | FrequencyPrefixTree] = {"ngram": pm_ngram, "fpt": pm_fpt}

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

# Generate initial elements for both models
initial_elements_ngram = generate_cytoscape_elements(pm_ngram)
initial_elements_fpt = generate_cytoscape_elements(pm_fpt)

app = dash.Dash(__name__)

app.layout = html.Div(
    [
        html.H1("Process Mining Dashboard"),
        dcc.Interval(id="interval-component", interval=1000, n_intervals=0),
        dcc.Store(id="previous-node-count-ngram", data=len(pm_ngram.state_info)),
        dcc.Store(id="previous-node-count-fpt", data=len(pm_fpt.state_info)),
        dcc.Store(id="ngram-elements", data=initial_elements_ngram),
        dcc.Store(id="fpt-elements", data=initial_elements_fpt),
        # Tabs component
        dcc.Tabs(
            id="model-tabs",
            value="ngram",
            children=[
                dcc.Tab(label="NGram", value="ngram"),
                dcc.Tab(label="Frequency Prefix Tree", value="fpt"),
            ],
        ),
        # Cytoscape component that will be updated based on active tab
        html.Div(id="tab-content"),
    ]
)


@app.callback(
    [
        Output("ngram-elements", "data"),
        Output("fpt-elements", "data"),
        Output("interval-component", "disabled"),
        Output("previous-node-count-ngram", "data"),
        Output("previous-node-count-fpt", "data"),
    ],
    [Input("interval-component", "n_intervals")],
    [
        DashState("previous-node-count-ngram", "data"),
        DashState("previous-node-count-fpt", "data"),
    ],
)
def update_models(
    _unused: Any,  # noqa: ANN401
    previous_node_count_ngram: int,
    previous_node_count_fpt: int,
) -> tuple:
    """Update process mining models with new data."""
    try:
        # Fetch the next element from the dataset iterator
        event = next(dataset)
    except StopIteration:
        # Stop the interval if the dataset is exhausted
        return dash.no_update, dash.no_update, True, previous_node_count_ngram, previous_node_count_fpt

    # Update all process mining models with the new data
    for model in pm_models.values():
        model.update(event)

    # Generate updated Cytoscape elements for both models
    elements_ngram = generate_cytoscape_elements(pm_models["ngram"])
    elements_fpt = generate_cytoscape_elements(pm_models["fpt"])

    # Track current node counts
    current_node_count_ngram = len(pm_models["ngram"].state_info)
    current_node_count_fpt = len(pm_models["fpt"].state_info)

    return elements_ngram, elements_fpt, False, current_node_count_ngram, current_node_count_fpt


@app.callback(
    Output("tab-content", "children"),
    [Input("model-tabs", "value"), Input("ngram-elements", "data"), Input("fpt-elements", "data")],
)
def render_tab_content(
    active_tab: str,
    ngram_elements: list,
    fpt_elements: list,
) -> html.Div:
    """Render the content of the selected tab."""
    # Select elements and determine layout based on active tab

    elements = ngram_elements if active_tab == "ngram" else fpt_elements

    # Determine layout based on node count changes
    layout = {"name": "breadthfirst", "directed": True, "animate": True}

    # Return the Cytoscape component for the active tab
    return html.Div(
        [
            cyto.Cytoscape(
                id=f"{active_tab}-graph",
                layout=layout,
                style={"width": "100%", "height": "600px"},
                elements=elements,
                stylesheet=graph_stylesheet,
            )
        ]
    )


if __name__ == "__main__":
    app.run(debug=True)
