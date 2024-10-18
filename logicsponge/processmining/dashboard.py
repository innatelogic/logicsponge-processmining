import dash
import dash_cytoscape as cyto
import matplotlib as mpl
from dash import State as DashState
from dash import dcc, html
from dash.dependencies import Input, Output

from logicsponge.processmining.algorithms_and_structures import FrequencyPrefixTree

# from innatelogic.circuits.process_mining.test_data import data
from logicsponge.processmining.test_data import dataset

mpl.use("Agg")


# ============================================================
# Data preparation
# ============================================================

# csv_file = pd.read_csv(data["file_path"], delimiter=data["delimiter"])
#
# dataset = [
#     (
#         handle_keys(data["case_keys"], row),  # Process case_keys
#         handle_keys(data["action_keys"], row),  # Process action_keys
#     )
#     for index, row in csv_file.iterrows()
# ]

pm = FrequencyPrefixTree(depth=1, min_total_visits=1)
# pm = NGram("NGram", window_length=1)


# ============================================================
# Generate graph
# ============================================================


def generate_cytoscape_elements(pm):
    elements = []
    for state_id in pm.state_info:
        # Each node displays total_visits/active_visits
        is_initial = "true" if pm.initial_state and pm.initial_state.state_id == state_id else "false"
        elements.append(
            {
                "data": {
                    "id": str(state_id),
                    "label": f"{pm.state_info[state_id]["total_visits"]}/{pm.state_info[state_id]["active_visits"]}",
                    "initial": is_initial,
                }
            }
        )
    for from_state, transitions in pm.transitions.items():
        for action_name, to_state in transitions.items():
            # Highlight the edge for the last transition
            edge_color = (
                "#FF0000"
                if (
                    pm.last_transition
                    and from_state == pm.last_transition[0]
                    and action_name == pm.last_transition[1]
                    and to_state == pm.last_transition[2]
                )
                else "#53585F"
            )
            elements.append(
                {
                    "data": {"source": str(from_state), "target": str(to_state), "label": action_name},
                    "style": {"line-color": edge_color},
                }
            )
    return elements


# Dash App
app = dash.Dash(__name__)

# Initial elements to show something on load
initial_elements = generate_cytoscape_elements(pm)

app.layout = html.Div(
    [
        dcc.Interval(id="interval-component", interval=1000, n_intervals=0),
        dcc.Store(id="previous-node-count", data=len(pm.state_info)),  # Store to keep track of node count
        cyto.Cytoscape(
            id="automaton-graph",
            layout={
                "name": "breadthfirst",
                "directed": True,
                # 'spacingFactor': 2  # Increase spacing to reduce edge crossings further
            },
            style={"width": "100%", "height": "600px"},
            elements=initial_elements,  # Initially populate with the first set of elements
            stylesheet=[
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
                # Style for the initial state node
                {
                    "selector": 'node[initial="true"]',
                    "style": {
                        "background-color": "#9FB4CD",
                    },
                },
                {
                    "selector": "edge",
                    "style": {
                        "label": "data(label)",
                        "curve-style": "bezier",
                        "target-arrow-shape": "triangle",
                        "target-arrow-size": 20,
                        "target-arrow-color": "#53585F",
                        "line-color": "#53585F",
                        "width": 2,
                        "font-size": "18px",
                        "text-background-color": "white",
                        "text-background-opacity": 1,
                        "text-background-shape": "round-rectangle",
                        "text-background-padding": "2px",
                        "color": "black",  # Font color for the edge labels
                    },
                },
                # Style for the highlighted edge
                {
                    "selector": "edge.highlighted",
                    "style": {"line-color": "#A9514C", "target-arrow-color": "#A9514C", "width": 8, "z-index": 10},
                },
            ],
        ),
    ]
)


@app.callback(
    [
        Output("automaton-graph", "elements"),
        Output("automaton-graph", "layout"),
        Output("interval-component", "disabled"),
        Output("previous-node-count", "data"),
    ],
    [Input("interval-component", "n_intervals")],
    [DashState("previous-node-count", "data")],
)
def update_graph(n, previous_node_count):
    if n >= len(dataset):
        # Disable the interval after processing the dataset
        return dash.no_update, dash.no_update, True, previous_node_count

    case_id, action_name = dataset[n]
    pm.update(case_id, action_name)

    # Generate the updated elements
    elements = generate_cytoscape_elements(pm)

    # Highlight the last transition
    if pm.last_transition:
        current_state, current_action, next_state = pm.last_transition
        elements = [
            {
                "data": {
                    "id": str(state_id),
                    "label": f"{pm.state_info[state_id]["total_visits"]}/{pm.state_info[state_id]["active_visits"]}",
                    "initial": "true" if pm.initial_state and pm.initial_state == state_id else "false",
                }
            }
            for state_id in pm.state_info
        ]
        elements += [
            {
                "data": {"source": str(from_state), "target": str(to_state), "label": action_name},
                "classes": "highlighted"
                if (from_state == current_state and action_name == current_action and to_state == next_state)
                else "",
            }
            for from_state, transitions in pm.transitions.items()
            for action_name, to_state in transitions.items()
        ]

    current_node_count = len(pm.state_info)

    if current_node_count > previous_node_count:
        layout = {"name": "breadthfirst", "directed": True, "animate": True}
    else:
        layout = {"name": "preset"}

    return elements, layout, False, current_node_count


if __name__ == "__main__":
    app.run_server(debug=True)
