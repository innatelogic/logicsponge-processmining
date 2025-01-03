import dash
import dash_cytoscape as cyto
import matplotlib as mpl
from dash import State as DashState
from dash import dcc, html
from dash.dependencies import Input, Output

from logicsponge.processmining.algorithms_and_structures import FrequencyPrefixTree
from logicsponge.processmining.test_data import dataset

mpl.use("Agg")

# ============================================================
# Data preparation
# ============================================================

# Create process mining object
pm = FrequencyPrefixTree()
# pm = NGram(window_length=3)

# Ensure `dataset` is an iterator
dataset = iter(dataset)  # Convert to an iterator if not already one


# ============================================================
# Generate graph
# ============================================================


def generate_cytoscape_elements(pm):
    elements = []
    for state_id in pm.state_info:
        is_initial = "true" if pm.initial_state and pm.initial_state.state_id == state_id else "false"
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
def update_graph(_unused, previous_node_count):
    try:
        # Fetch the next element from the dataset iterator
        event = next(dataset)
    except StopIteration:
        # Stop the interval if the dataset is exhausted
        return dash.no_update, dash.no_update, True, previous_node_count

    # Update the process mining object with the new data
    pm.update(event)

    # Generate updated Cytoscape elements
    elements = generate_cytoscape_elements(pm)

    current_node_count = len(pm.state_info)

    # Adjust layout if node count changes
    if current_node_count > previous_node_count:
        layout = {"name": "breadthfirst", "directed": True, "animate": True}
    else:
        layout = {"name": "preset"}

    return elements, layout, False, current_node_count


if __name__ == "__main__":
    app.run_server(debug=True)
