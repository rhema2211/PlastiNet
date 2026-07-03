"""
app.py
Flask API for PlastiNet.

Wraps plasticity.NeuralNetwork with:
  - region mapping (attention / memory / focus)
  - a single in-memory session (fine for a demo/viva; swap for a real
    session store or SQLite if you need multi-user support later)
  - endpoints the frontend visualizer calls after every mini-game round
"""

from flask import Flask, jsonify, request
from flask_cors import CORS

from plasticity import NeuralNetwork, CHILD_MODE, ADULT_MODE, PlasticityParams

app = Flask(__name__)
CORS(app)  # frontend runs on a different port during dev (Live Server)

NUM_NEURONS = 15

# ----------------------------------------------------------------------
# Region mapping: split neurons into three cognitive regions.
# This is what turns "connection weight went up" into a report the user
# can actually understand ("your attention area strengthened").
# Adjust the split if you want uneven region sizes.
# ----------------------------------------------------------------------
def build_regions(num_neurons: int) -> dict[str, list[str]]:
    nodes = [f"N{i}" for i in range(num_neurons)]
    third = num_neurons // 3
    return {
        "attention": nodes[0:third],
        "memory": nodes[third:2 * third],
        "focus": nodes[2 * third:],
    }

REGIONS = build_regions(NUM_NEURONS)

# session state -- created lazily on first /api/session/start call
state = {"net": None, "mode": "child", "history": []}


def region_report(net: NeuralNetwork) -> dict:
    return {region: round(net.region_strength(nodes), 4)
            for region, nodes in REGIONS.items()}


@app.route("/api/session/start", methods=["POST"])
def start_session():
    """(Re)initialize the network. Body: {"mode": "child" | "adult"}"""
    body = request.get_json(silent=True) or {}
    mode = body.get("mode", "child")
    params = ADULT_MODE if mode == "adult" else CHILD_MODE

    state["net"] = NeuralNetwork(num_neurons=NUM_NEURONS, params=params, seed=None)
    state["mode"] = mode
    state["history"] = [region_report(state["net"])]

    return jsonify({
        "snapshot": state["net"].snapshot(),
        "regions": REGIONS,
        "report": state["history"][-1],
        "mode": state["mode"],
    })


@app.route("/api/mode", methods=["POST"])
def set_mode():
    """Switch plasticity mode without resetting the network -- lets you
    demo 'what if this were an adult brain from here on' live."""
    if state["net"] is None:
        return jsonify({"error": "call /api/session/start first"}), 400

    body = request.get_json(silent=True) or {}
    mode = body.get("mode", "child")
    state["net"].params = ADULT_MODE if mode == "adult" else CHILD_MODE
    state["mode"] = mode
    return jsonify({"mode": state["mode"]})


@app.route("/api/train", methods=["POST"])
def train():
    """One training round. Body: {"region": "attention" | "memory" | "focus"}
    Runs a full stimulate -> hebbian -> grow -> prune cycle on that region
    and returns the updated graph + report for the frontend to render."""
    if state["net"] is None:
        return jsonify({"error": "call /api/session/start first"}), 400

    body = request.get_json(silent=True) or {}
    region = body.get("region")
    if region not in REGIONS:
        return jsonify({"error": f"region must be one of {list(REGIONS)}"}), 400

    before = set(state["net"].connections.keys())
    state["net"].train_step(REGIONS[region])
    after = set(state["net"].connections.keys())

    grown = list(after - before)
    pruned = list(before - after)

    report = region_report(state["net"])
    state["history"].append(report)

    return jsonify({
        "snapshot": state["net"].snapshot(),
        "report": report,
        "grown": grown,     # frontend uses these to trigger grow/prune animations
        "pruned": pruned,
        "region_trained": region,
    })


@app.route("/api/history", methods=["GET"])
def history():
    """Full report history, for the 'growth over time' comparison chart."""
    return jsonify({"history": state["history"]})


if __name__ == "__main__":
    app.run(debug=True, port=5000)
