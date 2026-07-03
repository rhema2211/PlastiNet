"""
plasticity.py
Core simulation engine for PlastiNet.

Implements:
  - Hebbian learning (functional plasticity): weights strengthen/weaken
  - Structural plasticity: synapses are physically grown or pruned
  - A "plasticity mode" that scales all rates (child/high vs adult/low)

Design: the network is a simple weighted directed graph.
  nodes            -> list of neuron ids, e.g. ["N0", "N1", ...]
  connections      -> dict[(pre, post)] = weight   (0.0 to 1.0)
  activity_trace   -> dict[node] = recent activation level (0.0 to 1.0)

This is intentionally dependency-light (pure Python + a bit of random)
so it's easy to read, defend in a viva, and port to JS for the frontend
visualizer later if you want the simulation running client-side too.
"""

import random
from dataclasses import dataclass, field


@dataclass
class PlasticityParams:
    """Tunable knobs. This dataclass IS your 'high plasticity vs low
    plasticity' switch -- just swap presets, nothing else changes."""
    learning_rate: float = 0.10        # how fast co-activation strengthens a synapse
    decay_rate: float = 0.02           # passive weakening per step (use-it-or-lose-it)
    growth_prob: float = 0.05          # chance to grow a new synapse when conditions are met
    growth_coactivation_threshold: float = 0.6   # how "co-active" two neurons must be to be candidates for a new synapse
    prune_threshold: float = 0.05      # weight below this -> synapse is removed
    max_weight: float = 1.0
    min_initial_weight: float = 0.05   # starting strength of a newly grown synapse


# Two ready-made presets you can literally point to in your report
CHILD_MODE = PlasticityParams(
    learning_rate=0.18, decay_rate=0.01, growth_prob=0.12,
    growth_coactivation_threshold=0.45, prune_threshold=0.03
)

ADULT_MODE = PlasticityParams(
    learning_rate=0.06, decay_rate=0.03, growth_prob=0.02,
    growth_coactivation_threshold=0.75, prune_threshold=0.08
)


class NeuralNetwork:
    def __init__(self, num_neurons: int, params: PlasticityParams = None, seed: int = None):
        if seed is not None:
            random.seed(seed)
        self.params = params or PlasticityParams()
        self.nodes = [f"N{i}" for i in range(num_neurons)]
        self.connections: dict[tuple[str, str], float] = {}
        self.activity: dict[str, float] = {n: 0.0 for n in self.nodes}

        # start with a light random scaffold so there's something to prune/grow from
        for _ in range(num_neurons):
            a, b = random.sample(self.nodes, 2)
            if (a, b) not in self.connections:
                self.connections[(a, b)] = random.uniform(0.1, 0.3)

    # ------------------------------------------------------------------
    # STEP 1: apply a task -> sets which neurons are "active" this round
    # ------------------------------------------------------------------
    def stimulate(self, active_nodes: list[str], intensity: float = 1.0):
        """Call this once per training round with the neurons involved
        in whatever mini-game the user just played (e.g. the 'memory'
        or 'focus' region). Everything else naturally decays."""
        for n in self.nodes:
            if n in active_nodes:
                self.activity[n] = min(1.0, self.activity[n] + intensity)
            else:
                self.activity[n] *= 0.5  # fade for non-active neurons this round

    # ------------------------------------------------------------------
    # STEP 2: Hebbian update (functional plasticity)
    # ------------------------------------------------------------------
    def hebbian_update(self):
        p = self.params
        for (pre, post), w in list(self.connections.items()):
            co_activation = self.activity[pre] * self.activity[post]

            # LTP: strengthen proportional to co-activation
            delta = p.learning_rate * co_activation

            # LTD: passive decay, larger if barely used at all
            delta -= p.decay_rate * (1 - co_activation)

            new_w = max(0.0, min(p.max_weight, w + delta))
            self.connections[(pre, post)] = new_w

    # ------------------------------------------------------------------
    # STEP 3: structural plasticity — grow new synapses
    # ------------------------------------------------------------------
    def grow_connections(self):
        p = self.params
        candidates = []
        for a in self.nodes:
            for b in self.nodes:
                if a == b or (a, b) in self.connections:
                    continue
                co_activation = self.activity[a] * self.activity[b]
                if co_activation >= p.growth_coactivation_threshold:
                    candidates.append((a, b))

        for (a, b) in candidates:
            if random.random() < p.growth_prob:
                self.connections[(a, b)] = p.min_initial_weight

    # ------------------------------------------------------------------
    # STEP 4: structural plasticity — prune weak synapses
    # ------------------------------------------------------------------
    def prune_connections(self):
        p = self.params
        to_remove = [k for k, w in self.connections.items() if w < p.prune_threshold]
        for k in to_remove:
            del self.connections[k]

    # ------------------------------------------------------------------
    # One full training round = the four steps above, in order
    # ------------------------------------------------------------------
    def train_step(self, active_nodes: list[str]):
        self.stimulate(active_nodes)
        self.hebbian_update()
        self.grow_connections()
        self.prune_connections()

    # ------------------------------------------------------------------
    # For your "35% stronger in attention area" style reports
    # ------------------------------------------------------------------
    def region_strength(self, region_nodes: list[str]) -> float:
        """Average weight of all connections touching a given region."""
        weights = [w for (a, b), w in self.connections.items()
                   if a in region_nodes or b in region_nodes]
        return sum(weights) / len(weights) if weights else 0.0

    def snapshot(self) -> dict:
        """JSON-serializable state for sending to the frontend visualizer."""
        return {
            "nodes": self.nodes,
            "edges": [{"source": a, "target": b, "weight": round(w, 3)}
                      for (a, b), w in self.connections.items()],
        }


if __name__ == "__main__":
    # Quick sanity demo: watch connection count evolve over training rounds
    net = NeuralNetwork(num_neurons=10, params=CHILD_MODE, seed=42)
    print(f"Start: {len(net.connections)} connections")

    for round_num in range(20):
        active = random.sample(net.nodes, 3)  # simulate a mini-game touching 3 neurons
        net.train_step(active)

    print(f"After 20 rounds: {len(net.connections)} connections")
    print(f"Attention-region strength (N0-N2): {net.region_strength(['N0','N1','N2']):.3f}")
