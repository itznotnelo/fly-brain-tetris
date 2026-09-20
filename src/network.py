"""Builds a fixed (non-trainable) Brian2 leaky-integrate-and-fire spiking
network directly from a connectome graph (real fly connectome, or the
degree-matched random control graph). Synaptic weights come straight from
synapse counts in the data; the network's connectivity is never trained.

MODELING ASSUMPTION (explicit, documented — see plan doc "Data source" caveat):
The connectome data has no excitatory/inhibitory sign per synapse or per
neuron. We assign each neuron a fixed Dale's-law-consistent sign (all of a
neuron's outgoing synapses are either excitatory or inhibitory) via a seeded
random draw, at an ~70% excitatory / 30% inhibitory split — a plausible but
NOT measured ratio for insect brains. This is a deliberate simplification,
not a discovered fact about the fly brain.
"""
from __future__ import annotations

import numpy as np
import networkx as nx
from brian2 import (
    NeuronGroup, Synapses, SpikeMonitor, StateMonitor, Network,
    ms, mV, prefs, defaultclock, BrianLogger,
)

# No C++ compiler confirmed present on this machine (Stage 0 env check) — use
# the numpy codegen target explicitly rather than relying on Brian2's silent
# auto-fallback-with-warning.
prefs.codegen.target = "numpy"

# We always build and run an explicit Network(...) rather than Brian2's
# implicit "magic" global network. Brian2's unused-object GC check is written
# for the magic-network case and false-positives on every explicit-Network
# object at interpreter shutdown; harmless, so it's suppressed rather than
# left to spam every episode during training.
BrianLogger.suppress_name("unused_brian_object")

EXC_FRACTION = 0.7

LIF_EQS = """
dv/dt = (v_rest - v + I) / tau + sigma * xi * tau**-0.5 : volt (unless refractory)
I : volt
"""
V_REST = 0 * mV
V_THRESH = 20 * mV
V_RESET = 0 * mV
TAU = 10 * ms
REFRACTORY = 3 * ms
# Stochastic membrane noise (standard Brian2 xi idiom) so subthreshold mean
# drive can still produce spontaneous, irregular spiking rather than dead
# silence or a rigid clock-like rate — this is what makes the reservoir's
# baseline dynamics look like plausible spontaneous cortical/insect activity.
SIGMA = 6 * mV


def build_lif_network(
    graph: nx.DiGraph,
    sensory_ids,
    dn_ids,
    seed: int = 0,
    weight_scale: float = 0.15,
):
    """Returns a dict with the Brian2 Network and everything needed to drive
    and read it: neuron group, id<->index maps, input/output index arrays,
    and the per-neuron E/I sign array (for transparency/debugging)."""
    rng = np.random.default_rng(seed)

    node_ids = list(graph.nodes)
    id_to_index = {nid: i for i, nid in enumerate(node_ids)}
    n_neurons = len(node_ids)

    sign = np.where(rng.random(n_neurons) < EXC_FRACTION, 1.0, -1.0)

    G = NeuronGroup(
        n_neurons,
        model=LIF_EQS,
        threshold="v > V_THRESH",
        reset="v = V_RESET",
        refractory=REFRACTORY,
        method="euler",
        namespace={
            "v_rest": V_REST,
            "V_THRESH": V_THRESH,
            "V_RESET": V_RESET,
            "tau": TAU,
            "sigma": SIGMA,
        },
    )
    G.v = V_REST

    pre_idx = []
    post_idx = []
    weights = []
    for u, v, data in graph.edges(data=True):
        i, j = id_to_index[u], id_to_index[v]
        w = sign[i] * data["weight"] * weight_scale
        pre_idx.append(i)
        post_idx.append(j)
        weights.append(w)

    S = Synapses(G, G, model="w : volt", on_pre="v_post += w")
    S.connect(i=pre_idx, j=post_idx)
    S.w = np.array(weights) * mV

    input_indices = np.array(
        sorted(id_to_index[nid] for nid in sensory_ids if nid in id_to_index)
    )
    output_indices = np.array(
        sorted(id_to_index[nid] for nid in dn_ids if nid in id_to_index)
    )

    net = Network(G, S)

    return {
        "network": net,
        "neurons": G,
        "synapses": S,
        "id_to_index": id_to_index,
        "index_to_id": node_ids,
        "input_indices": input_indices,
        "output_indices": output_indices,
        "sign": sign,
        "n_neurons": n_neurons,
    }


def build_random_control_graph(graph: nx.DiGraph, seed: int = 0) -> nx.DiGraph:
    """Degree- and weight-distribution-matched random control: shuffles WHO is
    connected to whom while preserving each node's in/out degree sequence and
    reusing the same multiset of synapse-count weights, placed on random
    edges. This isolates whether the real fly wiring pattern matters, vs. just
    having the same size/degree/weight statistics."""
    rng = np.random.default_rng(seed)

    node_ids = list(graph.nodes)
    n = len(node_ids)

    out_degrees = [d for _, d in graph.out_degree()]
    in_degrees = [d for _, d in graph.in_degree()]
    weights = [d["weight"] for _, _, d in graph.edges(data=True)]
    rng.shuffle(weights)

    # Directed configuration model preserves in/out degree sequences.
    random_graph = nx.directed_configuration_model(
        in_degrees, out_degrees, seed=int(rng.integers(0, 2**31 - 1))
    )
    random_graph = nx.DiGraph(random_graph)  # drop parallel edges/self-loops multigraph-ness
    random_graph.remove_edges_from(nx.selfloop_edges(random_graph))

    mapping = {i: node_ids[i] for i in range(n)}
    random_graph = nx.relabel_nodes(random_graph, mapping)

    weight_cycle = iter(weights)
    for u, v in random_graph.edges():
        try:
            w = next(weight_cycle)
        except StopIteration:
            weight_cycle = iter(weights)
            w = next(weight_cycle)
        random_graph[u][v]["weight"] = w

    return random_graph
