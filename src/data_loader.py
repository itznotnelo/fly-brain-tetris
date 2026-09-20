"""Loads the Winding et al. 2023 (Science) larval Drosophila brain connectome and
exposes it as a networkx.DiGraph, plus the Sensory (input) and DN (output) neuron
ID sets needed for the reservoir-computing pipeline.

Data source: https://github.com/brain-networks/larval-drosophila-connectome
(official Supplementary-Data-S1.zip from the paper).

Known caveat (see plan doc): the annotations file has no excitatory/inhibitory
sign per synapse. We do not silently invent one here — sign assignment is left
to network.py as an explicit, documented modeling choice, not baked into loading.
"""
from __future__ import annotations

import pandas as pd
import networkx as nx

DATA_DIR = None  # set by caller / resolved relative to this file by default


def _default_data_dir():
    from pathlib import Path
    return (
        Path(__file__).resolve().parent.parent
        / "data"
        / "Supplementary-Data-S1"
        / "Supplementary-Data-S1"
    )


def load_annotations(data_dir=None) -> pd.DataFrame:
    """Return a long-format DataFrame: one row per neuron id, with its celltype.

    The raw annotations.csv is one row per bilateral *pair* (left_id, right_id),
    with 'no pair' where a neuron is unilateral. We unpivot that into one row
    per actual neuron id.
    """
    data_dir = data_dir or _default_data_dir()
    ann = pd.read_csv(data_dir / "annotations.csv")

    long_rows = []
    for _, row in ann.iterrows():
        for side_col in ("left_id", "right_id"):
            val = row[side_col]
            if pd.isna(val) or str(val).strip().lower() == "no pair":
                continue
            long_rows.append(
                {
                    "neuron_id": str(val).strip(),
                    "celltype": row["celltype"],
                    "additional_annotations": row["additional_annotations"],
                }
            )
    return pd.DataFrame(long_rows).drop_duplicates(subset="neuron_id")


def load_connectome(data_dir=None) -> nx.DiGraph:
    """Load the full all-all synapse-count-weighted connectivity matrix as a
    directed graph. Node ids are strings (matching annotation ids)."""
    data_dir = data_dir or _default_data_dir()
    mat = pd.read_csv(data_dir / "all-all_connectivity_matrix.csv", index_col=0)
    mat.index = mat.index.astype(str)
    mat.columns = mat.columns.astype(str)

    G = nx.DiGraph()
    G.add_nodes_from(mat.index)

    # Build edges from nonzero entries: row = source (pre-synaptic), col = target
    # (post-synaptic) per the connectome convention used by this matrix export.
    nonzero = mat.stack()
    nonzero = nonzero[nonzero > 0]
    for (src, dst), weight in nonzero.items():
        G.add_edge(src, dst, weight=float(weight))

    return G


def load_full_dataset(data_dir=None):
    """Convenience loader returning (graph, annotations, sensory_ids, dn_ids)."""
    data_dir = data_dir or _default_data_dir()
    graph = load_connectome(data_dir)
    ann = load_annotations(data_dir)

    graph_ids = set(graph.nodes)
    ann_in_graph = ann[ann["neuron_id"].isin(graph_ids)]

    sensory_ids = set(
        ann_in_graph.loc[ann_in_graph["celltype"] == "sensory", "neuron_id"]
    )
    dn_ids = set(
        ann_in_graph.loc[
            ann_in_graph["celltype"].isin(["DN-VNC", "DN-SEZ"]), "neuron_id"
        ]
    )

    return graph, ann, sensory_ids, dn_ids


if __name__ == "__main__":
    graph, ann, sensory_ids, dn_ids = load_full_dataset()

    n_neurons = graph.number_of_nodes()
    n_synapse_edges = graph.number_of_edges()
    total_synapse_weight = sum(d["weight"] for _, _, d in graph.edges(data=True))

    print(f"Neurons in connectivity matrix: {n_neurons}")
    print(f"Directed synaptic edges (nonzero entries): {n_synapse_edges}")
    print(f"Total synapse count (sum of weights): {total_synapse_weight:.0f}")
    print()
    print(f"Annotated neuron ids (unpivoted): {len(ann)}")
    print(f"Annotated ids found in matrix: {ann['neuron_id'].isin(graph.nodes).sum()}")
    print()
    print(f"Sensory (input) neurons found in matrix: {len(sensory_ids)}")
    print(f"DN-VNC + DN-SEZ (output) neurons found in matrix: {len(dn_ids)}")
    print()
    print("Celltype breakdown (of annotated ids found in matrix):")
    ann_in_graph = ann[ann["neuron_id"].isin(graph.nodes)]
    print(ann_in_graph["celltype"].value_counts())
