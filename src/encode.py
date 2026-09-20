"""Encodes a Tetris board observation into per-tick current injection on the
connectome's Sensory-class input neurons.

This is an explicit abstraction, not simulated fly vision: larval Drosophila
has only ~12 photoreceptors (Bolwig's organ) and no real capacity to see a
Tetris board. The Sensory class in this dataset spans many real modalities
(olfactory, gustatory, thermosensory, mechanosensory, and some visual), and
here it is used purely as a generic pool of input channels driven by board
occupancy — not as a model of fly eyesight.
"""
from __future__ import annotations

import numpy as np
from brian2 import mV

BASELINE_MEAN = 10.0  # mV — matches the Stage 2 sanity baseline
BASELINE_STD = 3.0
BOARD_CURRENT_BOOST = 15.0  # mV added to a cell's assigned input neurons when occupied


def build_input_assignment(input_indices: np.ndarray, board_shape: tuple[int, int], seed: int = 0):
    """Partition the Sensory input neuron pool across board cells so each
    cell's occupancy drives a distinct subset of input neurons."""
    n_cells = board_shape[0] * board_shape[1]
    if len(input_indices) < n_cells:
        raise ValueError(
            f"Only {len(input_indices)} input neurons available for {n_cells} board cells; "
            "reduce board size or include more input celltypes."
        )
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(input_indices))
    shuffled = input_indices[order]
    return np.array_split(shuffled, n_cells)


def encode_board_to_current(n_total_neurons: int, board: np.ndarray, input_groups, rng: np.random.Generator):
    """Returns a length-n_total_neurons current array (in mV, unitless here —
    caller multiplies by mV) with baseline noise everywhere and an extra boost
    on input neurons assigned to occupied board cells."""
    currents = rng.normal(BASELINE_MEAN, BASELINE_STD, size=n_total_neurons)
    flat_board = board.flatten()
    for cell_idx, occupied in enumerate(flat_board):
        if occupied:
            neuron_idxs = input_groups[cell_idx]
            currents[neuron_idxs] += BOARD_CURRENT_BOOST
    return currents
