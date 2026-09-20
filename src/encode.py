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

from tetris_env import PIECE_INDEX

BASELINE_MEAN = 10.0  # mV — matches the Stage 2 sanity baseline
BASELINE_STD = 3.0
BOARD_CURRENT_BOOST = 15.0  # mV added to a cell's assigned input neurons when occupied
N_PIECE_TYPES = len(PIECE_INDEX)  # extra input channels reserved for the next-piece preview


def build_input_assignment(input_indices: np.ndarray, board_shape: tuple[int, int], seed: int = 0):
    """Partition the Sensory input neuron pool across board cells PLUS one
    extra channel per piece type (for the next-piece preview, one-hot), so
    each cell's occupancy — and the upcoming piece's identity — drives a
    distinct subset of input neurons."""
    n_channels = board_shape[0] * board_shape[1] + N_PIECE_TYPES
    if len(input_indices) < n_channels:
        raise ValueError(
            f"Only {len(input_indices)} input neurons available for {n_channels} channels "
            f"({board_shape[0] * board_shape[1]} board cells + {N_PIECE_TYPES} next-piece slots); "
            "reduce board size or include more input celltypes."
        )
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(input_indices))
    shuffled = input_indices[order]
    return np.array_split(shuffled, n_channels)


def encode_state_to_current(
    n_total_neurons: int,
    board: np.ndarray,
    next_piece_name: str,
    input_groups,
    rng: np.random.Generator,
):
    """Returns a length-n_total_neurons current array (in mV, unitless here —
    caller multiplies by mV) with baseline noise everywhere, an extra boost on
    input neurons assigned to occupied board cells, and an extra boost on the
    input neurons assigned to the upcoming piece's one-hot preview channel."""
    currents = rng.normal(BASELINE_MEAN, BASELINE_STD, size=n_total_neurons)
    flat_board = board.flatten()
    n_board_cells = flat_board.size
    for cell_idx, occupied in enumerate(flat_board):
        if occupied:
            currents[input_groups[cell_idx]] += BOARD_CURRENT_BOOST

    next_piece_slot = n_board_cells + (PIECE_INDEX[next_piece_name] - 1)
    currents[input_groups[next_piece_slot]] += BOARD_CURRENT_BOOST
    return currents
