"""Decodes DN (descending neuron) spike-count activity, over a decision
window, into a discrete Tetris action via a small trainable linear readout.
This readout is the ONLY thing ever trained — the connectome-derived
recurrent network itself stays fixed."""
from __future__ import annotations

import numpy as np

N_ACTIONS = 5  # noop, left, right, rotate, drop (see tetris_env.py)


def init_random_readout(n_output_neurons: int, n_actions: int = N_ACTIONS, seed: int = 0, scale: float = 0.05):
    rng = np.random.default_rng(seed)
    W = rng.normal(0.0, scale, size=(n_actions, n_output_neurons))
    b = np.zeros(n_actions)
    return W, b


def decode_action(spike_counts_output: np.ndarray, W: np.ndarray, b: np.ndarray) -> int:
    logits = W @ spike_counts_output + b
    return int(np.argmax(logits))


def flatten_readout(W: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.concatenate([W.flatten(), b.flatten()])


def unflatten_readout(flat: np.ndarray, n_actions: int, n_output_neurons: int):
    n_w = n_actions * n_output_neurons
    W = flat[:n_w].reshape(n_actions, n_output_neurons)
    b = flat[n_w:]
    return W, b
