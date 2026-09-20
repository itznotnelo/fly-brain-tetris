"""Renders a trained readout playing an episode, and plots real-connectome vs.
random-control-graph fitness curves for the head-to-head comparison that is
the actual scientific point of this project (does the real fly wiring help
over a generic random reservoir of the same size/degree/weight stats?)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from train import run_episode


def render_episode(graph, sensory_ids, dn_ids, W, b, env_seed=0, network_seed=0, encode_seed=0,
                    decision_window_ms=20.0, max_ticks=60):
    """Runs one episode and returns (total_reward, ticks, action_trace, final_ascii_board)."""
    total_reward, ticks, actions, env = run_episode(
        graph, sensory_ids, dn_ids, W, b,
        env_seed=env_seed, network_seed=network_seed, encode_seed=encode_seed,
        decision_window_ms=decision_window_ms, max_ticks=max_ticks,
    )
    return total_reward, ticks, actions, env.render_ascii()


def plot_comparison(real_history, control_history, out_path: Path):
    fig, ax = plt.subplots(figsize=(8, 5))

    real_gens = [h["gen"] for h in real_history]
    real_best = [h["best_reward"] for h in real_history]
    control_gens = [h["gen"] for h in control_history]
    control_best = [h["best_reward"] for h in control_history]

    ax.plot(real_gens, real_best, marker="o", color="tab:blue", label="real fly connectome (best)")
    ax.plot(control_gens, control_best, marker="s", color="tab:orange",
            label="degree-matched random control (best)")
    ax.set_xlabel("generation")
    ax.set_ylabel("episode reward")
    ax.set_title("Real connectome vs. random-graph control: readout training")
    ax.legend()
    fig.savefig(out_path, dpi=120)
    return out_path
