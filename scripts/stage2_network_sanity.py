"""Stage 2 verification: build the Brian2 LIF network straight from the real
connectome, drive it with baseline noise only (no Tetris yet), and confirm
stable, non-exploding spiking activity. Saves a spike-raster PNG.

Run: python scripts/stage2_network_sanity.py
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from brian2 import SpikeMonitor, ms, mV, run

from data_loader import load_full_dataset
from network import build_lif_network

OUT_DIR = Path(__file__).resolve().parent.parent / "results"
OUT_DIR.mkdir(exist_ok=True)


def main():
    print("Loading real connectome...")
    graph, ann, sensory_ids, dn_ids = load_full_dataset()
    print(f"  {graph.number_of_nodes()} neurons, {graph.number_of_edges()} edges")

    print("Building Brian2 LIF network (fixed weights from synapse counts)...")
    t0 = time.time()
    built = build_lif_network(graph, sensory_ids, dn_ids, seed=0)
    print(f"  built in {time.time() - t0:.1f}s")

    G = built["neurons"]
    net = built["network"]
    n = built["n_neurons"]

    # Baseline drive: small constant bias + noise into every neuron, standing
    # in for generic background activity (no Tetris board yet — this stage
    # only checks the reservoir itself is dynamically sane).
    rng = np.random.default_rng(0)
    G.I = (rng.normal(loc=10.0, scale=3.0, size=n)) * mV

    spikemon = SpikeMonitor(G)
    net.add(spikemon)

    duration = 500 * ms
    print(f"Running {duration} of baseline simulation...")
    t0 = time.time()
    net.run(duration)
    print(f"  ran in {time.time() - t0:.1f}s")

    n_spikes = spikemon.num_spikes
    mean_rate_hz = n_spikes / n / (duration / (1000 * ms))
    print(f"\nTotal spikes: {n_spikes}")
    print(f"Mean firing rate across all neurons: {mean_rate_hz:.2f} Hz")

    # Sanity thresholds: not silent, not runaway.
    silent = mean_rate_hz < 0.05
    exploding = mean_rate_hz > 150
    print(f"Silent network: {silent}")
    print(f"Exploding/runaway network: {exploding}")

    input_rate = (
        np.isin(spikemon.i, built["input_indices"]).sum()
        / max(len(built["input_indices"]), 1)
        / (duration / (1000 * ms))
    )
    output_rate = (
        np.isin(spikemon.i, built["output_indices"]).sum()
        / max(len(built["output_indices"]), 1)
        / (duration / (1000 * ms))
    )
    print(f"Mean rate of Sensory (input) neurons: {input_rate:.2f} Hz")
    print(f"Mean rate of DN (output) neurons: {output_rate:.2f} Hz")

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(spikemon.t / ms, spikemon.i, s=1, color="black")
    ax.set_xlabel("time (ms)")
    ax.set_ylabel("neuron index")
    ax.set_title(f"Stage 2 baseline spike raster ({n} neurons, {n_spikes} spikes, {mean_rate_hz:.1f} Hz mean)")
    out_path = OUT_DIR / "stage2_spike_raster.png"
    fig.savefig(out_path, dpi=120)
    print(f"\nSaved raster plot to {out_path}")

    if silent:
        print("\nWARNING: network looks silent under baseline drive — consider raising bias/weight_scale.")
    elif exploding:
        print("\nWARNING: network looks like it's exploding — consider lowering weight_scale.")
    else:
        print("\nStage 2 sanity check PASSED: activity is plausible (not silent, not exploding).")


if __name__ == "__main__":
    main()
