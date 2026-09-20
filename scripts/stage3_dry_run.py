"""Stage 3 verification: run one full Tetris episode end-to-end through the
real connectome reservoir with a fixed random readout. Doesn't need to play
well — just needs to run without errors and produce a real action trace.

Run: python scripts/stage3_dry_run.py
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from data_loader import load_full_dataset
from decode import init_random_readout
from train import run_episode


def main():
    print("Loading real connectome...")
    graph, ann, sensory_ids, dn_ids = load_full_dataset()

    # Need output neuron count to size the readout; do a throwaway peek via
    # network build indices count without running a sim.
    from network import build_lif_network
    built = build_lif_network(graph, sensory_ids, dn_ids, seed=0)
    n_output = len(built["output_indices"])
    print(f"Output (DN) neurons: {n_output}")

    W, b = init_random_readout(n_output_neurons=n_output, seed=0)

    print("\nRunning one dry-run episode (fixed random readout, not trained)...")
    t0 = time.time()
    total_reward, ticks, actions, env = run_episode(
        graph, sensory_ids, dn_ids, W, b,
        env_seed=0, network_seed=1, encode_seed=2,
        decision_window_ms=20.0, max_ticks=40, verbose=True,
    )
    print(f"\nEpisode finished in {time.time() - t0:.1f}s wall clock")
    print(f"Ticks: {ticks}, total_reward: {total_reward:.3f}")
    print(f"Action trace: {actions}")
    print("\nFinal board:")
    print(env.render_ascii())
    print("\nStage 3 dry run PASSED: full loop ran end-to-end without errors.")


if __name__ == "__main__":
    main()
