"""Stage 5 verification (and the project's actual scientific payoff): train
the readout on the REAL fly connectome reservoir and on a degree-matched
RANDOM-GRAPH control reservoir with identical hyperparameters, compare their
fitness curves, and render a final episode for each.

Run: python scripts/stage5_comparison.py
"""
import sys
import time
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from data_loader import load_full_dataset
from network import build_lif_network, build_random_control_graph
from train import train_cma
from evaluate import render_episode, plot_comparison

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

GENERATIONS = 5
POPSIZE = 6
MAX_TICKS = 40
DECISION_WINDOW_MS = 20.0


def train_and_report(label, graph, sensory_ids, dn_ids):
    built = build_lif_network(graph, sensory_ids, dn_ids, seed=0)
    n_output = len(built["output_indices"])
    print(f"\n[{label}] output (DN) neurons: {n_output}")

    t0 = time.time()
    best_W, best_b, history = train_cma(
        graph, sensory_ids, dn_ids, n_output=n_output,
        generations=GENERATIONS, popsize=POPSIZE, sigma0=0.05, seed=0,
        max_ticks=MAX_TICKS, decision_window_ms=DECISION_WINDOW_MS,
    )
    print(f"[{label}] training finished in {time.time() - t0:.1f}s")

    reward, ticks, actions, board = render_episode(
        graph, sensory_ids, dn_ids, best_W, best_b,
        env_seed=42, network_seed=42, encode_seed=42,
        decision_window_ms=DECISION_WINDOW_MS, max_ticks=60,
    )
    print(f"[{label}] held-out demo episode: reward={reward:+.3f}, ticks={ticks}")
    print(board)

    return history


def main():
    print("Loading real connectome...")
    graph, ann, sensory_ids, dn_ids = load_full_dataset()

    print("Building degree-matched random control graph (same neurons, shuffled wiring)...")
    control_graph = build_random_control_graph(graph, seed=0)
    print(f"  control graph: {control_graph.number_of_nodes()} nodes, {control_graph.number_of_edges()} edges "
          f"(real graph: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges)")

    real_history = train_and_report("REAL CONNECTOME", graph, sensory_ids, dn_ids)
    control_history = train_and_report("RANDOM CONTROL", control_graph, sensory_ids, dn_ids)

    out_path = plot_comparison(real_history, control_history, RESULTS_DIR / "stage5_comparison.png")
    print(f"\nSaved comparison plot to {out_path}")

    with open(RESULTS_DIR / "stage5_history.json", "w") as f:
        json.dump({"real": real_history, "control": control_history}, f, indent=2)

    real_final = real_history[-1]["best_reward"]
    control_final = control_history[-1]["best_reward"]
    print(f"\nFinal best reward — real connectome: {real_final:+.3f}, random control: {control_final:+.3f}")
    print("(Small-sample, short-training result — not a statistically powered claim; "
          "see README for how to run a longer, more conclusive comparison.)")


if __name__ == "__main__":
    main()
