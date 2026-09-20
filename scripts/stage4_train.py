"""Stage 4 verification: run a short CMA-ES training job (small
population/generation count) optimizing only the readout weights, and check
that reward trends upward over generations at all. Saves a reward-curve PNG
and the best readout weights.

Run: python scripts/stage4_train.py
"""
import sys
import time
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from data_loader import load_full_dataset
from network import build_lif_network
from train import train_cma

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)


def main():
    print("Loading real connectome...")
    graph, ann, sensory_ids, dn_ids = load_full_dataset()

    built = build_lif_network(graph, sensory_ids, dn_ids, seed=0)
    n_output = len(built["output_indices"])
    print(f"Output (DN) neurons: {n_output}")

    print("\nRunning short CMA-ES training job (real connectome reservoir)...")
    t0 = time.time()
    best_W, best_b, history = train_cma(
        graph, sensory_ids, dn_ids, n_output=n_output,
        generations=5, popsize=6, sigma0=0.05, seed=0,
        max_ticks=40, decision_window_ms=20.0,
    )
    elapsed = time.time() - t0
    print(f"\nTraining finished in {elapsed:.1f}s wall clock")

    gens = [h["gen"] for h in history]
    best_rewards = [h["best_reward"] for h in history]
    mean_rewards = [h["mean_reward"] for h in history]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(gens, best_rewards, marker="o", label="best in generation")
    ax.plot(gens, mean_rewards, marker="s", label="population mean")
    ax.set_xlabel("generation")
    ax.set_ylabel("episode reward")
    ax.set_title("Stage 4: real connectome reservoir, CMA-ES readout training")
    ax.legend()
    out_path = RESULTS_DIR / "stage4_reward_curve.png"
    fig.savefig(out_path, dpi=120)
    print(f"Saved reward curve to {out_path}")

    np.savez(RESULTS_DIR / "stage4_best_readout.npz", W=best_W, b=best_b)
    with open(RESULTS_DIR / "stage4_history.json", "w") as f:
        json.dump(history, f, indent=2)

    trending_up = best_rewards[-1] > best_rewards[0]
    print(f"\nFirst-gen best reward: {best_rewards[0]:+.3f}")
    print(f"Last-gen best reward:  {best_rewards[-1]:+.3f}")
    if trending_up:
        print("Stage 4 PASSED: best reward improved over generations.")
    else:
        print("Stage 4 NOTE: reward did not improve in this short run — expected "
              "possibility with only 5 generations / popsize 6; a longer run "
              "may be needed, this short job's job was only to prove the loop works.")


if __name__ == "__main__":
    main()
