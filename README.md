# Fly Larva Connectome → Tetris

A reservoir-computing experiment: can the *structural wiring* of a real
insect brain support a decodable policy for a simplified Tetris, and does
that real wiring do any better than a random graph with the same size/degree
statistics?

## What this actually is (and isn't)

This is **not** "load a fly brain and it plays Tetris." A connectome is a
wiring diagram — which neurons synapse onto which, and how many synapses —
not a set of weights trained for any task. What's implemented here is a
**reservoir computing / liquid state machine** design:

- The connectome-derived spiking network (leaky-integrate-and-fire neurons,
  synapse-count-weighted connections) is a **fixed, never-trained** recurrent
  reservoir.
- The Tetris board is injected as current into neurons the dataset labels
  `sensory`.
- Spiking activity of neurons labeled `DN-VNC`/`DN-SEZ` (descending
  neurons — the closest thing to motor output in this connectome) is read
  out.
- Only a small **linear readout** (DN spike counts → action) is trained, via
  CMA-ES (an evolutionary strategy, since spikes aren't differentiable).
- A **degree-matched random-graph control** (same neuron count, same in/out
  degree sequence, same multiset of synapse weights, but rewired at random)
  is trained identically, to test whether the *real* fly wiring helps over a
  generic random reservoir of the same statistics.

## Data source

Winding et al. 2023, *Science*, ["The connectome of an insect
brain"](https://www.science.org/doi/10.1126/science.add9330) — the full
larval *Drosophila melanogaster* brain connectome. Official supplementary
data mirrored at
[github.com/brain-networks/larval-drosophila-connectome](https://github.com/brain-networks/larval-drosophila-connectome).
Please cite the paper if you use this data.

**Note on numbers**: the paper's headline figures are ~3,013 neurons and
~544,000 synapses for the whole brain. The flat `all-all_connectivity_matrix.csv`
export used here contains 2,952 neurons, 110,677 directed nonzero
connections, and 352,611 total synapse count — smaller than the headline
totals, most likely due to a synapse-count/region threshold applied in this
particular matrix export. This is reported as-is rather than forced to match;
if you need the literal paper totals, review the raw CATMAID/per-synapse data
directly.

## Explicit modeling assumptions (read this before drawing conclusions)

1. **No excitatory/inhibitory sign in the data.** The connectome has no
   neurotransmitter or sign annotation. Each neuron is assigned a fixed,
   Dale's-law-consistent sign via a *seeded random draw* (70% excitatory /
   30% inhibitory — a plausible but **not measured** ratio). See `network.py`.
2. **"Sensory" input is not fly vision.** Larval *Drosophila* has ~12
   photoreceptors total and essentially no capacity to see a Tetris board.
   The `sensory` class here spans many real modalities (olfactory,
   gustatory, thermosensory, mechanosensory, some visual) and is used purely
   as a generic pool of input channels driven by board-cell occupancy — not
   as a model of fly eyesight. See `encode.py`.
3. **Decision window and board size are simulation conveniences**, not
   biological timescales: a 6×12 board and a 20ms decision window per tick
   were chosen for tractability on a CPU-only, no-C-compiler machine, not
   because they correspond to any real fly behavioral timescale.
4. **The Stage 4/5 runs in this repo are short** (5 generations, population 6)
   — enough to prove the pipeline works and that reward *can* improve, not
   enough to draw a statistically powered conclusion about whether the real
   connectome beats the random control. Run `scripts/stage5_comparison.py`
   with larger `GENERATIONS`/`POPSIZE` (and ideally multiple random seeds per
   condition) for anything more conclusive.

## Setup

```
python -m venv .venv
.venv\Scripts\pip install -r requirements.txt
```

The connectome data itself (~157MB) is **not** committed to this repo — fetch
it from the official mirror:

```
curl -sL -o data/Supplementary-Data-S1.zip "https://github.com/brain-networks/larval-drosophila-connectome/raw/main/Supplementary-Data-S1.zip"
unzip -o data/Supplementary-Data-S1.zip -d data/Supplementary-Data-S1
```

This should produce `data/Supplementary-Data-S1/Supplementary-Data-S1/*.csv`,
matching the paths `src/data_loader.py` expects.

## Pipeline / staged scripts

Run in order — each is a standalone verification stage:

- `scripts/stage1_test_env.py` — Tetris environment unit tests (no brain).
- `scripts/stage2_network_sanity.py` — builds the real connectome reservoir,
  drives it with baseline noise, checks spiking is plausible (not silent,
  not exploding). Saves `results/stage2_spike_raster.png`.
- `scripts/stage3_dry_run.py` — runs one full Tetris episode end-to-end
  through the reservoir with a random (untrained) readout.
- `scripts/stage4_train.py` — short CMA-ES training run on the real
  connectome only. Saves `results/stage4_reward_curve.png`.
- `scripts/stage5_comparison.py` — trains real-connectome and random-control
  reservoirs with identical hyperparameters, renders a demo episode for
  each, and saves the comparison plot `results/stage5_comparison.png`.

## Module layout

```
src/
  data_loader.py   loads the connectome edge list + cell-type annotations
  tetris_env.py    standalone Tetris environment (6x12 board, 5 actions)
  network.py       builds the fixed Brian2 LIF reservoir from a graph;
                    also builds the degree-matched random control graph
  encode.py        board state -> per-tick current injection (Sensory pool)
  decode.py        DN spike counts -> trainable linear readout -> action
  train.py         run_episode() + CMA-ES outer loop (parallelized)
  evaluate.py       episode rendering + real-vs-control comparison plot
```
