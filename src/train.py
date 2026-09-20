"""Episode runner (env <-> connectome reservoir <-> readout) and, once Stage 4
is added, the CMA-ES outer loop that trains ONLY the linear readout weights
against Tetris reward. The connectome-derived recurrent network is rebuilt
fresh each episode (cheap, ~0.1-0.2s) and never trained.
"""
from __future__ import annotations

import numpy as np
from brian2 import SpikeMonitor, ms, mV

from network import build_lif_network
from encode import build_input_assignment, encode_board_to_current
from decode import decode_action, init_random_readout, unflatten_readout, N_ACTIONS
from tetris_env import TetrisEnv


def run_episode(
    graph,
    sensory_ids,
    dn_ids,
    readout_W: np.ndarray,
    readout_b: np.ndarray,
    env_seed: int = 0,
    network_seed: int = 0,
    encode_seed: int = 0,
    decision_window_ms: float = 20.0,
    max_ticks: int = 60,
    verbose: bool = False,
):
    """Runs one full Tetris episode driven by the connectome reservoir.
    Returns (total_reward, n_ticks, actions_taken, final_env)."""
    built = build_lif_network(graph, sensory_ids, dn_ids, seed=network_seed)
    G = built["neurons"]
    net = built["network"]
    n = built["n_neurons"]
    output_indices = built["output_indices"]

    if len(output_indices) == 0:
        raise ValueError("No DN output neurons found in this graph — cannot decode actions.")

    env = TetrisEnv(seed=env_seed)
    obs = env.reset()
    input_groups = build_input_assignment(built["input_indices"], obs.shape, seed=encode_seed)

    spikemon = SpikeMonitor(G)
    net.add(spikemon)

    rng = np.random.default_rng(encode_seed)
    total_reward = 0.0
    actions_taken = []
    prev_spike_count = 0
    ticks = 0
    done = False

    while not done and ticks < max_ticks:
        currents = encode_board_to_current(n, obs, input_groups, rng)
        G.I = currents * mV
        net.run(decision_window_ms * ms)

        all_i = np.asarray(spikemon.i)
        new_i = all_i[prev_spike_count:]
        prev_spike_count = len(all_i)

        counts = np.zeros(n)
        if len(new_i):
            bc = np.bincount(new_i, minlength=n)
            counts[: len(bc)] = bc
        output_counts = counts[output_indices]

        action = decode_action(output_counts, readout_W, readout_b)
        actions_taken.append(action)

        obs, reward, done, info = env.step(action)
        total_reward += reward
        ticks += 1

        if verbose:
            print(f"tick {ticks:3d} action={action} reward={reward:+.3f} total={total_reward:+.3f} done={done}")

    return total_reward, ticks, actions_taken, env


# ---------------------------------------------------------------------------
# Stage 4: CMA-ES training of ONLY the readout weights. The reservoir (graph)
# is fixed and shared read-only across worker processes via the pool
# initializer, rebuilt into a fresh Brian2 network for each episode.
# ---------------------------------------------------------------------------

_WORKER_STATE = {}


def _worker_init(graph, sensory_ids, dn_ids, n_output, max_ticks, decision_window_ms):
    _WORKER_STATE.update(
        graph=graph,
        sensory_ids=sensory_ids,
        dn_ids=dn_ids,
        n_output=n_output,
        max_ticks=max_ticks,
        decision_window_ms=decision_window_ms,
    )


def _worker_fitness(args):
    flat, episode_seed, n_actions = args
    st = _WORKER_STATE
    W, b = unflatten_readout(flat, n_actions, st["n_output"])
    total_reward, ticks, actions, env = run_episode(
        st["graph"], st["sensory_ids"], st["dn_ids"], W, b,
        env_seed=episode_seed,
        network_seed=episode_seed + 1_000_000,
        encode_seed=episode_seed + 2_000_000,
        decision_window_ms=st["decision_window_ms"],
        max_ticks=st["max_ticks"],
    )
    return -total_reward  # cma minimizes


def train_cma(
    graph,
    sensory_ids,
    dn_ids,
    n_output: int,
    n_actions: int = N_ACTIONS,
    generations: int = 5,
    popsize: int = 6,
    sigma0: float = 0.05,
    seed: int = 0,
    max_ticks: int = 40,
    decision_window_ms: float = 20.0,
    n_workers: int | None = None,
):
    """CMA-ES optimizes only the flattened readout (W, b) against episode
    reward. Population members are evaluated in parallel worker processes
    (each rebuilding its own Brian2 network per episode — the reservoir
    connectivity itself is never modified/trained)."""
    import cma
    import multiprocessing as mp

    n_params = n_actions * n_output + n_actions
    x0 = np.zeros(n_params)
    es = cma.CMAEvolutionStrategy(x0, sigma0, {"popsize": popsize, "seed": seed})

    n_workers = n_workers or max(1, mp.cpu_count() - 1)
    history = []

    ctx = mp.get_context("spawn")
    with ctx.Pool(
        n_workers,
        initializer=_worker_init,
        initargs=(graph, sensory_ids, dn_ids, n_output, max_ticks, decision_window_ms),
    ) as pool:
        for gen in range(generations):
            solutions = es.ask()
            args = [(sol, gen * 10_000 + i, n_actions) for i, sol in enumerate(solutions)]
            fitnesses = pool.map(_worker_fitness, args)
            es.tell(solutions, fitnesses)

            best_reward = -min(fitnesses)
            mean_reward = -float(np.mean(fitnesses))
            history.append({"gen": gen, "best_reward": best_reward, "mean_reward": mean_reward})
            print(f"gen {gen:3d}  best_reward={best_reward:+.3f}  mean_reward={mean_reward:+.3f}")

    best_flat = es.result.xbest
    best_W, best_b = unflatten_readout(best_flat, n_actions, n_output)
    return best_W, best_b, history
