"""Visual GUI: watch the real fly-larva connectome reservoir play Tetris in
real time, with a live view of the trained readout's action logits (i.e.
what the "brain" is currently leaning toward).

Run: python scripts/play_gui.py
Options:
  --readout PATH     .npz with W,b (default: results/stage4_best_readout.npz)
  --random            ignore any saved readout, use a fresh random one
  --max-ticks N       episode length cap (default 300)
  --decision-window-ms N   simulated ms per tick (default 20)
  --cell-size N       pixel size per board cell (default 32)

Controls: ESC or close window to quit.
"""
import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np
import pygame
from brian2 import SpikeMonitor, ms, mV

from data_loader import load_full_dataset
from network import build_lif_network
from encode import build_input_assignment, encode_board_to_current
from decode import decode_action, init_random_readout
from tetris_env import TetrisEnv, INDEX_TO_PIECE, N_ACTIONS

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_READOUT = ROOT / "results" / "stage4_best_readout.npz"

PIECE_COLORS = {
    "I": (0, 240, 240),
    "O": (240, 240, 0),
    "T": (160, 0, 240),
    "S": (0, 240, 0),
    "Z": (240, 0, 0),
    "J": (0, 0, 240),
    "L": (240, 160, 0),
}
BG_COLOR = (18, 18, 24)
GRID_COLOR = (50, 50, 60)
EMPTY_COLOR = (30, 30, 38)
TEXT_COLOR = (230, 230, 235)
ACTION_NAMES = ["noop", "left", "right", "rotate", "drop"]
ACTION_BAR_COLOR = (90, 160, 240)
ACTION_BAR_ACTIVE = (240, 200, 60)


def parse_args():
    p = argparse.ArgumentParser(description="Fly connectome plays Tetris — live GUI")
    p.add_argument("--readout", type=str, default=str(DEFAULT_READOUT))
    p.add_argument("--random", action="store_true")
    p.add_argument("--max-ticks", type=int, default=300)
    p.add_argument("--decision-window-ms", type=float, default=20.0)
    p.add_argument("--cell-size", type=int, default=32)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def load_readout(args, n_output):
    if not args.random and Path(args.readout).exists():
        data = np.load(args.readout)
        print(f"Loaded trained readout from {args.readout}")
        return data["W"], data["b"]
    print("Using a fresh random (untrained) readout.")
    return init_random_readout(n_output_neurons=n_output, seed=args.seed)


def draw_board(screen, env, cell_size, origin):
    ox, oy = origin
    obs = env._obs()
    for r in range(env.height):
        for c in range(env.width):
            val = obs[r, c]
            rect = pygame.Rect(ox + c * cell_size, oy + r * cell_size, cell_size, cell_size)
            if val:
                piece = INDEX_TO_PIECE.get(int(val))
                color = PIECE_COLORS.get(piece, (200, 200, 200))
                pygame.draw.rect(screen, color, rect)
                pygame.draw.rect(screen, BG_COLOR, rect, width=1)
            else:
                pygame.draw.rect(screen, EMPTY_COLOR, rect)
                pygame.draw.rect(screen, GRID_COLOR, rect, width=1)


def draw_sidebar(screen, font, small_font, origin, width, stats, logits):
    ox, oy = origin
    y = oy
    lines = [
        f"tick: {stats['tick']}",
        f"reward (this tick): {stats['reward']:+.3f}",
        f"total reward: {stats['total_reward']:+.3f}",
        f"lines cleared: {stats['lines_cleared']}",
        f"last action: {ACTION_NAMES[stats['action']]}",
    ]
    for line in lines:
        surf = font.render(line, True, TEXT_COLOR)
        screen.blit(surf, (ox, y))
        y += 26

    y += 10
    surf = small_font.render("readout action logits (DN spike-driven):", True, TEXT_COLOR)
    screen.blit(surf, (ox, y))
    y += 22

    max_abs = max(1e-6, np.max(np.abs(logits)))
    bar_w_max = width - 20
    for i, name in enumerate(ACTION_NAMES):
        is_chosen = i == stats["action"]
        norm = logits[i] / max_abs  # in [-1, 1]
        bar_w = int(abs(norm) * (bar_w_max / 2))
        bar_x = ox + bar_w_max // 2
        color = ACTION_BAR_ACTIVE if is_chosen else ACTION_BAR_COLOR
        if norm >= 0:
            rect = pygame.Rect(bar_x, y, bar_w, 16)
        else:
            rect = pygame.Rect(bar_x - bar_w, y, bar_w, 16)
        pygame.draw.rect(screen, color, rect)
        label = small_font.render(name, True, TEXT_COLOR)
        screen.blit(label, (ox, y - 1))
        y += 22


def main():
    args = parse_args()

    print("Loading real connectome...")
    graph, ann, sensory_ids, dn_ids = load_full_dataset()

    print("Building Brian2 reservoir...")
    built = build_lif_network(graph, sensory_ids, dn_ids, seed=args.seed)
    G = built["neurons"]
    net = built["network"]
    n = built["n_neurons"]
    output_indices = built["output_indices"]

    W, b = load_readout(args, n_output=len(output_indices))

    env = TetrisEnv(seed=args.seed)
    obs = env.reset()
    input_groups = build_input_assignment(built["input_indices"], obs.shape, seed=args.seed)

    spikemon = SpikeMonitor(G)
    net.add(spikemon)
    rng = np.random.default_rng(args.seed)

    pygame.init()
    pygame.display.set_caption("Fly Larva Connectome plays Tetris")
    cell = args.cell_size
    board_w, board_h = env.width * cell, env.height * cell
    sidebar_w = 260
    margin = 20
    win_w = margin * 3 + board_w + sidebar_w
    win_h = margin * 2 + board_h
    screen = pygame.display.set_mode((win_w, win_h))
    font = pygame.font.SysFont("consolas", 18)
    small_font = pygame.font.SysFont("consolas", 15)
    clock = pygame.time.Clock()

    board_origin = (margin, margin)
    sidebar_origin = (margin * 2 + board_w, margin)

    prev_spike_count = 0
    total_reward = 0.0
    ticks = 0
    stats = {"tick": 0, "reward": 0.0, "total_reward": 0.0, "lines_cleared": 0, "action": 0}
    logits = np.zeros(N_ACTIONS)

    running = True
    done = False
    print("Starting GUI loop. Close the window or press ESC to quit.")
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                running = False

        if not running:
            break

        if not done and ticks < args.max_ticks:
            currents = encode_board_to_current(n, obs, input_groups, rng)
            G.I = currents * mV
            net.run(args.decision_window_ms * ms)

            all_i = np.asarray(spikemon.i)
            new_i = all_i[prev_spike_count:]
            prev_spike_count = len(all_i)
            counts = np.zeros(n)
            if len(new_i):
                bc = np.bincount(new_i, minlength=n)
                counts[: len(bc)] = bc
            output_counts = counts[output_indices]

            logits = W @ output_counts + b
            action = decode_action(output_counts, W, b)

            obs, reward, done, info = env.step(action)
            total_reward += reward
            ticks += 1

            stats = {
                "tick": ticks,
                "reward": reward,
                "total_reward": total_reward,
                "lines_cleared": info.get("lines_cleared_this_tick", 0),
                "action": action,
            }

        screen.fill(BG_COLOR)
        draw_board(screen, env, cell, board_origin)
        draw_sidebar(screen, font, small_font, sidebar_origin, sidebar_w, stats, logits)

        if done:
            over = font.render("TOPPED OUT — episode finished", True, (240, 90, 90))
            screen.blit(over, (margin, win_h - 30))
        elif ticks >= args.max_ticks:
            over = font.render("Max ticks reached — episode finished", True, (240, 200, 90))
            screen.blit(over, (margin, win_h - 30))

        pygame.display.flip()
        clock.tick(30)

    pygame.quit()
    print(f"\nFinished: {ticks} ticks, total_reward={total_reward:+.3f}")


if __name__ == "__main__":
    main()
