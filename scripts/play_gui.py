"""Visual GUI: the real fly-larva connectome reservoir plays Tetris
continuously and trains itself as it goes. Each episode played is one CMA-ES
individual; once a full population of episodes has been played, the readout
weights update (CMA-ES tell/ask) and the next generation starts automatically
— it just keeps playing, generation after generation, forever. Whenever an
episode beats the running high score, that readout is saved to disk.

Only the small linear readout is ever trained — the connectome-derived
recurrent reservoir itself is fixed, per the project's reservoir-computing
design (see README.md).

Run: python scripts/play_gui.py
Options:
  --readout PATH        .npz with W,b to warm-start CMA-ES from
                         (default: results/stage4_best_readout.npz)
  --fresh                ignore any saved readout, start from scratch
  --popsize N            episodes (individuals) per generation (default 8)
  --sigma0 N              CMA-ES initial step size (default 0.05)
  --max-ticks N          episode length cap (default 200)
  --decision-window-ms N  simulated ms per tick (default 20)
  --cell-size N          pixel size per board cell (default 32)

Controls: ESC or close window to quit.
"""
import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np
import pygame
import cma
from brian2 import SpikeMonitor, ms, mV

from data_loader import load_full_dataset
from network import build_lif_network
from encode import build_input_assignment, encode_state_to_current
from decode import decode_action, flatten_readout, unflatten_readout, N_ACTIONS
from tetris_env import TetrisEnv, INDEX_TO_PIECE, PIECE_SHAPES

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_READOUT = ROOT / "results" / "stage4_best_readout.npz"
HIGH_SCORE_READOUT = ROOT / "results" / "gui_high_score_readout.npz"

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
HIGH_SCORE_COLOR = (255, 215, 0)
ACTION_NAMES = ["noop", "left", "right", "rotate", "drop"]
ACTION_BAR_COLOR = (90, 160, 240)
ACTION_BAR_ACTIVE = (240, 200, 60)


def parse_args():
    p = argparse.ArgumentParser(description="Fly connectome trains and plays Tetris, forever — live GUI")
    p.add_argument("--readout", type=str, default=str(DEFAULT_READOUT))
    p.add_argument("--fresh", action="store_true")
    p.add_argument("--popsize", type=int, default=8)
    p.add_argument("--sigma0", type=float, default=0.05)
    p.add_argument("--max-ticks", type=int, default=200)
    p.add_argument("--decision-window-ms", type=float, default=20.0)
    p.add_argument("--cell-size", type=int, default=32)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


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


def draw_next_piece_preview(screen, ox, y, piece_name, cell=14):
    """Draws a small 4x4-cell preview of the upcoming piece and returns the
    y-coordinate just below it."""
    box_h = cell * 3
    box_rect = pygame.Rect(ox, y, cell * 4, box_h)
    pygame.draw.rect(screen, EMPTY_COLOR, box_rect)
    pygame.draw.rect(screen, GRID_COLOR, box_rect, width=1)

    if piece_name:
        color = PIECE_COLORS.get(piece_name, (200, 200, 200))
        for dr, dc in PIECE_SHAPES[piece_name]:
            rect = pygame.Rect(ox + dc * cell, y + dr * cell, cell, cell)
            pygame.draw.rect(screen, color, rect)
            pygame.draw.rect(screen, BG_COLOR, rect, width=1)

    return y + box_h + 10


def draw_sidebar(screen, font, small_font, origin, width, stats, logits):
    ox, oy = origin
    y = oy

    hs_surf = font.render(f"HIGH SCORE: {stats['high_score']:+.3f}", True, HIGH_SCORE_COLOR)
    screen.blit(hs_surf, (ox, y))
    y += 32

    lines = [
        f"generation: {stats['generation']}",
        f"individual: {stats['individual']}/{stats['popsize']}",
        f"tick: {stats['tick']}",
        f"episode reward: {stats['total_reward']:+.3f}",
        f"lines cleared: {stats['lines_cleared']}",
        f"last action: {ACTION_NAMES[stats['action']]}",
    ]
    for line in lines:
        surf = small_font.render(line, True, TEXT_COLOR)
        screen.blit(surf, (ox, y))
        y += 22

    y += 6
    surf = small_font.render("next piece (fed to the reservoir too):", True, TEXT_COLOR)
    screen.blit(surf, (ox, y))
    y += 20
    y = draw_next_piece_preview(screen, ox, y, stats["next_piece"])

    if stats["new_high_score_flash"]:
        flash = font.render("NEW HIGH SCORE!", True, HIGH_SCORE_COLOR)
        screen.blit(flash, (ox, y))
        y += 30

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


class EpisodeRunner:
    """Wraps a fresh network + env + readout for exactly one episode. The
    reservoir is rebuilt each episode (cheap, ~0.1-0.2s) and never trained —
    only the readout (W, b) passed in changes between episodes."""

    def __init__(self, graph, sensory_ids, dn_ids, W, b, episode_seed, max_ticks, decision_window_ms):
        self.W = W
        self.b = b
        self.max_ticks = max_ticks
        self.decision_window_ms = decision_window_ms

        self.built = build_lif_network(graph, sensory_ids, dn_ids, seed=episode_seed)
        self.net = self.built["network"]
        self.G = self.built["neurons"]
        self.n = self.built["n_neurons"]
        self.output_indices = self.built["output_indices"]

        self.env = TetrisEnv(seed=episode_seed)
        self.obs = self.env.reset()
        self.input_groups = build_input_assignment(self.built["input_indices"], self.obs.shape, seed=episode_seed)

        self.spikemon = SpikeMonitor(self.G)
        self.net.add(self.spikemon)
        self.rng = np.random.default_rng(episode_seed)

        self.prev_spike_count = 0
        self.total_reward = 0.0
        self.ticks = 0
        self.done = False
        self.last_action = 0
        self.last_lines = 0
        self.logits = np.zeros(N_ACTIONS)

    def step_tick(self):
        if self.done or self.ticks >= self.max_ticks:
            return False

        currents = encode_state_to_current(
            self.n, self.obs, self.env.next_piece_name, self.input_groups, self.rng
        )
        self.G.I = currents * mV
        self.net.run(self.decision_window_ms * ms)

        all_i = np.asarray(self.spikemon.i)
        new_i = all_i[self.prev_spike_count:]
        self.prev_spike_count = len(all_i)
        counts = np.zeros(self.n)
        if len(new_i):
            bc = np.bincount(new_i, minlength=self.n)
            counts[: len(bc)] = bc
        output_counts = counts[self.output_indices]

        self.logits = self.W @ output_counts + self.b
        action = decode_action(output_counts, self.W, self.b)

        self.obs, reward, self.done, info = self.env.step(action)
        self.total_reward += reward
        self.ticks += 1
        self.last_action = action
        self.last_lines = info.get("lines_cleared_this_tick", 0)
        return True

    def finished(self):
        return self.done or self.ticks >= self.max_ticks


def main():
    args = parse_args()

    print("Loading real connectome...")
    graph, ann, sensory_ids, dn_ids = load_full_dataset()

    print("Sizing readout dimensions...")
    probe = build_lif_network(graph, sensory_ids, dn_ids, seed=args.seed)
    n_output = len(probe["output_indices"])
    n_params = N_ACTIONS * n_output + N_ACTIONS

    x0 = np.zeros(n_params)
    if not args.fresh and Path(args.readout).exists():
        data = np.load(args.readout)
        x0 = flatten_readout(data["W"], data["b"])
        print(f"Warm-starting CMA-ES from {args.readout}")
    else:
        print("Starting CMA-ES from scratch (zero-mean readout).")

    es = cma.CMAEvolutionStrategy(x0, args.sigma0, {"popsize": args.popsize, "seed": args.seed})

    pygame.init()
    pygame.display.set_caption("Fly Larva Connectome trains & plays Tetris")
    cell = args.cell_size
    dummy_env = TetrisEnv()
    board_w, board_h = dummy_env.width * cell, dummy_env.height * cell
    sidebar_w = 280
    margin = 20
    win_w = margin * 3 + board_w + sidebar_w
    win_h = margin * 2 + board_h
    screen = pygame.display.set_mode((win_w, win_h))
    font = pygame.font.SysFont("consolas", 18)
    small_font = pygame.font.SysFont("consolas", 15)
    clock = pygame.time.Clock()

    board_origin = (margin, margin)
    sidebar_origin = (margin * 2 + board_w, margin)

    high_score = -1e9
    high_score_flash_ticks = 0

    generation = 0
    solutions = es.ask()
    fitnesses = []
    individual_idx = 0
    # Same episode seed (piece sequence, network noise) for every individual
    # within a generation, varying only across generations — otherwise
    # "who got an easier piece sequence" swamps "whose readout is better."
    episode_seed = args.seed + generation

    def new_runner(flat_params, seed):
        W, b = unflatten_readout(flat_params, N_ACTIONS, n_output)
        return EpisodeRunner(graph, sensory_ids, dn_ids, W, b, seed, args.max_ticks, args.decision_window_ms)

    runner = new_runner(solutions[individual_idx], episode_seed)

    running = True
    print("Starting continuous train+play loop. Close the window or press ESC to quit.")
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                running = False
        if not running:
            break

        if not runner.finished():
            runner.step_tick()
        else:
            # Episode over: record fitness, check high score, advance.
            fitnesses.append(-runner.total_reward)
            if runner.total_reward > high_score:
                high_score = runner.total_reward
                np.savez(HIGH_SCORE_READOUT, W=runner.W, b=runner.b)
                high_score_flash_ticks = 60  # ~2s at 30fps
                print(f"New high score: {high_score:+.3f} (gen {generation}, individual {individual_idx})")

            individual_idx += 1

            if individual_idx < len(solutions):
                runner = new_runner(solutions[individual_idx], episode_seed)
            else:
                es.tell(solutions, fitnesses)
                best_gen = -min(fitnesses)
                mean_gen = -float(np.mean(fitnesses))
                print(f"generation {generation:4d} complete — best={best_gen:+.3f} mean={mean_gen:+.3f} "
                      f"high_score={high_score:+.3f}")
                generation += 1
                episode_seed = args.seed + generation
                solutions = es.ask()
                fitnesses = []
                individual_idx = 0
                runner = new_runner(solutions[individual_idx], episode_seed)

        if high_score_flash_ticks > 0:
            high_score_flash_ticks -= 1

        stats = {
            "generation": generation,
            "individual": individual_idx + 1,
            "popsize": len(solutions),
            "tick": runner.ticks,
            "total_reward": runner.total_reward,
            "lines_cleared": runner.last_lines,
            "action": runner.last_action,
            "high_score": high_score,
            "new_high_score_flash": high_score_flash_ticks > 0,
            "next_piece": runner.env.next_piece_name,
        }

        screen.fill(BG_COLOR)
        draw_board(screen, runner.env, cell, board_origin)
        draw_sidebar(screen, font, small_font, sidebar_origin, sidebar_w, stats, runner.logits)
        pygame.display.flip()
        clock.tick(30)

    pygame.quit()
    print(f"\nStopped. Final high score: {high_score:+.3f} (saved to {HIGH_SCORE_READOUT})")


if __name__ == "__main__":
    main()
