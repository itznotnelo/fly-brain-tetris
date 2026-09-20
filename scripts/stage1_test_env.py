"""Stage 1 verification: unit-test the standalone Tetris environment (no brain
involved) per the approved plan. Run: python scripts/stage1_test_env.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np
from tetris_env import TetrisEnv, ACTION_NOOP, ACTION_LEFT, ACTION_RIGHT, ACTION_ROTATE, ACTION_DROP


def test_reset_shape():
    env = TetrisEnv(seed=0)
    obs = env.reset()
    assert obs.shape == (env.height, env.width)
    assert obs.sum() > 0  # spawned piece occupies some cells
    print("test_reset_shape: OK")


def test_piece_falls_with_noop():
    env = TetrisEnv(seed=1)
    env.reset()
    row0 = env.piece_row
    env.step(ACTION_NOOP)
    assert env.piece_row == row0 + 1, "piece should fall by one row per tick"
    print("test_piece_falls_with_noop: OK")


def test_left_right_bounds():
    env = TetrisEnv(seed=2)
    env.reset()
    # Push left repeatedly past the wall; should clamp, not error or wrap.
    for _ in range(20):
        env.step(ACTION_LEFT)
    assert env.piece_col >= 0
    for _ in range(40):
        env.step(ACTION_RIGHT)
    cells = env._current_cells()
    max_col = max(c for _, c in cells)
    assert env.piece_col + max_col <= env.width - 1
    print("test_left_right_bounds: OK")


def test_rotate_does_not_crash():
    env = TetrisEnv(seed=3)
    env.reset()
    for _ in range(8):
        env.step(ACTION_ROTATE)
    print("test_rotate_does_not_crash: OK")


def test_line_clear():
    env = TetrisEnv(width=4, height=8, seed=4)
    env.reset()
    # Manually fill the bottom row except last column, then drop an I-piece
    # (rotated vertical would not fill it; instead directly manipulate board
    # to deterministically test clear logic).
    env.board[:, :] = 0
    env.board[env.height - 1, :3] = 1  # 3 of 4 columns filled
    # Force-place a piece occupying the missing cell then lock it manually.
    env.piece_name = "O"
    env.rotation_idx = 0
    env.piece_row = env.height - 2
    env.piece_col = 2  # occupies cols 2,3 rows height-2,height-1
    cells = env._current_cells()
    env._lock_piece(cells)
    n = env._clear_lines()
    assert n == 1, f"expected 1 line cleared, got {n}"
    print("test_line_clear: OK")


def test_episode_runs_to_completion():
    env = TetrisEnv(seed=5)
    env.reset()
    rng = np.random.default_rng(5)
    steps = 0
    total_reward = 0.0
    while not env.done and steps < 2000:
        action = rng.integers(0, 5)
        obs, reward, done, info = env.step(int(action))
        total_reward += reward
        steps += 1
    assert steps > 0
    print(f"test_episode_runs_to_completion: OK ({steps} steps, total_reward={total_reward:.2f}, done={env.done})")


if __name__ == "__main__":
    test_reset_shape()
    test_piece_falls_with_noop()
    test_left_right_bounds()
    test_rotate_does_not_crash()
    test_line_clear()
    test_episode_runs_to_completion()
    print("\nAll Stage 1 tests passed.")
