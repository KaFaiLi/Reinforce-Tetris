import numpy as np
import pytest

from tetris_rl.env import TetrisPlacementEnv, board_features
from tetris_rl.game import (
    BOARD_HEIGHT,
    BOARD_WIDTH,
    PIECE_ROTATIONS,
    TetrisGame,
    bumpiness,
    column_heights,
    count_holes,
)


def test_rotation_counts():
    expected = {"I": 2, "O": 1, "T": 4, "S": 2, "Z": 2, "J": 4, "L": 4}
    for name, count in expected.items():
        assert len(PIECE_ROTATIONS[name]) == count


def test_board_metrics():
    board = np.zeros((BOARD_HEIGHT, BOARD_WIDTH), dtype=np.uint8)
    board[BOARD_HEIGHT - 1, :] = 1          # full bottom row
    board[BOARD_HEIGHT - 3, 0] = 1          # column 0: filled, hole below
    heights = column_heights(board)
    assert heights[0] == 3
    assert all(heights[1:] == 1)
    assert count_holes(board) == 1          # gap under the column-0 overhang
    assert bumpiness(board) == 2            # only the 3->1 step


def test_line_clear_and_scoring():
    game = TetrisGame(seed=1)
    # Fill the bottom row except one I-piece slot, then clear it manually.
    game.board[BOARD_HEIGHT - 1, :6] = 1
    game.board[BOARD_HEIGHT - 1, 6:] = 0
    game.piece_name = "I"
    game.rotation = 0
    cleared = game.apply_placement(0, 6)
    assert cleared == 1
    assert game.lines == 1
    assert game.score == 40                  # single at level 0
    assert (game.board == 0).all()           # the only filled row was cleared


def test_placement_enumeration_on_empty_board():
    game = TetrisGame(seed=0)
    game.piece_name = "I"
    game.rotation = 0
    placements = game.legal_placements()
    # Horizontal I: 7 columns; vertical I: 10 columns.
    assert len(placements) == 17
    for p in placements:
        assert p.lines_cleared == 0
        assert p.board_after.sum() > 0


def test_env_step_and_reward():
    env = TetrisPlacementEnv(seed=3)
    feats = env.reset()
    assert feats.ndim == 2 and feats.shape[1] == 4
    reward, done, info = env.step(0)
    assert reward == pytest.approx(1.0)      # no lines on an empty board
    assert not done
    assert info["pieces"] == 1


def test_env_full_random_episode_terminates():
    env = TetrisPlacementEnv(seed=7)
    feats = env.reset()
    rng = np.random.default_rng(7)
    done = False
    steps = 0
    while not done and steps < 5_000:
        reward, done, info = env.step(int(rng.integers(len(feats))))
        if not done:
            feats = env.candidate_features()
        steps += 1
    assert done, "random play should eventually top out"
    assert info["pieces"] == steps


def test_board_features_vector():
    board = np.zeros((BOARD_HEIGHT, BOARD_WIDTH), dtype=np.uint8)
    board[BOARD_HEIGHT - 1, 0] = 1
    feats = board_features(board, lines_cleared=2)
    assert feats.tolist() == [2.0, 0.0, 1.0, 1.0]
