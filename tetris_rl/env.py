"""RL environment over Tetris with placement-level actions.

An action is the index of one of the legal (rotation, column) hard-drop
placements for the current piece. Each candidate placement is described by a
4-feature vector of the resulting board ("afterstate"):

    [lines_cleared, holes, bumpiness, total_height]

A value network scores afterstates; the agent picks the placement whose
afterstate has the highest predicted value. This formulation is far more
sample-efficient for Tetris than frame-level actions.
"""

from __future__ import annotations

import numpy as np

from tetris_rl.game import (
    BOARD_WIDTH,
    TetrisGame,
    bumpiness,
    column_heights,
    count_holes,
)

FEATURE_DIM = 4


def board_features(board: np.ndarray, lines_cleared: int) -> np.ndarray:
    heights = column_heights(board)
    return np.array(
        [
            lines_cleared,
            count_holes(board),
            int(np.abs(np.diff(heights)).sum()),
            int(heights.sum()),
        ],
        dtype=np.float32,
    )


class TetrisPlacementEnv:
    """Single-environment wrapper used directly or inside a worker process."""

    feature_dim = FEATURE_DIM

    def __init__(self, seed: int | None = None, game_over_penalty: float = -5.0):
        self.game = TetrisGame(seed=seed)
        self.game_over_penalty = game_over_penalty
        self._actions: list[tuple[int, int]] = []

    def reset(self) -> np.ndarray:
        self.game.reset()
        return self.candidate_features()

    def candidate_features(self) -> np.ndarray:
        """Feature matrix (num_placements, FEATURE_DIM) for the current piece."""
        placements = self.game.legal_placements()
        self._actions = [(p.rotation, p.col) for p in placements]
        feats = np.stack(
            [board_features(p.board_after, p.lines_cleared) for p in placements]
        )
        return feats

    def step(self, action_index: int):
        """Apply placement ``action_index`` from the last candidate list.

        Returns ``(reward, done, info)``. Reward follows the proven shaping
        ``1 + lines^2 * BOARD_WIDTH`` (survive bonus plus a strong quadratic
        incentive for multi-line clears), with a penalty on game over.
        """
        rotation, col = self._actions[action_index]
        lines = self.game.apply_placement(rotation, col)
        reward = 1.0 + (lines**2) * BOARD_WIDTH
        done = self.game.game_over
        if done:
            reward += self.game_over_penalty
        info = {
            "score": self.game.score,
            "lines": self.game.lines,
            "pieces": self.game.pieces_placed,
        }
        return reward, done, info
