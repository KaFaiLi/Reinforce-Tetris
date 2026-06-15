"""Core Tetris game logic: board, pieces, movement, line clears and scoring.

The game exposes two interfaces:

* A step-wise interface (``move``, ``rotate``, ``tick``, ``soft_drop``,
  ``hard_drop``) used for human play.
* A placement interface (``legal_placements``, ``apply_placement``) used by
  the RL environment, where an action is "drop the current piece at rotation
  R, column C".
"""

from __future__ import annotations

import random
from dataclasses import dataclass

import numpy as np

BOARD_WIDTH = 10
BOARD_HEIGHT = 20

# Classic (NES-style) per-line scores, multiplied by (level + 1).
LINE_SCORES = {0: 0, 1: 40, 2: 100, 3: 300, 4: 1200}
LINES_PER_LEVEL = 10

SHAPES = {
    "I": [[1, 1, 1, 1]],
    "O": [[1, 1], [1, 1]],
    "T": [[0, 1, 0], [1, 1, 1]],
    "S": [[0, 1, 1], [1, 1, 0]],
    "Z": [[1, 1, 0], [0, 1, 1]],
    "J": [[1, 0, 0], [1, 1, 1]],
    "L": [[0, 0, 1], [1, 1, 1]],
}

PIECE_NAMES = list(SHAPES)
PIECE_IDS = {name: i + 1 for i, name in enumerate(PIECE_NAMES)}


def _unique_rotations(shape):
    rotations = []
    mat = np.array(shape, dtype=np.uint8)
    for _ in range(4):
        if not any(mat.shape == r.shape and (mat == r).all() for r in rotations):
            rotations.append(mat)
        mat = np.rot90(mat, -1)
    return rotations


PIECE_ROTATIONS = {name: _unique_rotations(shape) for name, shape in SHAPES.items()}


@dataclass
class Placement:
    """One legal final position for the current piece."""

    rotation: int
    col: int
    row: int
    lines_cleared: int
    board_after: np.ndarray


def column_heights(board: np.ndarray) -> np.ndarray:
    filled = board > 0
    return np.where(filled.any(axis=0), BOARD_HEIGHT - filled.argmax(axis=0), 0)


def count_holes(board: np.ndarray) -> int:
    """Empty cells with at least one filled cell above them in the same column."""
    filled = board > 0
    covered = np.cumsum(filled, axis=0) > 0
    return int((covered & ~filled).sum())


def bumpiness(board: np.ndarray) -> int:
    heights = column_heights(board)
    return int(np.abs(np.diff(heights)).sum())


class TetrisGame:
    def __init__(self, seed: int | None = None):
        self.rng = random.Random(seed)
        self.reset()

    # ------------------------------------------------------------------ setup

    def reset(self) -> None:
        self.board = np.zeros((BOARD_HEIGHT, BOARD_WIDTH), dtype=np.uint8)
        self.score = 0
        self.lines = 0
        self.pieces_placed = 0
        self.game_over = False
        self._bag: list[str] = []
        self.next_name = self._draw()
        self._spawn()

    def _draw(self) -> str:
        if not self._bag:
            self._bag = list(PIECE_NAMES)
            self.rng.shuffle(self._bag)
        return self._bag.pop()

    def _spawn(self) -> None:
        self.piece_name = self.next_name
        self.next_name = self._draw()
        self.rotation = 0
        shape = self.shape
        self.row = 0
        self.col = (BOARD_WIDTH - shape.shape[1]) // 2
        if self._collides(shape, self.row, self.col):
            self.game_over = True

    @property
    def shape(self) -> np.ndarray:
        return PIECE_ROTATIONS[self.piece_name][self.rotation]

    @property
    def level(self) -> int:
        return self.lines // LINES_PER_LEVEL

    # ------------------------------------------------------------- collisions

    def _collides(self, shape: np.ndarray, row: int, col: int) -> bool:
        h, w = shape.shape
        if row < 0 or col < 0 or row + h > BOARD_HEIGHT or col + w > BOARD_WIDTH:
            return True
        region = self.board[row : row + h, col : col + w]
        return bool(((region > 0) & (shape > 0)).any())

    def _landing_row(self, shape: np.ndarray, col: int) -> int | None:
        """Lowest row the shape can occupy when dropped in ``col`` from the top."""
        if self._collides(shape, 0, col):
            return None
        row = 0
        while not self._collides(shape, row + 1, col):
            row += 1
        return row

    # ------------------------------------------------------- step-wise (human)

    def move(self, dx: int) -> bool:
        if self.game_over:
            return False
        if not self._collides(self.shape, self.row, self.col + dx):
            self.col += dx
            return True
        return False

    def rotate(self, direction: int = 1) -> bool:
        if self.game_over:
            return False
        rotations = PIECE_ROTATIONS[self.piece_name]
        new_rotation = (self.rotation + direction) % len(rotations)
        shape = rotations[new_rotation]
        for kick in (0, -1, 1, -2, 2):
            if not self._collides(shape, self.row, self.col + kick):
                self.rotation = new_rotation
                self.col += kick
                return True
        return False

    def tick(self) -> int:
        """One gravity step. Returns lines cleared (non-zero only on lock)."""
        if self.game_over:
            return 0
        if not self._collides(self.shape, self.row + 1, self.col):
            self.row += 1
            return 0
        return self._lock()

    def soft_drop(self) -> int:
        if self.game_over:
            return 0
        if not self._collides(self.shape, self.row + 1, self.col):
            self.row += 1
            self.score += 1
            return 0
        return self._lock()

    def hard_drop(self) -> int:
        if self.game_over:
            return 0
        while not self._collides(self.shape, self.row + 1, self.col):
            self.row += 1
            self.score += 2
        return self._lock()

    # ----------------------------------------------------------------- locking

    def _lock(self) -> int:
        shape = self.shape
        h, w = shape.shape
        region = self.board[self.row : self.row + h, self.col : self.col + w]
        region[shape > 0] = PIECE_IDS[self.piece_name]
        cleared = self._clear_lines()
        self.score += LINE_SCORES[cleared] * (self.level + 1)
        self.lines += cleared
        self.pieces_placed += 1
        self._spawn()
        return cleared

    def _clear_lines(self) -> int:
        full = (self.board > 0).all(axis=1)
        n = int(full.sum())
        if n:
            remaining = self.board[~full]
            self.board = np.vstack(
                [np.zeros((n, BOARD_WIDTH), dtype=np.uint8), remaining]
            )
        return n

    # ----------------------------------------------------------- RL placements

    def legal_placements(self) -> list[Placement]:
        """Every (rotation, column) hard-drop position for the current piece."""
        placements = []
        for rotation, shape in enumerate(PIECE_ROTATIONS[self.piece_name]):
            for col in range(BOARD_WIDTH - shape.shape[1] + 1):
                row = self._landing_row(shape, col)
                if row is None:
                    continue
                board = self.board.copy()
                h, w = shape.shape
                region = board[row : row + h, col : col + w]
                region[shape > 0] = PIECE_IDS[self.piece_name]
                full = (board > 0).all(axis=1)
                lines = int(full.sum())
                if lines:
                    board = np.vstack(
                        [np.zeros((lines, BOARD_WIDTH), dtype=np.uint8), board[~full]]
                    )
                placements.append(Placement(rotation, col, row, lines, board))
        return placements

    def apply_placement(self, rotation: int, col: int) -> int:
        """Drop the current piece at the given rotation/column and lock it.

        Returns the number of lines cleared. Unlike ``hard_drop`` this awards
        no drop points, so the score reflects line clears only (keeps the RL
        score metric comparable across policies).
        """
        if self.game_over:
            return 0
        self.rotation = rotation
        self.col = col
        row = self._landing_row(self.shape, col)
        if row is None:
            raise ValueError(f"illegal placement rotation={rotation} col={col}")
        self.row = row
        shape = self.shape
        h, w = shape.shape
        region = self.board[self.row : self.row + h, self.col : self.col + w]
        region[shape > 0] = PIECE_IDS[self.piece_name]
        cleared = self._clear_lines()
        self.score += LINE_SCORES[cleared] * (self.level + 1)
        self.lines += cleared
        self.pieces_placed += 1
        self._spawn()
        return cleared

    # ---------------------------------------------------------------- render

    def render_text(self, show_piece: bool = True) -> str:
        board = self.board.copy()
        if show_piece and not self.game_over:
            shape = self.shape
            h, w = shape.shape
            region = board[self.row : self.row + h, self.col : self.col + w]
            region[shape > 0] = PIECE_IDS[self.piece_name]
        return render_board(board, score=self.score, lines=self.lines,
                            level=self.level, next_name=self.next_name)


_COLORS = {0: "", 1: "\033[96m", 2: "\033[93m", 3: "\033[95m",
           4: "\033[92m", 5: "\033[91m", 6: "\033[94m", 7: "\033[33m"}
_RESET = "\033[0m"


def render_board(board: np.ndarray, score: int = 0, lines: int = 0,
                 level: int = 0, next_name: str | None = None) -> str:
    rows = []
    for r in range(BOARD_HEIGHT):
        cells = []
        for c in range(BOARD_WIDTH):
            v = int(board[r, c])
            cells.append(f"{_COLORS[v]}[]{_RESET}" if v else " .")
        rows.append("|" + "".join(cells) + "|")
    rows.append("+" + "-" * (2 * BOARD_WIDTH) + "+")
    info = f"score {score}  lines {lines}  level {level}"
    if next_name:
        info += f"  next {next_name}"
    rows.append(info)
    return "\n".join(rows)
