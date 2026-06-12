"""Core Snake game logic on a rectangular grid.

The snake is a deque of (row, col) cells, head first. Directions are absolute
indices into ``DIRECTIONS``; a step in the direction opposite to travel is
ignored (the snake keeps going), matching classic Snake behaviour.
"""

from __future__ import annotations

import random
from collections import deque

GRID_WIDTH = 12
GRID_HEIGHT = 12

# up, right, down, left as (row, col) deltas; +1 mod 4 is a right turn
DIRECTIONS = [(-1, 0), (0, 1), (1, 0), (0, -1)]
UP, RIGHT, DOWN, LEFT = range(4)


class SnakeGame:
    def __init__(self, seed: int | None = None,
                 width: int = GRID_WIDTH, height: int = GRID_HEIGHT):
        self.rng = random.Random(seed)
        self.width = width
        self.height = height
        self.reset()

    def reset(self) -> None:
        r, c = self.height // 2, self.width // 2
        self.direction = RIGHT
        self.snake: deque[tuple[int, int]] = deque([(r, c), (r, c - 1), (r, c - 2)])
        self.score = 0
        self.steps = 0
        self.steps_since_food = 0
        self.game_over = False
        self.won = False
        self._place_food()

    def _place_food(self) -> None:
        body = set(self.snake)
        free = [(r, c) for r in range(self.height) for c in range(self.width)
                if (r, c) not in body]
        if not free:
            self.won = True
            self.game_over = True
            self.food = None
            return
        self.food = self.rng.choice(free)

    def collides(self, cell: tuple[int, int]) -> bool:
        r, c = cell
        if r < 0 or c < 0 or r >= self.height or c >= self.width:
            return True
        return cell in set(self.snake)

    def next_head(self, direction: int) -> tuple[int, int]:
        dr, dc = DIRECTIONS[direction]
        head = self.snake[0]
        return (head[0] + dr, head[1] + dc)

    def step(self, direction: int) -> dict:
        """Advance one cell. Returns ``{"ate": bool, "died": bool}``."""
        if self.game_over:
            return {"ate": False, "died": True}
        if (direction + 2) % 4 == self.direction:
            direction = self.direction  # cannot reverse into yourself
        self.direction = direction

        head = self.next_head(direction)
        self.steps += 1
        self.steps_since_food += 1
        if self.collides(head):
            self.game_over = True
            return {"ate": False, "died": True}

        self.snake.appendleft(head)
        if head == self.food:
            self.score += 1
            self.steps_since_food = 0
            self._place_food()
            return {"ate": True, "died": False}
        self.snake.pop()
        return {"ate": False, "died": False}
