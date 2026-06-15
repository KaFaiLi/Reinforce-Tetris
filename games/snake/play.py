"""Play Snake in the terminal (curses).

Controls: arrow keys steer, p pauses, q quits.

Usage:
    uv run python -m games.snake.play
"""

from __future__ import annotations

import curses
import time

from .game import DOWN, LEFT, RIGHT, UP, SnakeGame

KEY_TO_DIR = {
    curses.KEY_UP: UP,
    curses.KEY_RIGHT: RIGHT,
    curses.KEY_DOWN: DOWN,
    curses.KEY_LEFT: LEFT,
}


def speed_delay(score: int) -> float:
    return max(0.06, 0.22 - 0.01 * score)


def draw(stdscr, game: SnakeGame) -> None:
    stdscr.erase()
    stdscr.addstr(0, 0, "+" + "--" * game.width + "+")
    for r in range(game.height):
        stdscr.addstr(r + 1, 0, "|" + "  " * game.width + "|")
    stdscr.addstr(game.height + 1, 0, "+" + "--" * game.width + "+")
    if game.food:
        stdscr.addstr(game.food[0] + 1, 1 + 2 * game.food[1], "()",
                      curses.color_pair(1) | curses.A_BOLD)
    for i, (r, c) in enumerate(game.snake):
        attr = curses.color_pair(2) | (curses.A_BOLD if i == 0 else 0)
        stdscr.addstr(r + 1, 1 + 2 * c, "[]", attr)
    panel = 2 * game.width + 5
    stdscr.addstr(1, panel, f"score  {game.score}")
    stdscr.addstr(2, panel, f"steps  {game.steps}")
    stdscr.addstr(4, panel, "arrows steer, p pause, q quit")
    if game.game_over:
        msg = " YOU WIN! " if game.won else " GAME OVER - q to quit "
        stdscr.addstr(game.height // 2, 3, msg, curses.A_REVERSE | curses.A_BOLD)
    stdscr.refresh()


def run(stdscr) -> None:
    curses.curs_set(0)
    curses.start_color()
    curses.use_default_colors()
    curses.init_pair(1, curses.COLOR_RED, -1)
    curses.init_pair(2, curses.COLOR_GREEN, -1)
    stdscr.nodelay(True)

    game = SnakeGame()
    direction = game.direction
    last_tick = time.time()
    paused = False
    while True:
        draw(stdscr, game)
        key = stdscr.getch()
        if key == ord("q"):
            return
        if key == ord("p"):
            paused = not paused
        if key in KEY_TO_DIR:
            direction = KEY_TO_DIR[key]
        if not paused and not game.game_over:
            if time.time() - last_tick >= speed_delay(game.score):
                game.step(direction)
                direction = game.direction  # reversal attempts were ignored
                last_tick = time.time()
        time.sleep(0.01)


if __name__ == "__main__":
    curses.wrapper(run)
