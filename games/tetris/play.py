"""Play Tetris in the terminal (curses).

Controls:
    left/right arrows  move
    up arrow or x      rotate clockwise
    z                  rotate counter-clockwise
    down arrow         soft drop
    space              hard drop
    p                  pause
    q                  quit

Usage:
    uv run python -m games.tetris.play
"""

from __future__ import annotations

import curses
import time

from .game import (
    BOARD_HEIGHT,
    BOARD_WIDTH,
    PIECE_IDS,
    PIECE_ROTATIONS,
    TetrisGame,
)

COLOR_OF_ID = {1: 6, 2: 3, 3: 5, 4: 2, 5: 1, 6: 4, 7: 3}


def gravity_delay(level: int) -> float:
    return max(0.08, 0.8 - 0.07 * level)


def draw(stdscr, game: TetrisGame) -> None:
    stdscr.erase()
    board = game.board.copy()
    if not game.game_over:
        shape = game.shape
        h, w = shape.shape
        region = board[game.row : game.row + h, game.col : game.col + w]
        region[shape > 0] = 8  # marker for the falling piece

    stdscr.addstr(0, 0, "+" + "--" * BOARD_WIDTH + "+")
    for r in range(BOARD_HEIGHT):
        stdscr.addstr(r + 1, 0, "|")
        for c in range(BOARD_WIDTH):
            v = int(board[r, c])
            if v == 0:
                stdscr.addstr(r + 1, 1 + 2 * c, " .")
            else:
                color = COLOR_OF_ID.get(v, 7) if v != 8 else 7
                stdscr.addstr(r + 1, 1 + 2 * c, "[]",
                              curses.color_pair(color) | curses.A_BOLD)
        stdscr.addstr(r + 1, 1 + 2 * BOARD_WIDTH, "|")
    stdscr.addstr(BOARD_HEIGHT + 1, 0, "+" + "--" * BOARD_WIDTH + "+")

    panel = 2 * BOARD_WIDTH + 5
    stdscr.addstr(1, panel, f"score  {game.score}")
    stdscr.addstr(2, panel, f"lines  {game.lines}")
    stdscr.addstr(3, panel, f"level  {game.level}")
    stdscr.addstr(5, panel, f"next   {game.next_name}")
    shape = PIECE_ROTATIONS[game.next_name][0]
    for r in range(shape.shape[0]):
        row = "".join("[]" if cell else "  " for cell in shape[r])
        stdscr.addstr(6 + r, panel, row,
                      curses.color_pair(COLOR_OF_ID[PIECE_IDS[game.next_name]]))
    stdscr.addstr(BOARD_HEIGHT - 1, panel, "arrows move/rotate")
    stdscr.addstr(BOARD_HEIGHT, panel, "space drop, p pause, q quit")
    if game.game_over:
        stdscr.addstr(BOARD_HEIGHT // 2, 3, " GAME OVER - q to quit ",
                      curses.A_REVERSE | curses.A_BOLD)
    stdscr.refresh()


def run(stdscr) -> None:
    curses.curs_set(0)
    curses.start_color()
    curses.use_default_colors()
    for i in range(1, 8):
        curses.init_pair(i, i, -1)
    stdscr.nodelay(True)

    game = TetrisGame()
    last_tick = time.time()
    paused = False
    while True:
        draw(stdscr, game)
        key = stdscr.getch()
        if key == ord("q"):
            return
        if key == ord("p"):
            paused = not paused
        if not paused and not game.game_over:
            if key == curses.KEY_LEFT:
                game.move(-1)
            elif key == curses.KEY_RIGHT:
                game.move(1)
            elif key in (curses.KEY_UP, ord("x")):
                game.rotate(1)
            elif key == ord("z"):
                game.rotate(-1)
            elif key == curses.KEY_DOWN:
                game.soft_drop()
                last_tick = time.time()
            elif key == ord(" "):
                game.hard_drop()
                last_tick = time.time()
            if time.time() - last_tick >= gravity_delay(game.level):
                game.tick()
                last_tick = time.time()
        time.sleep(0.01)


if __name__ == "__main__":
    curses.wrapper(run)
