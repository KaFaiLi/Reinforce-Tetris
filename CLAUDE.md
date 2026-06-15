# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
uv sync                                   # install deps (numpy, torch; +pytest dev group, Python >=3.12)
# CPU-only torch: uv add torch --index-url https://download.pytorch.org/whl/cpu

uv run python -m pytest tests/ -q                            # all tests
uv run python -m pytest tests/test_tetris.py -q              # single file
uv run python -m pytest tests/test_tetris.py::test_name -q   # single test

uv run python -m games.tetris.train --workers 4 --total-pieces 200000        # train Tetris agent
uv run python -m games.snake.train --workers 4 --total-steps 400000          # train Snake agent
uv run python -m games.tetris.evaluate --checkpoint runs/latest/best.pt --episodes 5   # headless eval
uv run python -m games.tetris.evaluate --checkpoint runs/latest/best.pt --render       # watch in terminal
uv run python serve.py                          # web UI on :8000 (Tetris at /, Snake at /snake)
uv run python -m games.tetris.play              # play Tetris yourself (curses)
uv run python -m games.snake.play               # play Snake yourself (curses)
```

`uv run python -m games.tetris.train --help` lists hyperparameter flags (`--gamma`, `--lr`,
`--eps-decay-frac`, `--buffer-size`, `--game-over-penalty`, etc.). Checkpoints + `episodes.csv`
land in `--out-dir` (default `runs/latest`, Snake uses `runs/snake`).

## Platform gotcha

`vec_env.py` (both games) uses `fork` where available (Linux/macOS) and falls back to
`spawn` on Windows. `spawn` re-imports the module per worker, so the `train.py` `__main__`
guard is load-bearing on Windows — keep it. Tests, eval, play, and `serve.py` also run fine.

## Architecture

Two independent RL pipelines sharing one design, each its own package under `games/`:
`games/tetris/` and `games/snake/`. Each package has the same 5 library modules — `game.py`
(pure-numpy game logic), `env.py` (RL wrapper), `model.py` (network + `save_checkpoint`/
`load_checkpoint`), `replay.py` (buffer), `vec_env.py` (parallel workers) — plus 3 entry-point
scripts: `train.py`, `evaluate.py`, `play.py` (run as `python -m games.<game>.<script>`, using
relative imports within the package). The two games differ in their RL formulation:

**Tetris — afterstate value learning (TD(0)), not DQN.** The key idea: an action is a whole
*placement* (rotation + column hard-drop), not a frame-level move. `env.py` enumerates every
legal placement of the current piece and summarizes each resulting board as 4 afterstate
features `[lines_cleared, holes, bumpiness, total_height]` (`FEATURE_DIM=4`). `ValueNet` is a
tiny MLP `V(afterstate) -> scalar`; the agent picks the placement with the highest predicted
value. `train.py` trains with TD(0): `V(s'_t) -> r_t + γ·V(s'_{t+1})`. Reward is
`1 + lines² × BOARD_WIDTH` per piece (quadratic → Tetrises strongly rewarded) plus a top-out
penalty. This placement-level formulation is far more sample-efficient than frame-level actions.

**Snake — standard DQN.** Classic 11-feature compact state (danger straight/right/left,
direction one-hot, food direction), 3 relative actions (straight / turn-right / turn-left).
`games/snake/train.py` is textbook DQN: Q-network + target network synced every `--target-sync`
steps, epsilon-greedy, replay buffer storing `(s, a, r, s', done)`.

**Shared parallelism pattern (the non-obvious part).** `vec_env.py` is SubprocVecEnv-style:
N worker *processes* each own a game and do all numpy simulation + candidate-feature computation;
the **main process owns the single network**. Every global step the main process gathers
candidates/states from all workers, runs **one batched forward pass**, picks per-env
epsilon-greedy actions, and ships them back over pipes. Workers auto-reset on episode end and
return the fresh candidate set plus final stats in the same `step` reply. Note the consequence
for Tetris: each worker returns a *variable-length* candidate matrix, so `train.py` concatenates
all candidates across all envs into one flat batch, scores them, then slices back per-env by
offset (see the `offset`/`chosen_feats` loop).

**Transition bookkeeping in `games/tetris/train.py`:** because the target is the *next*
afterstate, each worker holds a `pending` (chosen_feats, reward) tuple that is only pushed to the
buffer once the *following* step's chosen afterstate is known (that becomes its `next_state`).
Terminal transitions push a zero next-state with `done=True`.

**`serve.py`** (project root) runs both games + models server-side behind a stdlib
`ThreadingHTTPServer`, serving a JSON API to `webui/index.html` (Tetris) and `webui/snake.html`
(Snake). It auto-discovers checkpoints under `runs/**` and `models/`, routing by whether the path
contains "snake". Ships pretrained `models/pretrained.pt` (Tetris) and `models/snake_pretrained.pt`
(Snake) so the UI works immediately after clone. Includes a server-side headless benchmark
endpoint for quantitative validation alongside the visual one.

## Adding a new game

Follow the `games/tetris/` or `games/snake/` layout: a package with `game.py`, `env.py`,
`model.py`, `replay.py`, `vec_env.py` (relative imports), plus `train.py`/`evaluate.py`/`play.py`
entry points runnable via `python -m games.<name>.<script>`. Wire a web UI page into `serve.py` +
`webui/` if you want it in the viewer.
