# Reinforce-Tetris

Reinforcement-learning demos: each game in `games/` is a standalone pipeline (game engine,
RL env, training script, web UI) built the same way — pure-numpy game logic, parallel
environment workers, a small PyTorch model, and a browser viewer for watching/validating the
trained agent. Currently: **Tetris** (afterstate value learning) and **Snake** (DQN, see
[Snake](#snake)).

## Setup

```bash
uv sync
```

### GPU (CUDA) training

`pyproject.toml` pins the **CUDA 12.4** torch build (`[tool.uv.sources]` →
`pytorch-cu124`), so `uv sync` installs a GPU-capable torch. Then pass
`--device cuda` (or leave the default `auto`, which uses CUDA when present and
falls back to CPU):

```bash
uv run python -m games.snake.train  --workers 4 --total-steps 400000 --device cuda
uv run python -m games.tetris.train --workers 4 --total-pieces 200000 --device cuda
```

Check the GPU is seen: `uv run python -c "import torch; print(torch.cuda.is_available())"`.

**CPU only?** Delete the `[tool.uv.sources]` + `[[tool.uv.index]]` blocks from
`pyproject.toml` and re-run `uv sync` (or `uv add torch --index-url https://download.pytorch.org/whl/cpu`).

> The torch CUDA wheel is ~2.4 GB. If your system drive is short on space, point
> uv's cache at a roomier drive first: `setx UV_CACHE_DIR "D:\uv-cache"`.

> Note: this loop is largely IPC/simulation-bound (worker processes do the numpy
> game sim), so the GPU mainly accelerates the per-step training batch — expect a
> few-fold speedup, not orders of magnitude. More `--workers` is the other lever.

## Play Tetris yourself

```bash
uv run python -m games.tetris.play
```

Arrow keys move/rotate, `z`/`x` rotate, space hard-drops, `p` pauses, `q` quits.
Classic NES-style scoring (40/100/300/1200 × level+1, plus drop points).

## Train the agent

```bash
uv run python -m games.tetris.train --workers 4 --total-pieces 200000
```

Progress is printed every 500 steps; per-episode results land in
`runs/latest/episodes.csv` and checkpoints in `runs/latest/{latest,best}.pt`.
Useful flags: `--gamma`, `--lr`, `--eps-decay-frac`, `--buffer-size`
(see `uv run python -m games.tetris.train --help`).

## Web UI — watch & validate the model

```bash
uv run python serve.py     # auto-loads the newest checkpoint (models/pretrained.pt ships with the repo)
# open http://localhost:8000
```

![Agent viewer](docs/ui.png)

The browser viewer lets you validate model performance directly:

- **Live play** with drop animation, speed slider (1–20 pieces/s), play/pause/
  single-step/reset, and auto-restart.
- **Decision insight**: the network's value estimate for each chosen placement
  and the number of candidate placements it picked from.
- **Episode history** of every finished game in the session.
- **Benchmark button**: runs N greedy headless episodes server-side and reports
  per-episode score/lines plus mean/max — a quick quantitative check to go with
  the visual one.
- **Checkpoint dropdown** to hot-swap any `.pt` under `runs/` or `models/` and
  compare training stages.

A pretrained checkpoint (`models/pretrained.pt`, 18k pieces of training) is
included so the UI works immediately after cloning.

## Evaluate / watch the agent (terminal)

```bash
uv run python -m games.tetris.evaluate --checkpoint runs/latest/best.pt --episodes 5
uv run python -m games.tetris.evaluate --checkpoint runs/latest/best.pt --render   # watch it play
```

## How it works

**Game** (`games/tetris/game.py`) — pure numpy Tetris: 10×20 board, 7-bag
randomizer, wall-kick rotation, line clears, NES scoring. It exposes both a
step-wise interface for human play and a placement interface
(`legal_placements` / `apply_placement`) for the agent.

**Environment** (`games/tetris/env.py`) — an action is one of the legal
(rotation, column) hard-drop placements of the current piece. Each candidate
placement is summarized by 4 afterstate features:
`[lines_cleared, holes, bumpiness, total_height]`. Reward is
`1 + lines² × 10` per piece with a penalty on top-out, so multi-line clears
(especially Tetrises) are strongly rewarded. This placement-level formulation
is dramatically more sample-efficient than frame-level actions.

**Learning** (`games/tetris/train.py`) — a small MLP value network `V(afterstate)` is
trained with TD(0): `V(s'_t) → r_t + γ·V(s'_{t+1})`, with epsilon-greedy
exploration over candidate placements and a replay buffer.

**Parallelism** (`games/tetris/vec_env.py`) — N worker processes each run their
own game and compute candidate features (numpy only). Every global step the
main process batches *all* candidates from *all* workers through one forward
pass, picks per-environment actions, and sends them back over pipes —
SubprocVecEnv-style synchronous parallel rollout with a single shared network.

## Snake

The same recipe applied to Snake, in `games/snake/`:

![Snake agent viewer](docs/snake_ui.png)

```bash
uv run python -m games.snake.play                                      # play yourself (curses)
uv run python -m games.snake.train --workers 4 --total-steps 400000    # train
uv run python -m games.snake.evaluate --checkpoint runs/snake/best.pt  # headless eval
uv run python serve.py                                                 # web UI: open /snake
```

- **Game** (`games/snake/game.py`): classic Snake on a 12×12 grid — eat food,
  grow, die on walls/yourself, reversing is ignored.
- **Environment** (`games/snake/env.py`): a 14-feature state — the classic 11
  (danger straight/right/left, direction one-hot, food direction) **plus 3
  flood-fill freespace features** (reachable open area from each candidate
  next-head, as a fraction of the board). The freespace features give the agent
  body/space awareness the 11-feature state lacks, so it stops sealing itself
  into a pocket when long. 3 relative actions (straight / turn right / turn
  left). Reward +10 for food, −10 for dying or starving past a hunger limit.
- **Learning** (`games/snake/train.py`): standard DQN — Q-network with a target
  network, epsilon-greedy exploration, replay buffer — using the same
  parallel-worker design as Tetris (`games/snake/vec_env.py`): workers simulate,
  the main process batches all states through one forward pass per step.
- **Web UI**: `uv run python serve.py` then open http://localhost:8000/snake for live
  play with Q-value readout, episode history, benchmark button and checkpoint
  hot-swap. A pretrained model (`models/snake_pretrained.pt`) ships with the
  repo; it averages ~18–21 food per game (max ~32, snake length 35 on a
  144-cell board).

## Tests

```bash
uv run python -m pytest tests/ -q
```

## Adding another game

Each game is a self-contained package under `games/<name>/` with `game.py`, `env.py`,
`model.py`, `replay.py`, `vec_env.py`, plus `train.py`/`evaluate.py`/`play.py` entry points
(see `games/tetris/` or `games/snake/` for the pattern). Wire it into `serve.py` + `webui/`
to get a viewer page for free.
