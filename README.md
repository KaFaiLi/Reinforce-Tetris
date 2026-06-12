# Reinforce-Tetris

A standard Tetris game plus a reinforcement-learning pipeline that trains an
agent to score, using parallel environment workers.

## Setup

```bash
pip install -r requirements.txt
# CPU-only torch: pip install torch --index-url https://download.pytorch.org/whl/cpu
```

## Play Tetris yourself

```bash
python play.py
```

Arrow keys move/rotate, `z`/`x` rotate, space hard-drops, `p` pauses, `q` quits.
Classic NES-style scoring (40/100/300/1200 × level+1, plus drop points).

## Train the agent

```bash
python train.py --workers 4 --total-pieces 200000
```

Progress is printed every 500 steps; per-episode results land in
`runs/latest/episodes.csv` and checkpoints in `runs/latest/{latest,best}.pt`.
Useful flags: `--gamma`, `--lr`, `--eps-decay-frac`, `--buffer-size`
(see `python train.py --help`).

## Evaluate / watch the agent

```bash
python evaluate.py --checkpoint runs/latest/best.pt --episodes 5
python evaluate.py --checkpoint runs/latest/best.pt --render   # watch it play
```

## How it works

**Game** (`tetris_rl/game.py`) — pure numpy Tetris: 10×20 board, 7-bag
randomizer, wall-kick rotation, line clears, NES scoring. It exposes both a
step-wise interface for human play and a placement interface
(`legal_placements` / `apply_placement`) for the agent.

**Environment** (`tetris_rl/env.py`) — an action is one of the legal
(rotation, column) hard-drop placements of the current piece. Each candidate
placement is summarized by 4 afterstate features:
`[lines_cleared, holes, bumpiness, total_height]`. Reward is
`1 + lines² × 10` per piece with a penalty on top-out, so multi-line clears
(especially Tetrises) are strongly rewarded. This placement-level formulation
is dramatically more sample-efficient than frame-level actions.

**Learning** (`train.py`) — a small MLP value network `V(afterstate)` is
trained with TD(0): `V(s'_t) → r_t + γ·V(s'_{t+1})`, with epsilon-greedy
exploration over candidate placements and a replay buffer.

**Parallelism** (`tetris_rl/vec_env.py`) — N worker processes each run their
own game and compute candidate features (numpy only). Every global step the
main process batches *all* candidates from *all* workers through one forward
pass, picks per-environment actions, and sends them back over pipes —
SubprocVecEnv-style synchronous parallel rollout with a single shared network.

## Tests

```bash
python -m pytest tests/ -q
```
