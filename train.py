"""Train a Tetris value network with parallel environment workers.

Each global step places one piece in every parallel environment:

1. Workers send the candidate afterstate features for their current piece.
2. The main process scores all candidates of all environments in one batched
   forward pass and picks per-environment epsilon-greedy placements.
3. Workers apply the placements; transitions go into a shared replay buffer
   and the value network is updated by TD(0) on afterstates:

       V(s'_t) -> r_t + gamma * V(s'_{t+1})

Usage:
    python train.py --workers 4 --total-pieces 200000
"""

from __future__ import annotations

import argparse
import csv
import os
import time
from collections import deque

import numpy as np
import torch
import torch.nn as nn

from tetris_rl.env import FEATURE_DIM
from tetris_rl.model import ValueNet, save_checkpoint
from tetris_rl.replay import ReplayBuffer
from tetris_rl.vec_env import ParallelTetris


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--workers", type=int, default=min(4, os.cpu_count() or 1),
                   help="number of parallel environment processes")
    p.add_argument("--total-pieces", type=int, default=200_000,
                   help="total pieces to place per environment")
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--buffer-size", type=int, default=50_000)
    p.add_argument("--warmup", type=int, default=2_000,
                   help="transitions collected before training starts")
    p.add_argument("--eps-start", type=float, default=1.0)
    p.add_argument("--eps-end", type=float, default=0.02)
    p.add_argument("--eps-decay-frac", type=float, default=0.5,
                   help="fraction of total steps over which epsilon decays")
    p.add_argument("--game-over-penalty", type=float, default=-5.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out-dir", type=str, default="runs/latest")
    p.add_argument("--log-every", type=int, default=500,
                   help="global steps between progress prints")
    p.add_argument("--save-every", type=int, default=5_000,
                   help="global steps between checkpoint saves")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    model = ValueNet()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.MSELoss()
    buffer = ReplayBuffer(args.buffer_size, rng=rng)

    vec = ParallelTetris(args.workers, base_seed=args.seed,
                         game_over_penalty=args.game_over_penalty)
    candidates = vec.reset()
    # Transition (chosen afterstate, reward) waiting for the next afterstate.
    pending: list[tuple[np.ndarray, float] | None] = [None] * args.workers

    total_steps = args.total_pieces
    decay_steps = max(1, int(total_steps * args.eps_decay_frac))
    recent_scores: deque[float] = deque(maxlen=50)
    recent_lines: deque[float] = deque(maxlen=50)
    recent_pieces: deque[float] = deque(maxlen=50)
    episodes = 0
    best_mean_score = float("-inf")
    losses: deque[float] = deque(maxlen=200)
    start = time.time()

    log_path = os.path.join(args.out_dir, "episodes.csv")
    log_file = open(log_path, "w", newline="")
    logger = csv.writer(log_file)
    logger.writerow(["episode", "global_step", "score", "lines", "pieces", "epsilon"])

    for step in range(total_steps):
        eps = args.eps_start + (args.eps_end - args.eps_start) * min(1.0, step / decay_steps)

        # Batched value estimates for every candidate of every env.
        flat = np.concatenate(candidates, axis=0)
        with torch.no_grad():
            values = model(torch.from_numpy(flat)).numpy()
        actions = []
        chosen_feats = []
        offset = 0
        for feats in candidates:
            n = len(feats)
            if rng.random() < eps:
                a = int(rng.integers(n))
            else:
                a = int(values[offset : offset + n].argmax())
            actions.append(a)
            chosen_feats.append(feats[a])
            offset += n

        # The afterstate chosen now is the "next state" of the previous
        # transition in the same environment.
        for i in range(args.workers):
            if pending[i] is not None:
                prev_feats, prev_reward = pending[i]
                buffer.push(prev_feats, prev_reward, chosen_feats[i], False)

        results = vec.step(actions)
        for i, (reward, done, info, next_feats) in enumerate(results):
            if done:
                buffer.push(chosen_feats[i], reward,
                            np.zeros(FEATURE_DIM, dtype=np.float32), True)
                pending[i] = None
                episodes += 1
                recent_scores.append(info["score"])
                recent_lines.append(info["lines"])
                recent_pieces.append(info["pieces"])
                logger.writerow([episodes, step, info["score"], info["lines"],
                                 info["pieces"], round(eps, 4)])
            else:
                pending[i] = (chosen_feats[i], reward)
            candidates[i] = next_feats

        if len(buffer) >= args.warmup:
            states, rewards, next_states, dones = buffer.sample(args.batch_size)
            with torch.no_grad():
                next_values = model(torch.from_numpy(next_states))
                targets = torch.from_numpy(rewards) + args.gamma * next_values * (
                    1.0 - torch.from_numpy(dones)
                )
            predictions = model(torch.from_numpy(states))
            loss = loss_fn(predictions, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        if (step + 1) % args.log_every == 0:
            mean_score = float(np.mean(recent_scores)) if recent_scores else 0.0
            mean_lines = float(np.mean(recent_lines)) if recent_lines else 0.0
            mean_pieces = float(np.mean(recent_pieces)) if recent_pieces else 0.0
            mean_loss = float(np.mean(losses)) if losses else 0.0
            rate = (step + 1) * args.workers / (time.time() - start)
            print(
                f"step {step + 1}/{total_steps}  eps {eps:.3f}  episodes {episodes}  "
                f"avg score {mean_score:.1f}  avg lines {mean_lines:.1f}  "
                f"avg pieces {mean_pieces:.1f}  loss {mean_loss:.3f}  "
                f"{rate:.0f} pieces/s",
                flush=True,
            )
            log_file.flush()

        if (step + 1) % args.save_every == 0 or step + 1 == total_steps:
            save_checkpoint(os.path.join(args.out_dir, "latest.pt"), model,
                            step=step + 1, args=vars(args))
            mean_score = float(np.mean(recent_scores)) if recent_scores else float("-inf")
            if mean_score > best_mean_score:
                best_mean_score = mean_score
                save_checkpoint(os.path.join(args.out_dir, "best.pt"), model,
                                step=step + 1, mean_score=mean_score, args=vars(args))

    vec.close()
    log_file.close()
    print(f"done: {episodes} episodes, best avg score {best_mean_score:.1f}, "
          f"checkpoints in {args.out_dir}")


if __name__ == "__main__":
    main()
