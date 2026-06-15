"""Train a Snake DQN agent with parallel environment workers.

Standard DQN over the 11-feature state with 3 relative actions:

    Q(s_t, a_t) -> r_t + gamma * max_a Q_target(s_{t+1}, a)

with epsilon-greedy exploration, a replay buffer and a periodically synced
target network. Worker processes simulate the games; the main process picks
actions for all of them in one batched forward pass per step.

Usage:
    python train_snake.py --workers 4 --total-steps 400000
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

from snake_rl.model import QNet, save_checkpoint
from snake_rl.replay import ReplayBuffer
from snake_rl.vec_env import ParallelSnake


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--workers", type=int, default=min(4, os.cpu_count() or 1))
    p.add_argument("--total-steps", type=int, default=400_000,
                   help="environment steps per worker")
    p.add_argument("--width", type=int, default=12)
    p.add_argument("--height", type=int, default=12)
    p.add_argument("--gamma", type=float, default=0.9)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--buffer-size", type=int, default=100_000)
    p.add_argument("--warmup", type=int, default=2_000)
    p.add_argument("--target-sync", type=int, default=1_000,
                   help="global steps between target-network syncs")
    p.add_argument("--eps-start", type=float, default=1.0)
    p.add_argument("--eps-end", type=float, default=0.01)
    p.add_argument("--eps-decay-frac", type=float, default=0.3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out-dir", type=str, default="runs/snake")
    p.add_argument("--log-every", type=int, default=5_000)
    p.add_argument("--save-every", type=int, default=20_000)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    model = QNet()
    target = QNet()
    target.load_state_dict(model.state_dict())
    target.eval()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.SmoothL1Loss()
    buffer = ReplayBuffer(args.buffer_size, rng=rng)

    vec = ParallelSnake(args.workers, base_seed=args.seed,
                        width=args.width, height=args.height)
    states = np.stack(vec.reset())

    decay_steps = max(1, int(args.total_steps * args.eps_decay_frac))
    recent_scores: deque[float] = deque(maxlen=100)
    recent_steps: deque[float] = deque(maxlen=100)
    episodes = 0
    best_mean_score = float("-inf")
    losses: deque[float] = deque(maxlen=200)
    start = time.time()

    log_path = os.path.join(args.out_dir, "episodes.csv")
    log_file = open(log_path, "w", newline="")
    logger = csv.writer(log_file)
    logger.writerow(["episode", "global_step", "score", "steps", "epsilon"])

    for step in range(args.total_steps):
        eps = args.eps_start + (args.eps_end - args.eps_start) * min(1.0, step / decay_steps)

        with torch.no_grad():
            q_values = model(torch.from_numpy(states)).numpy()
        greedy = q_values.argmax(axis=1)
        explore = rng.random(args.workers) < eps
        actions = np.where(explore, rng.integers(0, 3, size=args.workers), greedy)

        results = vec.step(actions)
        for i, (reward, done, info, next_state) in enumerate(results):
            # on done, next_state is the fresh reset state; the terminal
            # next-state is irrelevant because the target is masked by done
            buffer.push(states[i], actions[i], reward, next_state, done)
            states[i] = next_state
            if done:
                episodes += 1
                recent_scores.append(info["score"])
                recent_steps.append(info["steps"])
                logger.writerow([episodes, step, info["score"], info["steps"],
                                 round(eps, 4)])

        if len(buffer) >= args.warmup:
            s, a, r, s2, d = buffer.sample(args.batch_size)
            with torch.no_grad():
                next_q = target(torch.from_numpy(s2)).max(dim=1).values
                targets = torch.from_numpy(r) + args.gamma * next_q * (
                    1.0 - torch.from_numpy(d)
                )
            q = model(torch.from_numpy(s)).gather(
                1, torch.from_numpy(a).unsqueeze(1)
            ).squeeze(1)
            loss = loss_fn(q, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        if (step + 1) % args.target_sync == 0:
            target.load_state_dict(model.state_dict())

        if (step + 1) % args.log_every == 0:
            mean_score = float(np.mean(recent_scores)) if recent_scores else 0.0
            mean_len = float(np.mean(recent_steps)) if recent_steps else 0.0
            mean_loss = float(np.mean(losses)) if losses else 0.0
            rate = (step + 1) * args.workers / (time.time() - start)
            print(
                f"step {step + 1}/{args.total_steps}  eps {eps:.3f}  "
                f"episodes {episodes}  avg score {mean_score:.1f}  "
                f"avg steps {mean_len:.0f}  loss {mean_loss:.4f}  {rate:.0f} steps/s",
                flush=True,
            )
            log_file.flush()

        if (step + 1) % args.save_every == 0 or step + 1 == args.total_steps:
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
