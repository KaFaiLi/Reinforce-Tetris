"""Evaluate a trained Snake checkpoint with greedy play.

Usage:
    python evaluate_snake.py --checkpoint runs/snake/best.pt --episodes 20
    python evaluate_snake.py --checkpoint runs/snake/best.pt --render
"""

from __future__ import annotations

import argparse
import time

import numpy as np
import torch

from snake_rl.env import SnakeEnv
from snake_rl.model import load_checkpoint


def render(env: SnakeEnv) -> str:
    game = env.game
    grid = [["." for _ in range(game.width)] for _ in range(game.height)]
    if game.food:
        grid[game.food[0]][game.food[1]] = "*"
    for r, c in list(game.snake)[1:]:
        grid[r][c] = "o"
    head_r, head_c = game.snake[0]
    grid[head_r][head_c] = "O"
    lines = ["+" + "-" * (2 * game.width) + "+"]
    lines += ["|" + " ".join(row) + " |" for row in grid]
    lines.append("+" + "-" * (2 * game.width) + "+")
    lines.append(f"score {game.score}  steps {game.steps}")
    return "\n".join(lines)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=str, default="runs/snake/best.pt")
    p.add_argument("--episodes", type=int, default=20)
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--render", action="store_true")
    p.add_argument("--delay", type=float, default=0.05)
    args = p.parse_args()

    model, payload = load_checkpoint(args.checkpoint)
    print(f"loaded {args.checkpoint} (training step {payload.get('step', '?')})")

    scores, lengths = [], []
    for episode in range(args.episodes):
        env = SnakeEnv(seed=args.seed + episode)
        state = env.reset()
        done = False
        info = {"score": 0, "steps": 0}
        while not done:
            with torch.no_grad():
                q = model(torch.from_numpy(state))
            _, done, info = env.step(int(q.argmax()))
            if args.render:
                print("\033[2J\033[H" + render(env))
                time.sleep(args.delay)
            if not done:
                state = env.state()
        scores.append(info["score"])
        lengths.append(info["steps"])
        print(f"episode {episode + 1}: score {info['score']}  steps {info['steps']}")

    print(f"\nmean score {np.mean(scores):.1f}  max score {max(scores)}  "
          f"mean steps {np.mean(lengths):.0f}")


if __name__ == "__main__":
    main()
