"""Evaluate a trained checkpoint by playing greedy episodes.

Usage:
    python evaluate.py --checkpoint runs/latest/best.pt --episodes 5
    python evaluate.py --checkpoint runs/latest/best.pt --render
"""

from __future__ import annotations

import argparse
import time

import numpy as np
import torch

from tetris_rl.env import TetrisPlacementEnv
from tetris_rl.game import render_board
from tetris_rl.model import load_checkpoint


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=str, default="runs/latest/best.pt")
    p.add_argument("--episodes", type=int, default=5)
    p.add_argument("--max-pieces", type=int, default=1_000,
                   help="cap per episode so strong agents terminate")
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--render", action="store_true",
                   help="draw the board after every placement")
    p.add_argument("--delay", type=float, default=0.05,
                   help="seconds between rendered frames")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    model, payload = load_checkpoint(args.checkpoint)
    print(f"loaded {args.checkpoint} (training step {payload.get('step', '?')})")

    scores, lines, pieces = [], [], []
    for episode in range(args.episodes):
        env = TetrisPlacementEnv(seed=args.seed + episode)
        feats = env.reset()
        done = False
        info = {"score": 0, "lines": 0, "pieces": 0}
        while not done and info["pieces"] < args.max_pieces:
            with torch.no_grad():
                values = model(torch.from_numpy(feats))
            _, done, info = env.step(int(values.argmax()))
            if args.render:
                print("\033[2J\033[H" + render_board(
                    env.game.board, score=info["score"], lines=info["lines"],
                    level=env.game.level, next_name=env.game.next_name))
                time.sleep(args.delay)
            if not done:
                feats = env.candidate_features()
        scores.append(info["score"])
        lines.append(info["lines"])
        pieces.append(info["pieces"])
        print(f"episode {episode + 1}: score {info['score']}  "
              f"lines {info['lines']}  pieces {info['pieces']}")

    print(f"\nmean score {np.mean(scores):.1f}  mean lines {np.mean(lines):.1f}  "
          f"mean pieces {np.mean(pieces):.1f}  max score {max(scores)}")


if __name__ == "__main__":
    main()
