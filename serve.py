"""Web UI for watching and validating a trained Tetris agent.

Runs the game and model server-side and exposes a small JSON API consumed by
``webui/index.html``. No dependencies beyond the training stack.

Usage:
    uv run python serve.py --checkpoint runs/latest/best.pt --port 8000
    # then open http://localhost:8000
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import random
import threading

import numpy as np
import torch
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from games.snake.env import SnakeEnv
from games.snake.model import load_checkpoint as load_snake_checkpoint
from games.tetris.env import TetrisPlacementEnv
from games.tetris.game import PIECE_IDS, PIECE_ROTATIONS
from games.tetris.model import load_checkpoint

WEBUI_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "webui")


def _all_checkpoints() -> list[str]:
    paths = glob.glob("runs/**/*.pt", recursive=True) + glob.glob("models/*.pt")
    return sorted(set(paths), key=os.path.getmtime, reverse=True)


def find_checkpoints() -> list[str]:
    return [p for p in _all_checkpoints() if "snake" not in p.lower()]


def find_snake_checkpoints() -> list[str]:
    return [p for p in _all_checkpoints() if "snake" in p.lower()]


def _piece_json(name: str, rotation: int = 0) -> dict:
    shape = PIECE_ROTATIONS[name][rotation]
    return {
        "name": name,
        "id": PIECE_IDS[name],
        "cells": np.argwhere(shape > 0).tolist(),
    }


class AgentSession:
    """Single shared game session driven by the loaded value network."""

    def __init__(self):
        self.lock = threading.Lock()
        self.model = None
        self.checkpoint: str | None = None
        self.env: TetrisPlacementEnv | None = None
        self.history: list[dict] = []
        self.benchmark = {"running": False, "episodes": [], "total": 0}

    def load(self, path: str) -> None:
        model, _ = load_checkpoint(path)
        with self.lock:
            self.model = model
            self.checkpoint = path
            self._reset_locked()

    def _reset_locked(self) -> None:
        self.env = TetrisPlacementEnv(seed=random.randrange(1_000_000))
        self.env.reset()

    def reset(self) -> dict:
        with self.lock:
            self._reset_locked()
            return self._state_locked()

    def _state_locked(self) -> dict:
        game = self.env.game
        return {
            "loaded": self.model is not None,
            "checkpoint": self.checkpoint,
            "board": game.board.tolist(),
            "score": game.score,
            "lines": game.lines,
            "level": game.level,
            "pieces": game.pieces_placed,
            "done": game.game_over,
            "next": _piece_json(game.next_name),
        }

    def state(self) -> dict:
        with self.lock:
            if self.env is None:
                return {"loaded": False, "checkpoint": None}
            return self._state_locked()

    def step(self) -> dict:
        """Apply the model's greedy placement; returns animation-ready data."""
        with self.lock:
            env, game = self.env, self.env.game
            if game.game_over:
                return self._state_locked()
            feats = env.candidate_features()
            with torch.no_grad():
                values = self.model(torch.from_numpy(feats)).numpy()
            idx = int(values.argmax())
            rotation, col = env._actions[idx]
            name = game.piece_name
            shape = PIECE_ROTATIONS[name][rotation]
            row = game._landing_row(shape, col)
            board_before = game.board.tolist()

            placed = game.board.copy()
            h, w = shape.shape
            region = placed[row : row + h, col : col + w]
            region[shape > 0] = PIECE_IDS[name]
            cleared_rows = np.flatnonzero((placed > 0).all(axis=1)).tolist()

            reward, done, info = env.step(idx)
            if done:
                self.history.append(
                    {"score": info["score"], "lines": info["lines"],
                     "pieces": info["pieces"]}
                )
            order = np.argsort(values)[::-1]
            return {
                "piece": {**_piece_json(name, rotation), "col": col, "row": row},
                "board_before": board_before,
                "board_placed": placed.tolist(),
                "cleared_rows": cleared_rows,
                "board": game.board.tolist(),
                "score": info["score"],
                "lines": info["lines"],
                "level": game.level,
                "pieces": info["pieces"],
                "done": done,
                "reward": reward,
                "value": float(values[idx]),
                "num_candidates": len(values),
                "alternatives": [round(float(values[i]), 2) for i in order[:3]],
                "next": _piece_json(game.next_name) if not done else None,
                "history": self.history[-20:],
            }

    # ------------------------------------------------------------- benchmark

    def start_benchmark(self, episodes: int, max_pieces: int) -> bool:
        with self.lock:
            if self.model is None or self.benchmark["running"]:
                return False
            self.benchmark = {"running": True, "episodes": [], "total": episodes}
        threading.Thread(
            target=self._run_benchmark, args=(episodes, max_pieces), daemon=True
        ).start()
        return True

    def _run_benchmark(self, episodes: int, max_pieces: int) -> None:
        for ep in range(episodes):
            env = TetrisPlacementEnv(seed=1000 + ep)
            feats = env.reset()
            done = False
            info = {"score": 0, "lines": 0, "pieces": 0}
            while not done and info["pieces"] < max_pieces:
                with torch.no_grad():
                    values = self.model(torch.from_numpy(feats))
                _, done, info = env.step(int(values.argmax()))
                if not done:
                    feats = env.candidate_features()
            with self.lock:
                self.benchmark["episodes"].append(
                    {"score": info["score"], "lines": info["lines"],
                     "pieces": info["pieces"], "topped_out": bool(done)}
                )
        with self.lock:
            self.benchmark["running"] = False

    def benchmark_status(self) -> dict:
        with self.lock:
            status = {**self.benchmark}
            eps = status["episodes"]
            if eps:
                scores = [e["score"] for e in eps]
                status["mean_score"] = round(float(np.mean(scores)), 1)
                status["max_score"] = max(scores)
                status["mean_lines"] = round(
                    float(np.mean([e["lines"] for e in eps])), 1
                )
            return status


class SnakeSession:
    """Single shared Snake game driven by the loaded Q-network."""

    def __init__(self):
        self.lock = threading.Lock()
        self.model = None
        self.checkpoint: str | None = None
        self.env: SnakeEnv | None = None
        self.history: list[dict] = []
        self.benchmark = {"running": False, "episodes": [], "total": 0}

    def load(self, path: str) -> None:
        model, _ = load_snake_checkpoint(path)
        with self.lock:
            self.model = model
            self.checkpoint = path
            self._reset_locked()

    def _reset_locked(self) -> None:
        self.env = SnakeEnv(seed=random.randrange(1_000_000))
        self.env.reset()

    def reset(self) -> dict:
        with self.lock:
            self._reset_locked()
            return self._state_locked()

    def _state_locked(self) -> dict:
        game = self.env.game
        return {
            "loaded": self.model is not None,
            "checkpoint": self.checkpoint,
            "width": game.width,
            "height": game.height,
            "snake": list(game.snake),
            "food": game.food,
            "score": game.score,
            "steps": game.steps,
            "length": len(game.snake),
            "done": game.game_over,
            "won": game.won,
        }

    def state(self) -> dict:
        with self.lock:
            if self.env is None:
                return {"loaded": False, "checkpoint": None}
            return self._state_locked()

    def step(self) -> dict:
        with self.lock:
            env = self.env
            if env.game.game_over:
                return self._state_locked()
            features = env.state()
            with torch.no_grad():
                q = self.model(torch.from_numpy(features)).numpy()
            action = int(q.argmax())
            reward, done, info = env.step(action)
            if done:
                self.history.append({"score": info["score"], "steps": info["steps"]})
            return {
                **self._state_locked(),
                "ate": reward > 0,
                "q_value": round(float(q[action]), 2),
                "q_values": [round(float(v), 2) for v in q],
                "history": self.history[-20:],
            }

    def start_benchmark(self, episodes: int) -> bool:
        with self.lock:
            if self.model is None or self.benchmark["running"]:
                return False
            self.benchmark = {"running": True, "episodes": [], "total": episodes}
        threading.Thread(target=self._run_benchmark, args=(episodes,),
                         daemon=True).start()
        return True

    def _run_benchmark(self, episodes: int) -> None:
        for ep in range(episodes):
            env = SnakeEnv(seed=2000 + ep)
            state = env.reset()
            done = False
            info = {"score": 0, "steps": 0}
            while not done:  # the hunger limit bounds every episode
                with torch.no_grad():
                    q = self.model(torch.from_numpy(state))
                _, done, info = env.step(int(q.argmax()))
                if not done:
                    state = env.state()
            with self.lock:
                self.benchmark["episodes"].append(
                    {"score": info["score"], "steps": info["steps"]}
                )
        with self.lock:
            self.benchmark["running"] = False

    def benchmark_status(self) -> dict:
        with self.lock:
            status = {**self.benchmark}
            eps = status["episodes"]
            if eps:
                scores = [e["score"] for e in eps]
                status["mean_score"] = round(float(np.mean(scores)), 1)
                status["max_score"] = max(scores)
                status["mean_steps"] = round(
                    float(np.mean([e["steps"] for e in eps])), 0
                )
            return status


SESSION = AgentSession()
SNAKE = SnakeSession()


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):  # quiet
        pass

    def _json(self, payload, code: int = 200) -> None:
        body = json.dumps(payload).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _body(self) -> dict:
        length = int(self.headers.get("Content-Length") or 0)
        if not length:
            return {}
        try:
            return json.loads(self.rfile.read(length))
        except json.JSONDecodeError:
            return {}

    def _html(self, filename: str) -> None:
        with open(os.path.join(WEBUI_DIR, filename), "rb") as f:
            body = f.read()
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            self._html("index.html")
        elif self.path in ("/snake", "/snake.html"):
            self._html("snake.html")
        elif self.path == "/api/state":
            self._json({**SESSION.state(), "checkpoints": find_checkpoints()})
        elif self.path == "/api/checkpoints":
            self._json({"checkpoints": find_checkpoints()})
        elif self.path == "/api/benchmark/status":
            self._json(SESSION.benchmark_status())
        elif self.path == "/api/snake/state":
            self._json({**SNAKE.state(), "checkpoints": find_snake_checkpoints()})
        elif self.path == "/api/snake/benchmark/status":
            self._json(SNAKE.benchmark_status())
        else:
            self._json({"error": "not found"}, 404)

    def do_POST(self):
        try:
            if self.path == "/api/reset":
                body = self._body()
                checkpoint = body.get("checkpoint")
                if checkpoint:
                    if checkpoint not in find_checkpoints():
                        self._json({"error": "unknown checkpoint"}, 400)
                        return
                    SESSION.load(checkpoint)
                if SESSION.model is None:
                    self._json({"error": "no checkpoint loaded"}, 400)
                    return
                self._json(SESSION.reset())
            elif self.path == "/api/step":
                if SESSION.model is None:
                    self._json({"error": "no checkpoint loaded"}, 400)
                    return
                self._json(SESSION.step())
            elif self.path == "/api/benchmark/start":
                body = self._body()
                episodes = max(1, min(int(body.get("episodes", 5)), 50))
                max_pieces = max(50, min(int(body.get("max_pieces", 500)), 5000))
                ok = SESSION.start_benchmark(episodes, max_pieces)
                self._json({"started": ok})
            elif self.path == "/api/snake/reset":
                body = self._body()
                checkpoint = body.get("checkpoint")
                if checkpoint:
                    if checkpoint not in find_snake_checkpoints():
                        self._json({"error": "unknown checkpoint"}, 400)
                        return
                    SNAKE.load(checkpoint)
                if SNAKE.model is None:
                    self._json({"error": "no checkpoint loaded"}, 400)
                    return
                self._json(SNAKE.reset())
            elif self.path == "/api/snake/step":
                if SNAKE.model is None:
                    self._json({"error": "no checkpoint loaded"}, 400)
                    return
                self._json(SNAKE.step())
            elif self.path == "/api/snake/benchmark/start":
                body = self._body()
                episodes = max(1, min(int(body.get("episodes", 10)), 100))
                ok = SNAKE.start_benchmark(episodes)
                self._json({"started": ok})
            else:
                self._json({"error": "not found"}, 404)
        except Exception as exc:  # surface errors to the UI instead of dying
            self._json({"error": str(exc)}, 500)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="checkpoint to load (default: newest under runs/ or models/)")
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    checkpoint = args.checkpoint
    if checkpoint is None:
        found = find_checkpoints()
        checkpoint = found[0] if found else None
    if checkpoint:
        SESSION.load(checkpoint)
        print(f"loaded tetris checkpoint {checkpoint}")
    else:
        print("no tetris checkpoint found - train one or pick from the UI")

    snake_found = find_snake_checkpoints()
    if snake_found:
        SNAKE.load(snake_found[0])
        print(f"loaded snake checkpoint {snake_found[0]}")
    else:
        print("no snake checkpoint found - train one with `uv run python -m games.snake.train`")

    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"Agent UI on http://{args.host}:{args.port}  "
          f"(Tetris: /  Snake: /snake)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
