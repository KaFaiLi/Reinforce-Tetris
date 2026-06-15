"""Parallel Tetris environments, one per worker process.

Workers run the (numpy-only) game simulation and candidate-feature
computation; the main process owns the network and picks actions for all
environments in a single batched forward pass. Communication is over pipes,
SubprocVecEnv-style. On episode end a worker auto-resets and returns the
fresh candidate set together with the final episode stats.
"""

from __future__ import annotations

import multiprocessing as mp

from tetris_rl.env import TetrisPlacementEnv


def _worker(remote, parent_remote, seed: int, game_over_penalty: float) -> None:
    parent_remote.close()
    env = TetrisPlacementEnv(seed=seed, game_over_penalty=game_over_penalty)
    try:
        while True:
            cmd, data = remote.recv()
            if cmd == "reset":
                remote.send(env.reset())
            elif cmd == "step":
                reward, done, info = env.step(data)
                feats = env.reset() if done else env.candidate_features()
                remote.send((reward, done, info, feats))
            elif cmd == "close":
                break
    except (EOFError, KeyboardInterrupt):
        pass
    finally:
        remote.close()


class ParallelTetris:
    def __init__(self, num_envs: int, base_seed: int = 0,
                 game_over_penalty: float = -5.0):
        self.num_envs = num_envs
        ctx = mp.get_context("fork")
        self._remotes = []
        self._procs = []
        for i in range(num_envs):
            parent, child = ctx.Pipe()
            proc = ctx.Process(
                target=_worker,
                args=(child, parent, base_seed + i, game_over_penalty),
                daemon=True,
            )
            proc.start()
            child.close()
            self._remotes.append(parent)
            self._procs.append(proc)

    def reset(self) -> list:
        """Returns one candidate-feature matrix per environment."""
        for remote in self._remotes:
            remote.send(("reset", None))
        return [remote.recv() for remote in self._remotes]

    def step(self, action_indices) -> list:
        """Returns ``[(reward, done, info, next_candidate_feats), ...]``."""
        for remote, action in zip(self._remotes, action_indices):
            remote.send(("step", int(action)))
        return [remote.recv() for remote in self._remotes]

    def close(self) -> None:
        for remote in self._remotes:
            try:
                remote.send(("close", None))
            except BrokenPipeError:
                pass
        for proc in self._procs:
            proc.join(timeout=2)
        for remote in self._remotes:
            remote.close()
