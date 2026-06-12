"""Parallel Snake environments, one per worker process (same design as
``tetris_rl.vec_env``): workers simulate, the main process owns the network."""

from __future__ import annotations

import multiprocessing as mp

from snake_rl.env import SnakeEnv


def _worker(remote, parent_remote, seed: int, width: int, height: int) -> None:
    parent_remote.close()
    env = SnakeEnv(seed=seed, width=width, height=height)
    try:
        while True:
            cmd, data = remote.recv()
            if cmd == "reset":
                remote.send(env.reset())
            elif cmd == "step":
                reward, done, info = env.step(data)
                state = env.reset() if done else env.state()
                remote.send((reward, done, info, state))
            elif cmd == "close":
                break
    except (EOFError, KeyboardInterrupt):
        pass
    finally:
        remote.close()


class ParallelSnake:
    def __init__(self, num_envs: int, base_seed: int = 0,
                 width: int = 12, height: int = 12):
        self.num_envs = num_envs
        ctx = mp.get_context("fork")
        self._remotes = []
        self._procs = []
        for i in range(num_envs):
            parent, child = ctx.Pipe()
            proc = ctx.Process(
                target=_worker,
                args=(child, parent, base_seed + i, width, height),
                daemon=True,
            )
            proc.start()
            child.close()
            self._remotes.append(parent)
            self._procs.append(proc)

    def reset(self) -> list:
        for remote in self._remotes:
            remote.send(("reset", None))
        return [remote.recv() for remote in self._remotes]

    def step(self, actions) -> list:
        """Returns ``[(reward, done, info, next_state), ...]``."""
        for remote, action in zip(self._remotes, actions):
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
