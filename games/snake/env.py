"""RL environment over Snake with relative actions and compact features.

State (14 features):

    [danger_straight, danger_right, danger_left,
     dir_up, dir_right, dir_down, dir_left,
     food_up, food_right, food_down, food_left,
     freespace_straight, freespace_right, freespace_left]

The three freespace features are flood-fill reachable area (as a fraction of
the board) from each candidate next-head. They give the agent the body/space
awareness the classic 11-feature state lacks, so it can avoid sealing itself
into a pocket when long (the self-trapping failure).

Actions: 0 = keep going straight, 1 = turn right, 2 = turn left.
Reward: +10 for eating, -10 for dying (or starving past the hunger limit).
"""

from __future__ import annotations

import numpy as np

from .game import SnakeGame

STATE_DIM = 14
NUM_ACTIONS = 3

# action -> change of absolute direction (mod 4): straight, right turn, left turn
_TURN = {0: 0, 1: 1, 2: 3}


class SnakeEnv:
    state_dim = STATE_DIM
    num_actions = NUM_ACTIONS

    def __init__(self, seed: int | None = None, width: int = 12, height: int = 12,
                 hunger_factor: int = 100):
        self.game = SnakeGame(seed=seed, width=width, height=height)
        self.hunger_factor = hunger_factor

    def reset(self) -> np.ndarray:
        self.game.reset()
        return self.state()

    def state(self) -> np.ndarray:
        game = self.game
        d = game.direction
        head_r, head_c = game.snake[0]
        food_r, food_c = game.food if game.food else (head_r, head_c)
        next_heads = [game.next_head((d + _TURN[a]) % 4) for a in range(3)]
        dangers = [game.collides(h) for h in next_heads]
        area = game.width * game.height
        freespace = [game.free_space(h) / area for h in next_heads]
        return np.array(
            [
                *[float(x) for x in dangers],
                float(d == 0), float(d == 1), float(d == 2), float(d == 3),
                float(food_r < head_r),  # food up
                float(food_c > head_c),  # food right
                float(food_r > head_r),  # food down
                float(food_c < head_c),  # food left
                *freespace,
            ],
            dtype=np.float32,
        )

    def step(self, action: int):
        game = self.game
        direction = (game.direction + _TURN[int(action)]) % 4
        events = game.step(direction)
        reward = 0.0
        done = False
        if events["died"]:
            reward = -10.0
            done = True
        elif events["ate"]:
            reward = 10.0
            if game.game_over:  # board completely filled
                done = True
        if not done and game.steps_since_food > self.hunger_factor * len(game.snake):
            # starving in a loop: end the episode like a death
            reward = -10.0
            done = True
            game.game_over = True
        info = {"score": game.score, "steps": game.steps}
        return reward, done, info
