import numpy as np

from snake_rl.env import STATE_DIM, SnakeEnv
from snake_rl.game import DOWN, LEFT, RIGHT, UP, SnakeGame


def test_movement_and_growth():
    game = SnakeGame(seed=0)
    head = game.snake[0]
    game.food = (head[0], head[1] + 1)  # plant food directly ahead
    events = game.step(RIGHT)
    assert events["ate"] and not events["died"]
    assert game.score == 1
    assert len(game.snake) == 4
    events = game.step(RIGHT)
    assert len(game.snake) == 4  # no growth without food


def test_cannot_reverse():
    game = SnakeGame(seed=0)
    assert game.direction == RIGHT
    game.step(LEFT)  # ignored: keeps moving right
    assert game.direction == RIGHT
    assert not game.game_over


def test_wall_death():
    game = SnakeGame(seed=0)
    for _ in range(game.width):
        events = game.step(RIGHT)
        if events["died"]:
            break
    assert game.game_over


def test_self_collision():
    game = SnakeGame(seed=0)
    game.food = (0, 0)  # keep food out of the way
    game.snake.append((game.snake[-1][0], game.snake[-1][1] - 1))  # length 4
    game.step(UP)
    game.step(LEFT)
    events = game.step(DOWN)  # loops back into its own body
    assert events["died"]


def test_env_state_shape_and_danger_flags():
    env = SnakeEnv(seed=0)
    state = env.reset()
    assert state.shape == (STATE_DIM,)
    assert state.dtype == np.float32
    # heading right from the middle of an empty board: no dangers
    assert state[:3].tolist() == [0.0, 0.0, 0.0]
    # direction one-hot says right
    assert state[3:7].tolist() == [0.0, 1.0, 0.0, 0.0]


def test_env_rewards_and_termination():
    env = SnakeEnv(seed=0)
    env.reset()
    head = env.game.snake[0]
    env.game.food = (head[0], head[1] + 1)
    reward, done, info = env.step(0)  # straight into the food
    assert reward == 10.0 and not done and info["score"] == 1
    # drive straight into the right wall
    done = False
    while not done:
        reward, done, info = env.step(0)
    assert reward == -10.0
    assert env.game.game_over


def test_hunger_termination():
    env = SnakeEnv(seed=0, hunger_factor=2)  # starve after 2*len steps
    env.reset()
    env.game.food = (0, 0)  # unreachable by circling
    done = False
    steps = 0
    while not done and steps < 100:
        # circle forever: alternate right turns to stay alive
        reward, done, info = env.step(1)
        steps += 1
    assert done and reward == -10.0
