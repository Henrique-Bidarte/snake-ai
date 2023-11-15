from collections import deque
import gymnasium
import numpy as np
import random
import cv2
import time
from gymnasium import spaces


def collision_with_apple(apple_position, score):
    apple_position = [random.randrange(1, 50) * 10, random.randrange(1, 50) * 10]
    return apple_position, score


def collision_with_boundaries(snake_head):
    if (
        snake_head[0] >= 500
        or snake_head[0] < 0
        or snake_head[1] >= 500
        or snake_head[1] < 0
    ):
        return 1
    else:
        return 0


def collision_with_self(snake_position):
    snake_head = snake_position[0]
    if snake_head in snake_position[1:]:
        return 1
    else:
        return 0


class SnakeCustomEnv(gymnasium.Env):
    metadata = {"render.modes": ["human"], "render_fps": 30}

    def __init__(self):
        super(SnakeCustomEnv, self).__init__()
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(
            low=-500, high=500, shape=(35,), dtype=np.int32
        )

    def get_observation(self):
        head_x = self.snake_head[0]
        head_y = self.snake_head[1]

        apple_delta_x = head_x - self.apple_position[0]
        apple_delta_y = head_y - self.apple_position[1]

        snake_length = len(self.snake_position)

        self.past_actions = deque(maxlen=30)
        for _ in range(30):
            self.past_actions.append(-1)

        return np.array(
            [head_x, head_y, apple_delta_x, apple_delta_y, snake_length]
            + list(self.past_actions)
        )

    def step(self, action):
        # self.render()
        if action == 0 and self.prev_button_direction != 1:
            self.snake_head[0] -= 10
            self.prev_button_direction = 0
        elif action == 1 and self.prev_button_direction != 0:
            self.snake_head[0] += 10
            self.prev_button_direction = 1
        elif action == 2 and self.prev_button_direction != 3:
            self.snake_head[1] += 10
            self.prev_button_direction = 2
        elif action == 3 and self.prev_button_direction != 2:
            self.snake_head[1] -= 10
            self.prev_button_direction = 3

        apple_reward = 0
        if self.snake_head == self.apple_position:
            self.apple_position, self.score = collision_with_apple(
                self.apple_position, self.score
            )
            self.snake_position.insert(0, list(self.snake_head))
            apple_reward = 10000

        else:
            self.snake_position.insert(0, list(self.snake_head))
            self.snake_position.pop()

        if (
            collision_with_boundaries(self.snake_head) == 1
            or collision_with_self(self.snake_position) == 1
        ):
            self.terminated = True
            self.truncated = True

        if self.terminated or self.truncated:
            self.reward = -10
        else:
            euclidean_dist_to_apple = np.linalg.norm(
                np.array(self.snake_head) - np.array(self.apple_position)
            )

            self.reward = (((250 - euclidean_dist_to_apple) + apple_reward) / 100) * (
                self.score + 1
            )

        self.info = {}
        self.past_actions.append(action)
        self.observation = self.get_observation()

        return (
            self.observation,
            self.reward,
            self.terminated,
            self.truncated,
            self.info,
        )

    def reset(self, seed=None, options=None):
        self.terminated = False
        self.truncated = False
        self.img = np.zeros((500, 500, 3), dtype="uint8")
        self.snake_position = [[250, 250], [240, 250], [230, 250]]
        self.apple_position = [
            random.randrange(1, 50) * 10,
            random.randrange(1, 50) * 10,
        ]
        self.score = 0
        self.prev_button_direction = 1
        self.snake_head = [250, 250]

        info = {}
        self.observation = self.get_observation()

        return np.array(self.observation), info

    def render(self):
        cv2.imshow("Snake_AI", self.img)
        cv2.waitKey(1)
        self.img = np.zeros((500, 500, 3), dtype="uint8")

        t_end = time.time() + 0.05
        while time.time() < t_end:
            continue

        cv2.rectangle(
            self.img,
            (self.apple_position[0], self.apple_position[1]),
            (self.apple_position[0] + 10, self.apple_position[1] + 10),
            (0, 0, 255),
            3,
        )

        for position in self.snake_position:
            cv2.rectangle(
                self.img,
                (position[0], position[1]),
                (position[0] + 10, position[1] + 10),
                (0, 255, 0),
                3,
            )
