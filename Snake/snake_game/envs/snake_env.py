"""Custom Gymnasium environment for Snake game."""

from __future__ import annotations

from typing import Optional, List, Tuple

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
from gymnasium import spaces


class SnakeEnv(gym.Env):
    """A simple Snake environment."""

    metadata = {"render_modes": ["human"], "render_fps": 10}

    def __init__(
        self,
        *,
        size: int = 6,
        max_steps: int = 200,
        render_mode: Optional[str] = None,
    ) -> None:
        super().__init__()
        self.size = size
        self.max_steps = max_steps
        self.render_mode = render_mode

        # Actions: 0: Up, 1: Right, 2: Down, 3: Left
        self.action_space = spaces.Discrete(4)
        
        # Observation: (head_x, head_y, food_x, food_y)
        # This is a simplified state space for Q-table learning on small grids.
        self.observation_space = spaces.Box(
            low=0,
            high=self.size - 1,
            shape=(4,),
            dtype=np.int32,
        )

        self.snake: List[np.ndarray] = []
        self.food: np.ndarray | None = None
        self.step_count = 0
        
        self._fig: Optional[plt.Figure] = None
        self._ax: Optional[plt.Axes] = None

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        
        # Initialize snake in the middle
        start_x = self.size // 2
        start_y = self.size // 2
        self.snake = [np.array([start_x, start_y], dtype=np.int32)]
        
        self._spawn_food()
        self.step_count = 0

        if self.render_mode == "human":
            self._render_frame()

        return self._get_obs(), {}

    def _spawn_food(self) -> None:
        while True:
            food = np.array(
                [
                    self.np_random.integers(0, self.size),
                    self.np_random.integers(0, self.size),
                ],
                dtype=np.int32,
            )
            # Ensure food doesn't spawn on snake
            collision = False
            for segment in self.snake:
                if np.array_equal(segment, food):
                    collision = True
                    break
            if not collision:
                self.food = food
                break

    def step(self, action: int):
        self.step_count += 1
        
        head = self.snake[0].copy()
        
        # 0: Up (x-1), 1: Right (y+1), 2: Down (x+1), 3: Left (y-1)
        # Note: Using matrix coordinates (row, col) -> (x, y)
        if action == 0:
            head[0] -= 1
        elif action == 1:
            head[1] += 1
        elif action == 2:
            head[0] += 1
        elif action == 3:
            head[1] -= 1
            
        # Check collisions
        terminated = False
        reward = -0.1 # Small penalty for each step to encourage speed
        
        # Wall collision
        if not (0 <= head[0] < self.size and 0 <= head[1] < self.size):
            terminated = True
            reward = -10.0
        # Self collision
        else:
            for segment in self.snake[:-1]: # Ignore tail as it will move
                if np.array_equal(segment, head):
                    terminated = True
                    reward = -10.0
                    break
        
        if not terminated:
            self.snake.insert(0, head)
            
            # Check food
            if np.array_equal(head, self.food):
                reward = 10.0
                self._spawn_food()
                # Don't pop tail, snake grows
            else:
                self.snake.pop()
                
        truncated = self.step_count >= self.max_steps
        
        if self.render_mode == "human":
            self._render_frame()
            
        return self._get_obs(), reward, terminated, truncated, {}

    def _get_obs(self) -> np.ndarray:
        # Return (head_x, head_y, food_x, food_y)
        head = self.snake[0]
        food = self.food if self.food is not None else np.zeros(2, dtype=np.int32)
        return np.concatenate([head, food])

    def render(self):
        if self.render_mode == "human":
            return self._render_frame()

    def _render_frame(self) -> None:
        if self._fig is None or self._ax is None:
            self._fig, self._ax = plt.subplots()
            
        self._ax.clear()
        
        # Draw grid
        grid = np.zeros((self.size, self.size))
        self._ax.imshow(grid, cmap="Greys", vmin=0, vmax=1, extent=[0, self.size, self.size, 0])
        
        # Draw snake
        for i, segment in enumerate(self.snake):
            color = "green" if i == 0 else "lime" # Head is darker green
            self._ax.add_patch(plt.Rectangle((segment[1], segment[0]), 1, 1, color=color))
            
        # Draw food
        if self.food is not None:
            self._ax.add_patch(plt.Rectangle((self.food[1], self.food[0]), 1, 1, color="red"))
            
        self._ax.set_xlim(0, self.size)
        self._ax.set_ylim(self.size, 0) # Invert y axis to match matrix coordinates
        self._ax.set_title(f"Step {self.step_count} | Score: {len(self.snake) - 1}")
        self._ax.grid(True, color='gray', linestyle='-', linewidth=0.5)
        
        plt.pause(0.05)

    def close(self) -> None:
        if self._fig is not None:
            plt.close(self._fig)
            self._fig = None
            self._ax = None
        super().close()


__all__ = ["SnakeEnv"]
