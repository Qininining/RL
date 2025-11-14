"""Custom Gymnasium environment for a simple peak-seeking task."""

from __future__ import annotations

from typing import Optional, Tuple

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
from gymnasium import spaces


class PeakSeekingEnv(gym.Env):
    """A grid-world style environment where the agent must reach the mountain peak."""

    metadata = {"render_modes": ["human"], "render_fps": 10}

    def __init__(
        self,
        *,
        size: int = 20,
        max_steps: int = 100,
        threshold_ratio: float = 0.98,
        peak_value_range: Tuple[float, float] = (5.0, 10.0),
        render_mode: Optional[str] = None,
    ) -> None:
        super().__init__()
        if size <= 1:
            raise ValueError("`size` must be greater than 1.")
        if not 0.0 < threshold_ratio <= 1.0:
            raise ValueError("`threshold_ratio` must be within (0, 1].")
        self.size = size
        self.max_steps = max_steps
        self.threshold_ratio = threshold_ratio
        self.peak_value_range = peak_value_range
        self.render_mode = render_mode

        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(
            low=0,
            high=self.size - 1,
            shape=(2,),
            dtype=np.int32,
        )

        self.height_map: np.ndarray | None = None
        self.global_max: float | None = None
        self.threshold: float | None = None
        self.agent_pos = np.zeros(2, dtype=np.int32)
        self.step_count = 0

        self._fig: Optional[plt.Figure] = None
        self._ax: Optional[plt.Axes] = None

    def _generate_height_map(self, peak_value: float) -> np.ndarray:
        x = np.linspace(-2, 2, self.size)
        y = np.linspace(-2, 2, self.size)
        x_grid, y_grid = np.meshgrid(x, y)
        return np.exp(-(x_grid**2 + y_grid**2)) * peak_value

    def _init_render(self) -> None:
        if self._fig is None or self._ax is None:
            self._fig, self._ax = plt.subplots()

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):  # type: ignore[override]
        super().reset(seed=seed)
        del options  # unused but part of API

        low, high = self.peak_value_range
        random_peak_value = self.np_random.uniform(low, high)
        self.height_map = self._generate_height_map(random_peak_value)
        self.global_max = float(self.height_map.max())
        self.threshold = self.global_max * self.threshold_ratio

        self.agent_pos = np.array(
            [
                self.np_random.integers(1, self.size - 1),
                self.np_random.integers(1, self.size - 1),
            ],
            dtype=np.int32,
        )
        self.step_count = 0

        if self.render_mode == "human":
            self._render_frame()

        return self._get_obs(), {}

    def step(self, action: int):  # type: ignore[override]
        if self.height_map is None:
            raise RuntimeError("Call `reset` before `step`.")

        self.step_count += 1
        if action == 0:
            self.agent_pos[0] = max(self.agent_pos[0] - 1, 0)
        elif action == 1:
            self.agent_pos[1] = min(self.agent_pos[1] + 1, self.size - 1)
        elif action == 2:
            self.agent_pos[0] = min(self.agent_pos[0] + 1, self.size - 1)
        elif action == 3:
            self.agent_pos[1] = max(self.agent_pos[1] - 1, 0)
        else:
            raise ValueError(f"Invalid action {action}.")

        current_height = float(self.height_map[self.agent_pos[0], self.agent_pos[1]])
        done = current_height >= (self.threshold or float("inf"))
        truncated = self.step_count >= self.max_steps
        reward = 1.0 if done else 0.0

        if self.render_mode == "human":
            self._render_frame()

        return self._get_obs(), reward, done, truncated, {}

    def _get_obs(self) -> np.ndarray:
        return self.agent_pos.copy()

    def _render_frame(self) -> None:
        if self.height_map is None:
            return
        self._init_render()
        assert self._ax is not None
        self._ax.clear()
        self._ax.imshow(self.height_map, cmap="viridis", extent=[0, self.size, 0, self.size])
        self._ax.plot(self.agent_pos[1], self.agent_pos[0], "ro")
        self._ax.set_title(
            f"Step {self.step_count} | Height: {self.height_map[self.agent_pos[0], self.agent_pos[1]]:.2f}"
        )
        plt.pause(0.1)

    def close(self) -> None:
        if self._fig is not None:
            plt.close(self._fig)
            self._fig = None
            self._ax = None
        super().close()


__all__ = ["PeakSeekingEnv"]
