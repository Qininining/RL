"""Tests for the Snake environment."""

import numpy as np
import pytest

from snake_game.envs.snake_env import SnakeEnv


def test_env_initialization():
    env = SnakeEnv(size=10)
    assert env.size == 10
    assert env.action_space.n == 4
    assert env.observation_space.shape == (4,)


def test_reset():
    env = SnakeEnv(size=10)
    obs, _ = env.reset()
    assert obs.shape == (4,)
    # Snake starts in middle
    assert obs[0] == 5
    assert obs[1] == 5
    # Food should be within bounds
    assert 0 <= obs[2] < 10
    assert 0 <= obs[3] < 10


def test_step():
    env = SnakeEnv(size=10)
    env.reset()
    # Move Up (0)
    obs, reward, terminated, truncated, _ = env.step(0)
    assert obs.shape == (4,)
    assert isinstance(reward, float)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
