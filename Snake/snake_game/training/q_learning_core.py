"""Generic Q-learning agent implementation."""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np


class QLearningAgent:
    """
    A generic tabular Q-learning agent.
    """

    def __init__(
        self,
        observation_shape: Tuple[int, ...],
        action_space_n: int,
        learning_rate: float,
        discount_factor: float,
        seed: int | None = None,
    ) -> None:
        """
        Initialize the Q-learning agent.

        Args:
            observation_shape: The shape of the observation space (excluding action dimension).
            action_space_n: The number of discrete actions available.
            learning_rate: The learning rate (alpha).
            discount_factor: The discount factor (gamma).
            seed: Random seed for action selection.
        """
        # Initialize Q-table with zeros. Shape is (*obs_shape, n_actions)
        self.q_table = np.zeros(observation_shape + (action_space_n,), dtype=np.float32)
        self.action_space_n = action_space_n
        self.lr = learning_rate
        self.gamma = discount_factor
        self.rng = np.random.default_rng(seed)

    def get_action(self, state: Tuple[int, ...], epsilon: float) -> int:
        """
        Select an action using epsilon-greedy strategy.

        Args:
            state: The current state (tuple of indices).
            epsilon: The probability of choosing a random action.

        Returns:
            The selected action index.
        """
        if self.rng.random() < epsilon:
            return int(self.rng.integers(0, self.action_space_n))
        return int(np.argmax(self.q_table[state]))

    def update(
        self,
        state: Tuple[int, ...],
        action: int,
        reward: float,
        next_state: Tuple[int, ...],
        terminated: bool,
    ) -> None:
        """
        Update the Q-table using the Q-learning update rule.

        Args:
            state: The previous state.
            action: The action taken.
            reward: The reward received.
            next_state: The new state after taking the action.
            terminated: Whether the episode terminated after this step.
        """
        # If terminated, the value of the next state is 0.
        # Otherwise, it's the max Q-value for the next state.
        best_next = 0.0 if terminated else np.max(self.q_table[next_state])
        
        td_target = reward + self.gamma * best_next
        td_error = td_target - self.q_table[state][action]
        
        self.q_table[state][action] += self.lr * td_error

    def save(self, path: Path) -> None:
        """Save the Q-table to a file."""
        np.save(path, self.q_table)

    def load(self, path: Path) -> None:
        """Load the Q-table from a file."""
        self.q_table = np.load(path)
