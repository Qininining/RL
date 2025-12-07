"""Simple Q-learning trainer for the snake game environment."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from snake_game.config import EnvironmentConfig, TrainingConfig
from snake_game.envs.snake_env import SnakeEnv


@dataclass
class TrainingResult:
    rewards: List[float]
    q_table: np.ndarray
    success_rate: float
    model_path: Path


def create_env(env_cfg: EnvironmentConfig, *, render_mode: str | None = None) -> SnakeEnv:
    return SnakeEnv(
        size=env_cfg.size,
        max_steps=env_cfg.max_steps,
        render_mode=render_mode,
    )


def epsilon_greedy_action(q_values: np.ndarray, epsilon: float, action_space_n: int, rng: np.random.Generator) -> int:
    if rng.random() < epsilon:
        return int(rng.integers(0, action_space_n))
    return int(np.argmax(q_values))


def train_q_learning(
    env_cfg: EnvironmentConfig,
    train_cfg: TrainingConfig,
    *,
    output_dir: Path | None = None,
) -> TrainingResult:
    env = create_env(env_cfg, render_mode=train_cfg.render_mode)
    
    # Q-table shape: (size, size, size, size, 4) -> (head_x, head_y, food_x, food_y, actions)
    q_table_shape = (env_cfg.size, env_cfg.size, env_cfg.size, env_cfg.size, env.action_space.n)
    q_table = np.zeros(q_table_shape, dtype=np.float32)
    
    epsilon = train_cfg.epsilon_start
    rng = np.random.default_rng(train_cfg.seed)
    rewards: List[float] = []
    success_counter = 0 # Count episodes where snake ate at least one food

    if output_dir is None:
        output_dir = Path("runs")
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / "q_table.npy"

    for episode in range(train_cfg.episodes):
        obs, _ = env.reset(seed=(train_cfg.seed + episode) if train_cfg.seed is not None else None)
        state = tuple(int(x) for x in obs)
        total_reward = 0.0
        food_eaten = 0

        for _ in range(env_cfg.max_steps):
            action = epsilon_greedy_action(q_table[state], epsilon, env.action_space.n, rng)
            next_obs, reward, terminated, truncated, _ = env.step(action)
            next_state = tuple(int(x) for x in next_obs)

            # Q-learning update
            # Handle terminal state: if terminated, max Q(next) is 0
            best_next = 0.0 if terminated else np.max(q_table[next_state])
            
            td_target = reward + train_cfg.discount_factor * best_next
            td_error = td_target - q_table[state][action]
            q_table[state][action] += train_cfg.learning_rate * td_error

            state = next_state
            total_reward += reward
            
            if reward > 1.0: # Food eaten
                food_eaten += 1

            if terminated or truncated:
                if food_eaten > 0:
                    success_counter += 1
                break

        rewards.append(total_reward)
        epsilon = max(epsilon * train_cfg.epsilon_decay, train_cfg.epsilon_end)

        if (episode + 1) % train_cfg.log_interval == 0:
            recent = rewards[-train_cfg.log_interval :]
            avg_reward = float(np.mean(recent)) if recent else 0.0
            print(
                f"Episode {episode + 1:04d} | avg_reward={avg_reward:.2f} | "
                f"epsilon={epsilon:.3f} | food_eaten_rate={success_counter/(episode+1):.2%}"
            )

    np.save(model_path, q_table)
    env.close()

    success_rate = success_counter / max(train_cfg.episodes, 1)
    return TrainingResult(
        rewards=rewards,
        q_table=q_table,
        success_rate=success_rate,
        model_path=model_path,
    )


__all__ = ["TrainingResult", "train_q_learning", "create_env"]
