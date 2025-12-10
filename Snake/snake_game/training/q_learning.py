"""Simple Q-learning trainer for the snake game environment."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List

import numpy as np

from snake_game.config import EnvironmentConfig, TrainingConfig
from snake_game.envs.snake_env import SnakeEnv
from snake_game.training.q_learning_core import QLearningAgent


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


def train_q_learning(
    env_cfg: EnvironmentConfig,
    train_cfg: TrainingConfig,
    *,
    output_dir: Path | None = None,
) -> TrainingResult:
    env = create_env(env_cfg, render_mode=train_cfg.render_mode)
    
    # Initialize generic agent
    # Q-table shape: (size, size, size, size, 4) -> (head_x, head_y, food_x, food_y, actions)
    agent = QLearningAgent(
        observation_shape=(env_cfg.size, env_cfg.size, env_cfg.size, env_cfg.size),
        action_space_n=env.action_space.n,
        learning_rate=train_cfg.learning_rate,
        discount_factor=train_cfg.discount_factor,
        seed=train_cfg.seed,
    )
    
    epsilon = train_cfg.epsilon_start
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
            action = agent.get_action(state, epsilon)
            next_obs, reward, terminated, truncated, _ = env.step(action)
            next_state = tuple(int(x) for x in next_obs)

            agent.update(state, action, reward, next_state, terminated)

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

    agent.save(model_path)
    env.close()

    success_rate = success_counter / max(train_cfg.episodes, 1)
    return TrainingResult(
        rewards=rewards,
        q_table=agent.q_table,
        success_rate=success_rate,
        model_path=model_path,
    )


__all__ = ["TrainingResult", "train_q_learning", "create_env"]
