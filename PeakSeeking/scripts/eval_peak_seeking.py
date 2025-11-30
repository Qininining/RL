"""Entry point to evaluate a trained Q-learning agent on the peak seeking environment."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import yaml

from peak_seeking.config import ProjectConfig, parse_config_from_dict
from peak_seeking.training.q_learning import create_env

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "peak_seeking.yaml"
RUNS_DIR = PROJECT_ROOT / "runs"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained Q-learning agent.")
    parser.add_argument(
        "--config",
        type=str,
        default=str(DEFAULT_CONFIG_PATH),
        help="Path to the YAML config file.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Path to the trained Q-table (.npy file). If not provided, the latest run will be used.",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=None,
        help="Number of evaluation episodes. Defaults to config value.",
    )
    parser.add_argument(
        "--render",
        action="store_true",
        help="Force human rendering mode (overrides config).",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=0.1,
        help="Delay between steps for visualization (seconds).",
    )
    return parser.parse_args()


def load_config(path: str | Path) -> ProjectConfig:
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as file:
        raw = yaml.safe_load(file) or {}
    return parse_config_from_dict(raw)


def evaluate(cfg: ProjectConfig, model_path: Path, episodes: int | None, render_override: bool, delay: float) -> None:
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    q_table = np.load(model_path)
    print(f"Loaded Q-table from {model_path}")

    # Determine render mode and episodes
    render_mode = "human" if render_override else cfg.evaluation.render_mode
    num_episodes = episodes if episodes is not None else cfg.evaluation.episodes

    env = create_env(cfg.env, render_mode=render_mode)

    success_count = 0
    total_rewards = []

    for episode in range(num_episodes):
        obs, _ = env.reset()
        state = tuple(int(x) for x in obs)
        episode_reward = 0.0
        done = False

        print(f"Episode {episode + 1}/{num_episodes} started...")

        while not done:
            # Greedy action selection
            action = int(np.argmax(q_table[state]))

            next_obs, reward, terminated, truncated, _ = env.step(action)
            next_state = tuple(int(x) for x in next_obs)

            state = next_state
            episode_reward += reward

            if render_mode == "human":
                env.render()
                time.sleep(delay)

            if terminated or truncated:
                done = True
                # Assuming positive reward at the end means success/finding peak
                # You might need to adjust this condition based on your specific reward structure
                if reward > 0:
                    success_count += 1

        total_rewards.append(episode_reward)
        print(f"Episode {episode + 1} finished. Reward: {episode_reward:.2f}")

    env.close()

    avg_reward = np.mean(total_rewards) if total_rewards else 0.0
    success_rate = success_count / num_episodes if num_episodes > 0 else 0.0
    print("\nEvaluation Results:")
    print(f"Average Reward: {avg_reward:.2f}")
    print(f"Success Rate: {success_rate:.2%}")


def get_latest_model_path(runs_dir: Path) -> Path:
    if not runs_dir.exists():
        raise FileNotFoundError(f"Runs directory not found: {runs_dir}")

    # Find all q_table.npy files in subdirectories
    candidates = list(runs_dir.glob("*/q_table.npy"))
    
    # Also check for old style model in root of runs
    old_style = runs_dir / "q_table.npy"
    if old_style.exists():
        candidates.append(old_style)

    if not candidates:
        raise FileNotFoundError(f"No model files found in {runs_dir}")

    # Sort by modification time, newest first
    latest_model = max(candidates, key=lambda p: p.stat().st_mtime)
    return latest_model


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)

    model_path = Path(args.model) if args.model else None

    if model_path is None:
        try:
            model_path = get_latest_model_path(RUNS_DIR)
            print(f"No model specified. Using latest found: {model_path}")
        except FileNotFoundError as e:
            print(f"Error: {e}")
            return

    evaluate(cfg, model_path, args.episodes, args.render, args.delay)


if __name__ == "__main__":
    main()
