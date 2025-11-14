"""Entry point to train a Q-learning agent on the peak seeking environment."""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from peak_seeking.config import ProjectConfig, parse_config_from_dict
from peak_seeking.training.q_learning import train_q_learning


DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent.parent / "configs" / "peak_seeking.yaml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a Q-learning agent for the PeakSeekingEnv.")
    parser.add_argument(
        "--config",
        type=str,
        default=str(DEFAULT_CONFIG_PATH),
        help="Path to the YAML config file controlling env/training parameters.",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=None,
        help="Override the number of training episodes declared in the config file.",
    )
    parser.add_argument(
        "--render",
        action="store_true",
        help="Enable human rendering mode while training (slows execution).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="runs",
        help="Directory to store artifacts such as the learned Q-table.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Override the random seed defined in the config file.",
    )
    return parser.parse_args()


def apply_overrides(config: ProjectConfig, args: argparse.Namespace) -> ProjectConfig:
    if args.episodes is not None:
        config.training.episodes = args.episodes
    if args.seed is not None:
        config.training.seed = args.seed
    if args.render:
        config.env.render_mode = "human"
    return config


def load_config(path: str | Path) -> ProjectConfig:
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as file:
        raw = yaml.safe_load(file) or {}
    return parse_config_from_dict(raw)


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    cfg = apply_overrides(cfg, args)
    result = train_q_learning(cfg.env, cfg.training, output_dir=Path(args.output_dir))
    print(f"Training complete. Success rate: {result.success_rate:.2%}. Model saved to {result.model_path}.")


if __name__ == "__main__":
    main()
